import re
import json
import copy
import numpy as np
import asyncio
import random

from logging import getLogger
from typing import Awaitable, Callable, cast, List, Tuple

from typing_extensions import TypedDict, Unpack

from inspect_ai._util._async import is_callable_coroutine
from inspect_ai.model._cache import CachePolicy
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.model._chat_message import ChatMessage, ChatMessageTool, ChatMessageUser
from inspect_ai.model._model import get_model
from inspect_ai.scorer._metric import Score, ValueToFloat, value_to_float
from inspect_ai.scorer._score import score
from inspect_ai.solver._chain import chain
from inspect_ai.tool._tool import Tool, ToolResult, tool
from inspect_ai.tool._tool_with import tool_with
from inspect_ai.util._limit import token_limit as create_token_limit

from ._prompt import system_message
from ._solver import Generate, Solver, solver
from ._task_state import TaskState
from ._use_tools import use_tools

logger = getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = """
You are a helpful assistant attempting to submit the correct answer. You have several functions available to help with finding the answer. Each message may perform one function call. You will see the result of the function right after sending the message. If you need to perform multiple actions, you can always send more messages with subsequent function calls. Do some reasoning before your actions, describing what function calls you are going to use and how they fit into your plan.

When you have completed the task and have an answer, call the {submit}() function to report it. Example: <tool_call>\n{"name": "submit", "arguments": {"answer": "1860"}}\n</tool_call>
"""

DEFAULT_INCORRECT_MESSAGE = """
Your submission was incorrect. Please proceed and attempt to find the correct answer.
"""
DEFAULT_CONTINUE_MESSAGE = "Please proceed to the next step using your best judgement. Once you have the final answer, you MUST call the submit() tool with the answer as the argument."
DEFAULT_SUBMIT_NAME = "submit"
DEFAULT_SUBMIT_DESCRIPTION = "Submit an answer for evaluation."


class BasicAgentDeprecatedArgs(TypedDict, total=False):
    max_messages: int | None


@solver
def basic_agent_trajectory(
    *,
    init: Solver | list[Solver] | None = None,
    tools: list[Tool] | Solver | None = None,
    cache: bool | CachePolicy = False,
    max_attempts: int = 1,
    message_limit: int | None = None,
    token_limit: int | None = None,
    max_tool_output: int | None = None,
    score_value: ValueToFloat | None = None,
    incorrect_message: str
    | Callable[
        [TaskState, list[Score]], str | Awaitable[str]
    ] = DEFAULT_INCORRECT_MESSAGE,
    continue_message: str = DEFAULT_CONTINUE_MESSAGE,
    submit_name: str = DEFAULT_SUBMIT_NAME,
    submit_description: str = DEFAULT_SUBMIT_DESCRIPTION,
    submit_append: bool = False,
    num_initial_search: int = 8,
    num_samples_per_step: int = 8,
    **kwargs: Unpack[BasicAgentDeprecatedArgs],
) -> Solver:
    """Basic ReAct agent.

    Agent that runs a tool use loop until the model submits an answer using the
    `submit()` tool. Tailor the model's instructions by passing a `system_message()`
    and/or other steps to `init` (if no `init` is specified then a default system
    message will be used). Use `max_attempts` to support additional submissions if
    the initial submission(s) are incorrect.

    Submissions are evaluated using the task's main scorer, with value of 1.0
    indicating a correct answer. Scorer values are converted to float (e.g.
    "C" becomes 1.0) using the standard value_to_float() function. Provide an
    alternate conversion scheme as required via `score_value`.

    Args:
       init: Agent initialisation (defaults to system_message with basic ReAct prompt)
       tools: Tools available for the agent. Either a list of tools or a Solver that
          can yield dynamic tools per-sample.
       cache: Caching behaviour for generate responses (defaults to no caching).
       max_attempts: Maximum number of submissions to accept before terminating.
       message_limit: Limit on messages in sample before terminating agent.
          If not specified, will use limit_messages defined for the task. If there is none
          defined for the task and there is no `token_limit`, 50 will be used as a default.
       token_limit: Limit on tokens used in sample before terminating agent.
       max_tool_output: Maximum output length (in bytes).
          Defaults to max_tool_output from active GenerateConfig.
       score_value: Function used to extract float from scores (defaults
          to standard value_to_float())
       incorrect_message: User message reply for an incorrect submission from the model.
          Alternatively, a function which returns a message (function may optionally be async)
       continue_message: User message to urge the model to continue when it
          doesn't make a tool call.
       submit_name: Name for tool used to make submissions
          (defaults to 'submit')
       submit_description: Description of submit tool (defaults to
          'Submit an answer for evaluation')
       submit_append: Append the submit tool output to the model completion
           text (defaults to `False`, which means the submission overwrites
           the model completion).
       **kwargs: Deprecated arguments for backward compatibility.

    Returns:
        Plan for agent.
    """
    # resolve deprecated
    for arg, value in kwargs.items():
        if arg == "max_messages":
            # deprecated, don't warn yet
            message_limit = int(cast(int, value))

    # resolve init
    if init is None:
        init = system_message(DEFAULT_SYSTEM_MESSAGE, submit=submit_name)
    init = init if isinstance(init, list) else [init]

    # resolve tools
    if tools is None:
        tools = []
    tools = tools if isinstance(tools, Solver) else use_tools(tools, append=True)

    # resolve score_value function
    score_value_fn = score_value or value_to_float()
    # M value
    num_initial_search = num_initial_search
    num_samples_per_step = num_samples_per_step

    # submission tool
    @tool
    def submit() -> Tool:
        async def execute(answer: str) -> ToolResult:
            """Submit an answer for evaluation.

            Args:
              answer (str): Submitted answer
            """
            return answer

        return execute

    # solver that adds submission tool
    @solver
    def submit_tool() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.tools.append(tool_with(submit(), submit_name, submit_description))
            return state

        return solve

    # helper to extract a submitted answer
    def submission(tool_results: list[ChatMessage]) -> str | None:
        return next(
            (
                result.text
                for result in tool_results
                if isinstance(result, ChatMessageTool)
                and result.function == submit_name
            ),
            None,
        )

    async def rollout_from_state(
        base_state: TaskState, max_depth: int = None
    ) -> Tuple[str, float]:
        if max_depth is None:
            max_depth = 20

        # Create a deep copy of the state for rollout
        rollout_state = copy.deepcopy(base_state)
        depth = 0

        # Save the first candidate step for future use
        initial = True
        initial_state = []

        # Verify if the current state is about submitting answer
        # If so, simply return the argument.
        answer = submission([rollout_state.messages[-1]])
        if answer:
            rollout_state.output.completion = answer

            # Score the answer
            try:
                answer_scores = await score(rollout_state)
                score_val = score_value_fn(answer_scores[0].value)
                return answer, score_val, depth, initial_state
            except Exception as e:
                logger.warning(f"Scoring failed in rollout: {e}")
                return "", 0.0, depth, initial_state

        while depth < max_depth and not rollout_state.completed:
            try:
                # Generate next step
                rollout_state.output = await get_model().generate(
                    input=rollout_state.messages,
                    tools=rollout_state.tools,
                    cache=cache,
                )
                rollout_state.messages.append(rollout_state.output.message)

                # check for context window overflow
                if rollout_state.output.stop_reason == "model_length":
                    from inspect_ai.log._transcript import transcript

                    transcript().info("Agent terminated: model context window exceeded")
                    break

                depth += 1

                if initial:
                    initial_state.append(rollout_state.output.message)

                # Handle tool calls
                if rollout_state.output.message.tool_calls:
                    tool_results, _ = await execute_tools(
                        [rollout_state.output.message],
                        rollout_state.tools,
                        max_output=max_tool_output,
                    )
                    rollout_state.messages.extend(tool_results)

                    if initial:
                        initial_state.extend(tool_results)

                    # Check for answer
                    answer = submission(tool_results)
                    if answer:
                        rollout_state.output.completion = answer

                        # Score the answer
                        try:
                            answer_scores = await score(rollout_state)
                            score_val = score_value_fn(answer_scores[0].value)
                            return answer, score_val, depth, initial_state
                        except Exception as e:
                            logger.warning(f"Scoring failed in rollout: {e}")
                            return "", 0.0, depth, initial_state
                else:
                    # No tool calls, add continue message
                    rollout_state.messages.append(
                        ChatMessageUser(content=continue_message)
                    )

                    if initial:
                        initial_state.append(ChatMessageUser(content=continue_message))

                initial = False

            except Exception as e:
                logger.warning(f"Error in rollout at depth {depth}: {e}")
                break

        return "", 0.0, depth, initial_state

    async def evaluate_step_candidates(
        base_state: TaskState,
        initial: bool,
    ):
        # Sample M rollouts from the current step
        M = num_initial_search if initial else num_samples_per_step
        rollout_tasks = [rollout_from_state(base_state) for _ in range(M)]

        try:
            # Execute rollouts concurrently
            sample_results = await asyncio.gather(
                *rollout_tasks, return_exceptions=True
            )

            # Process results
            valid_results = []
            for result in sample_results:
                if isinstance(result, Exception):
                    logger.warning(f"Rollout failed: {result}")
                else:
                    valid_results.append(result)

            # Calculate overall acurracy
            answers, scores, depths, next_steps = (
                zip(*valid_results) if valid_results else ([], [], [], [])
            )

            return answers, scores, depths, next_steps

        except Exception as e:
            logger.error(f"Failed to evaluate current step: {e}")

    def simple_extract(answers, accs, depths, next_steps):
        # Find successful indices (acc = 1.0)
        success_idx = [i for i, acc in enumerate(accs) if acc == 1.0]

        assert len(success_idx) != 0

        # Select the best path
        best_idx = min(success_idx, key=lambda i: depths[i])
        best_anwser = answers[best_idx]
        best_depth = depths[best_idx]
        best_step = next_steps[best_idx]

        # Select randomly for exploration, excluding for the steps with the same depth and acc.
        excluded_indices = [
            idx
            for idx, (acc, depth) in enumerate(zip(accs, depths))
            if acc == 1.0 and depth == best_depth
        ]
        other_steps = [
            step for i, step in enumerate(next_steps) if i not in excluded_indices
        ]
        # In case where all samples have the same reasoning length and acc.
        if len(other_steps) == 0:
            other_steps = [i for i in range(len(accs)) if i != best_idx]

        random_step = random.choice(other_steps)

        return best_anwser, best_depth, best_step, random_step

    # main agent loop
    @solver
    def basic_agent_loop() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # resolve message_limit -- prefer parameter then fall back to task.
            # if there is no message limit AND no token limit then provide
            # a default message limit of 50 (so that the task can't run forever
            # if the model never submits)
            state.message_limit = message_limit or state.message_limit
            if state.message_limit is None and token_limit is None:
                state.message_limit = 50

            state.metadata = {
                "start_state": None,
                "win_sample": [],
                "lose_sample": [],
            }

            sub_state = None
            initial = True

            with create_token_limit(token_limit):
                # main loop
                while not state.completed:
                    current_step_index = len(state.messages)

                    # Evaluate the overall acc of the current state
                    (
                        answers,
                        accs,
                        depths,
                        next_steps,
                    ) = await evaluate_step_candidates(state, initial)
                    initial = False
                    overall_acc = sum(accs) / len(accs)
                    overall_depth = sum(depths) / len(depths)

                    if sub_state is None:
                        # filter the sample where it is too easy (avg_acc = 1) or too difficult (avg_acc = 0)
                        if overall_acc == 0 or overall_acc == 1.0:
                            state.messages.extend(next_steps[0])
                            state.output.completion = answers[0]
                            break

                        # prepare for the next reasoning step (potential winning and losing pairs)
                        best_answer, best_depth, best_next_step, random_next_step = (
                            simple_extract(answers, accs, depths, next_steps)
                        )

                        # Also, filter the question that can be answered at once without tool use.
                        if best_depth == 1:
                            state.messages.extend(best_next_step)
                            state.output.completion = best_answer
                            break

                        start_info = {
                            "index": 1,
                            "messages": [state.messages[1]],
                            "overall_accuracy": overall_acc,
                            "steps_to_completion": overall_depth,
                        }
                        state.metadata["start_state"] = start_info

                        sub_state = copy.deepcopy(state)
                        state.messages.extend(best_next_step)
                        sub_state.messages.extend(random_next_step)

                    # Evaluate the overall acc of the sub-state
                    else:
                        (
                            sub_answers,
                            sub_accs,
                            sub_depths,
                            sub_next_steps,
                        ) = await evaluate_step_candidates(sub_state, initial)
                        sub_overall_acc = sum(sub_accs) / len(sub_accs)
                        sub_overall_depth = sum(sub_depths) / len(sub_depths)

                        # stop annotation if in the current reasoning step both potential winning and losing steps acheive all success (1) or all failure (0), making it meaningless to choose which one is better.
                        # This will also filter the case where the model has to output only the answer.
                        if (overall_acc == sub_overall_acc == 1.0) or (
                            overall_acc == sub_overall_acc == 0.0
                        ):
                            state.messages.extend(sub_next_steps[0])
                            state.output.completion = sub_answers[0]
                            break

                        # if the current state is the winning sample
                        elif overall_acc > sub_overall_acc or (
                            overall_acc == sub_overall_acc
                            and sub_overall_depth >= overall_depth
                        ):
                            avg_steps_to_completion_w = sum(depths) / len(depths)
                            avg_steps_to_completion_l = sum(sub_depths) / len(
                                sub_depths
                            )
                            win_info = {
                                "index": current_step_index - 2,
                                "messages": state.messages[
                                    current_step_index - 2 : current_step_index
                                ],
                                "overall_accuracy": overall_acc,
                                "steps_to_completion": avg_steps_to_completion_w,
                            }
                            lose_info = {
                                "index": current_step_index - 2,
                                "messages": sub_state.messages[
                                    current_step_index - 2 : current_step_index
                                ],
                                "overall_accuracy": sub_overall_acc,
                                "steps_to_completion": avg_steps_to_completion_l,
                            }
                            state.metadata["win_sample"].append(win_info)
                            state.metadata["lose_sample"].append(lose_info)

                            # In the intermediate state if it is no worth annotating anymore (avg_acc = 1), stop annotations
                            if overall_acc == 1.0:
                                state.messages.extend(next_steps[0])
                                state.output.completion = answers[0]
                                break

                            (
                                best_answer,
                                best_depth,
                                best_next_step,
                                random_next_step,
                            ) = simple_extract(answers, accs, depths, next_steps)
                            sub_state = copy.deepcopy(state)
                            state.messages.extend(best_next_step)
                            sub_state.messages.extend(random_next_step)

                        # if sub-state turns out to be the winning sample
                        else:
                            avg_steps_to_completion_w = sum(sub_depths) / len(
                                sub_depths
                            )
                            avg_steps_to_completion_l = sum(depths) / len(depths)

                            win_info = {
                                "index": current_step_index - 2,
                                "messages": sub_state.messages[
                                    current_step_index - 2 : current_step_index
                                ],
                                "overall_accuracy": sub_overall_acc,
                                "steps_to_completion": avg_steps_to_completion_w,
                            }
                            lose_info = {
                                "index": current_step_index - 2,
                                "messages": state.messages[
                                    current_step_index - 2 : current_step_index
                                ],
                                "overall_accuracy": overall_acc,
                                "steps_to_completion": avg_steps_to_completion_l,
                            }
                            state.metadata["win_sample"].append(win_info)
                            state.metadata["lose_sample"].append(lose_info)

                            # In the intermediate state if it is no worth annotating anymore (avg_acc = 1), stop annotations
                            if sub_overall_acc == 1.0:
                                state.messages.extend(sub_next_steps[0])
                                state.output.completion = sub_answers[0]
                                break

                            (
                                best_answer,
                                best_depth,
                                best_next_step,
                                random_next_step,
                            ) = simple_extract(
                                sub_answers, sub_accs, sub_depths, sub_next_steps
                            )
                            state.messages = copy.deepcopy(sub_state.messages)
                            state.messages.extend(best_next_step)
                            sub_state.messages.extend(random_next_step)

            return state

        return solve

    # return chain
    return chain(
        init
        + [
            tools,
            submit_tool(),
            basic_agent_loop(),
        ]
    )
