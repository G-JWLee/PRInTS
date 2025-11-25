import re
import json
import numpy as np
import asyncio
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance
import copy

from logging import getLogger
from typing import Awaitable, Callable, cast, List

from typing_extensions import TypedDict, Unpack

from inspect_ai.model._cache import CachePolicy
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    ChatMessageSystem,
)
from inspect_ai.model._model import get_model
from inspect_ai.solver._chain import chain

from ._prompt import system_message
from ._solver import Generate, Solver, solver
from ._task_state import TaskState


logger = getLogger(__name__)

SUMMARY_GENERATION_FORMAT = """Instruction:
You are a reasoning trace summarizer for multi-step information seeking problems. Your task is to incrementally build a concise summary of an information-seeking process. Your summary should capture the process's state of knowledge, uncertainty, hypothesis, and next actions.

Input Sources:
# Information Seeking Problem - the original user question.
# Historical Reasoning Trace Summary - the accumulated summary built from all previous reasoning steps and tool responses.
# Previous Tool Response - the tool response from the immediately preceding step (not yet incorporated into Historical Summary).
# Current Reasoning Step - the reasoning and tool interaction from the current step (not the complete reasoning trace).

SUMMARIZATION RULES:
- Keep essential information from the Previous Tool Response and Current Reasoning Step needed for the next action.
- Incorporate what the current process believes, suspects, verifies, or is planning further verification.
- For the Current Reasoning Step's action, summarize tool name and key parameters.
- Preserve the Historical Reasoning Trace Summary unless explicitly contradicted or superseded. Justify any removals.
- Do NOT infer or invent missing information. It is normal for reasoning to be incomplete.
- Output the COMPLETE updated summary.

[Input]:
# Information Seeking Problem
{problem}

# Historical Reasoning Trace Summary
{historical_summary}

# Previous Tool Response
{previous_tool_response}

# Current Reasoning Step
{current_reasoning}

[Output format]:
1. Analysis: [In 3-5 sentences, explain what key information from Previous Tool Response and Current Reasoning Step is being added, what (if anything) from Historical Summary is being removed or updated and why, and why the resulting summary is sufficient for next steps.]
2. Updated Summary: [Provide the complete summary within \\boxed_summary{{}} containing:
- **Confirmed Knowledge**: Verified facts.
- **Uncertainty**: What remains unknown.
- **Previous Hypotheses**: Abandoned hypotheses (if relevant).
- **Previous Action**: Previous tool calls with key parameters in the Historical Reasoning Trace Summary.
- **Current Hypothesis**: Current working hypothesis in the Current Reasoning Step and Historical Reasoning Trace Summary.
- **Current Action**: Most recent tool call with key parameters in the Current Reasoning Step.]
"""  # noqa: E501


def chat_messages_to_prompt(messages) -> str:
    formatted_lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        if isinstance(msg["content"], str):
            formatted_lines.append(f"**{role}**: {msg['content'].strip()}")
        else:
            formatted_lines.append(f"**{role}**:")
            for content in msg["content"]:
                type_name = content["type"]
                formatted_lines.append(
                    f"   - {type_name}: {content[type_name].strip()}"
                )

        # If a tool call exists, include it in a structured way
        if "tool_calls" in msg and msg["tool_calls"] is not None:
            for tool_call in msg["tool_calls"]:
                formatted_lines.append(f"   - tool_call: {tool_call}")

    # join all formatted lines into a single string
    return "\n".join(formatted_lines).strip()


def extract_summary(solution_str):
    # this also tests the formatting of the model
    solutions = re.findall(r"\\boxed_summary\{(.*?)\}", solution_str, re.DOTALL)
    if len(solutions) > 0 and solutions[-1].strip() != "":
        return solutions[-1].strip()

    match = re.search(
        r"2\.\s*\*{0,2}Updated Summary\*{0,2}:\s*(.*)(?=\\boxed_summary\{|$)",
        solution_str,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    return ""


class BasicAgentDeprecatedArgs(TypedDict, total=False):
    max_messages: int | None


@solver
def basic_agent_summary(
    *,
    init: Solver | list[Solver] | None = None,
    cache: bool | CachePolicy = False,
    message_limit: int | None = None,
    token_limit: int | None = None,
    max_tool_output: int | None = None,
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
    # resolve init
    if init is None:
        init = system_message("")
    init = init if isinstance(init, list) else [init]

    # main agent loop
    @solver
    def basic_agent_loop() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            chosen_content = state.metadata["chosen"]["trajectory"]
            problem = chat_messages_to_prompt(chosen_content[1:2])
            for i in range(len(chosen_content) // 2 - 1):
                if i == 0:
                    historical_summary = ""
                    previous_tool_response = ""
                    current_reasoning = chat_messages_to_prompt(
                        chosen_content[2 * i + 2 : 2 * i + 3]
                    )
                else:
                    historical_summary = extract_summary(
                        state.output.message.content[-1].text
                    )
                    previous_tool_response = chat_messages_to_prompt(
                        chosen_content[2 * i + 1 : 2 * i + 2]
                    )
                    current_reasoning = chat_messages_to_prompt(
                        chosen_content[2 * i + 2 : 2 * i + 3]
                    )

                input_prompt = SUMMARY_GENERATION_FORMAT.format(
                    problem=problem,
                    historical_summary=historical_summary,
                    previous_tool_response=previous_tool_response,
                    current_reasoning=current_reasoning,
                )

                state.messages[1] = ChatMessageUser(content=input_prompt)
                state.output = await get_model().generate(
                    input=state.messages, cache=cache
                )

            state.metadata["chosen"]["historical_summary"] = historical_summary
            state.metadata["chosen"]["current_summary"] = copy.deepcopy(
                state.output.message
            )

            # Note that the rejected counterpart shares the KG, except for the last step.
            rejected_content = state.metadata["rejected"]["trajectory"]
            current_reasoning = chat_messages_to_prompt(
                rejected_content[2 * i + 2 : 2 * i + 3]
            )
            input_prompt = SUMMARY_GENERATION_FORMAT.format(
                problem=problem,
                historical_summary=historical_summary,
                previous_tool_response=previous_tool_response,
                current_reasoning=current_reasoning,
            )
            state.messages[1] = ChatMessageUser(content=input_prompt)
            state.output = await get_model().generate(input=state.messages, cache=cache)

            state.metadata["rejected"]["historical_summary"] = historical_summary
            state.metadata["rejected"]["current_summary"] = state.output.message

            return state

        return solve

    # return chain
    return chain(
        init
        + [
            basic_agent_loop(),
        ]
    )
