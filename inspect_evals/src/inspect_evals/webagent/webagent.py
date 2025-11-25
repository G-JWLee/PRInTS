from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.solver import Solver, basic_agent, system_message
from inspect_ai.solver._basic_agent_trajectory import basic_agent_trajectory
from inspect_ai.tool import bash, python, web_browser, web_search

from inspect_evals.webagent.dataset import webagent_dataset
from inspect_evals.webagent.scorer import webagent_scorer
from inspect_evals.webagent.prompts import SYSTEM_PROMPT

from inspect_ai.util._sandbox.environment import SandboxEnvironmentType
from inspect_ai.tool import mcp_server_stdio, mcp_tools

TASK_DIR = Path(__file__).parent
COMPOSE_FILE = TASK_DIR / "compose.yaml"
VALUES_FILE = TASK_DIR / "values.yaml"

DEFAULT_DOCKER_SANDBOX = ("docker", COMPOSE_FILE.as_posix())
DEFAULT_K8S_SANDBOX = ("k8s", VALUES_FILE.as_posix())


@task
def webagent(
    solver: Solver | None = None,
    max_attempts: int = 1,
    max_messages: int = 100,
    subset: Literal[
        "webdancer",
        "websailor",
        "webshaper",
        "MiroVerse-Voyager1.0_filtered",
        "all",
    ] = "all",
    num_initial_search: int = 8,
    num_samples_per_step: int = 8,
    instance_ids: str | list[str] | None = None,
    sandbox: SandboxEnvironmentType = DEFAULT_DOCKER_SANDBOX,
    inference_mode: str = "plain",
    random_seed: int = None,
) -> Task:
    # read dataset
    dataset = webagent_dataset(subdirectory=subset)

    # filter by instance id if requested
    if instance_ids:
        instance_ids = [instance_ids] if isinstance(instance_ids, str) else instance_ids
        dataset = dataset.filter(lambda x: x.id in instance_ids)

    if random_seed is not None:
        dataset.shuffle(random_seed)

    # resolve solver
    solver = solver or default_solver(
        max_attempts,
        max_messages,
        num_initial_search,
        num_samples_per_step,
        inference_mode,
    )

    # resolve scorer (test split has no answers)
    scorer = webagent_scorer()

    # return task
    return Task(
        dataset=dataset,
        plan=solver,
        scorer=scorer,
        sandbox=sandbox,
    )


def default_solver(
    max_attempts: int,
    max_messages: int,
    num_initial_search: int = 8,
    num_samples_per_step: int = 8,
    inference_mode: str = "plain",
    code_timeout: int = 180,
) -> Solver:
    search_server = mcp_server_stdio(
        command="npx",
        args=["-y", "serper-search-scrape-mcp-server"],
        env={"SERPER_API_KEY": "YOUR_API_KEY"},
    )
    search_server = mcp_tools(search_server, tools=["google_search"])

    return basic_agent_trajectory(
        init=system_message(SYSTEM_PROMPT),
        tools=[
            bash(code_timeout),
            python(code_timeout),
            search_server,
        ]
        + web_browser(),
        max_attempts=max_attempts,
        max_messages=max_messages,
        inference_mode=inference_mode,
        num_initial_search=num_initial_search,
        num_samples_per_step=num_samples_per_step,
    )
