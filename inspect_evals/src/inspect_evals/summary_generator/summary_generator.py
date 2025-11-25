from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.solver import Solver, system_message
from inspect_ai.solver._basic_agent_summary import basic_agent_summary

from inspect_evals.summary_generator.dataset import summary_generator_dataset
from inspect_evals.summary_generator.prompts import SYSTEM_PROMPT


@task
def summary_generator(
    instance_ids: str | list[str] | None = None,
) -> Task:
    # read dataset
    dataset = summary_generator_dataset()
    # filter by instance id if requested
    if instance_ids:
        instance_ids = [instance_ids] if isinstance(instance_ids, str) else instance_ids
        dataset = dataset.filter(lambda x: x.id in instance_ids)

    # resolve solver
    solver = default_solver()

    # return task
    return Task(
        dataset=dataset,
        plan=solver,
    )


def default_solver() -> Solver:
    return basic_agent_summary(
        init=system_message(SYSTEM_PROMPT),
    )
