import os
from inspect_ai.dataset import Dataset, Sample, MemoryDataset

import json


def summary_generator_dataset() -> Dataset:
    SUMMARY_GENERATOR_DIRECTORY = "/annotated_data_dir"
    data = json.load(
        open(
            os.path.join(SUMMARY_GENERATOR_DIRECTORY, "annotated_sample.json"),
            "r",
        )
    )

    samples = []
    for example in data:
        sample = Sample(
            input="",
            target="",
            metadata=example,
        )
        samples.append(sample)

    return MemoryDataset(samples=samples)
