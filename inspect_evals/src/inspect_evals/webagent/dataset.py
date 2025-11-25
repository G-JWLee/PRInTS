import os
import glob
import json
import re

from inspect_ai.dataset import Dataset, Sample, MemoryDataset


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    # Unicode ranges for Chinese characters:
    # \u4e00-\u9fff: CJK Unified Ideographs
    # \u3400-\u4dbf: CJK Unified Ideographs Extension A
    # \uf900-\ufaff: CJK Compatibility Ideographs
    chinese_pattern = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")
    return bool(chinese_pattern.search(text))


def webagent_dataset(
    subdirectory: str,  # Add subdirectory parameter
) -> Dataset:
    WEBAGENT_DIRECTORY = "/webagent_corpus_directory"
    if subdirectory == "all":
        data_files = glob.glob(os.path.join(WEBAGENT_DIRECTORY, "*.jsonl"))
    elif subdirectory == "webagent":
        data_files = glob.glob(os.path.join(WEBAGENT_DIRECTORY, "web*.jsonl"))
    else:
        data_files = [os.path.join(WEBAGENT_DIRECTORY, f"{subdirectory}.jsonl")]

    data = []
    for data_file in data_files:
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                file_data = [json.loads(line) for line in f]
                data.extend(file_data)

    samples = []
    for example in data:
        prompt = DEFAULT_INPUT_PROMPT
        sample = Sample(
            input=prompt.format(question=example["question"]),
            target=example["answer"],
            metadata={"source": example["source"] if "source" in example else None},
        )
        samples.append(sample)

    return MemoryDataset(samples=samples)


DEFAULT_INPUT_PROMPT = """Please answer the question below. You should return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.

Here is the question:

{question}"""
