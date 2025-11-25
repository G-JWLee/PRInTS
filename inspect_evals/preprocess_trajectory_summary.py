import json
import os
import glob
import random

import matplotlib.pyplot as plt
import numpy as np


def extract_samples(filepath):
    samples = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data["samples"]:
            samples.append(sample["metadata"])

    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"Error reading {filepath}: {e}")
        return []

    return samples


if __name__ == "__main__":
    target_files = [
        "/annotated_data_dir/annotation_raw_trajectory_summary.json",
    ]
    print(f"Found {len(target_files)} files to check:")

    annotated_samples = []

    for file_path in target_files:
        print(f"  - {file_path}")
        file_samples = extract_samples(file_path)

        if len(file_samples) == 0:
            print(f"Skipping - {file_path}")

        annotated_samples.extend(file_samples)

    with open(
        "/annotated_data_dir/annotated_sample_summary.json",
        "w",
    ) as f:
        json.dump(annotated_samples, f)

    print("finished!")
