import json
import os
import glob
import random

import matplotlib.pyplot as plt
import numpy as np


def extract_samples(filepath, filtering=False):
    samples = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0

        for sample in data["samples"]:
            if len(sample["metadata"]["win_sample"]) > 1:
                count += 1

            initial_prompt = sample["messages"][0:2]
            chosen_sample_traj = initial_prompt
            rejected_sample_traj = initial_prompt
            if "start_state" not in sample["metadata"]:
                continue
            else:
                initial_score = sample["metadata"]["start_state"]
            for i in range(len(sample["metadata"]["win_sample"])):
                # Filter sample where chosen and rejected steps have the same overall accuracy and average depth.
                if (
                    sample["metadata"]["win_sample"][i]["overall_accuracy"]
                    == sample["metadata"]["lose_sample"][i]["overall_accuracy"]
                    and sample["metadata"]["win_sample"][i]["steps_to_completion"]
                    == sample["metadata"]["lose_sample"][i]["steps_to_completion"]
                ):
                    continue

                if i == 0:
                    chosen_sample_traj = (
                        chosen_sample_traj
                        + sample["metadata"]["win_sample"][i]["messages"]
                    )
                    chosen_sample_info = {
                        "trajectory": chosen_sample_traj,
                        "current_avg_acc": sample["metadata"]["win_sample"][i][
                            "overall_accuracy"
                        ],
                        "current_avg_depth": sample["metadata"]["win_sample"][i][
                            "steps_to_completion"
                        ],
                        "prev_avg_acc": initial_score["overall_accuracy"],
                        "prev_avg_depth": initial_score["steps_to_completion"],
                    }
                    rejected_sample_traj = (
                        rejected_sample_traj
                        + sample["metadata"]["lose_sample"][i]["messages"]
                    )
                    rejected_sample_info = {
                        "trajectory": rejected_sample_traj,
                        "current_avg_acc": sample["metadata"]["lose_sample"][i][
                            "overall_accuracy"
                        ],
                        "current_avg_depth": sample["metadata"]["lose_sample"][i][
                            "steps_to_completion"
                        ],
                        "prev_avg_acc": initial_score["overall_accuracy"],
                        "prev_avg_depth": initial_score["steps_to_completion"],
                    }
                else:
                    chosen_sample_traj = (
                        chosen_sample_traj
                        + sample["metadata"]["win_sample"][i]["messages"]
                    )
                    chosen_sample_info = {
                        "trajectory": chosen_sample_traj,
                        "current_avg_acc": sample["metadata"]["win_sample"][i][
                            "overall_accuracy"
                        ],
                        "current_avg_depth": sample["metadata"]["win_sample"][i][
                            "steps_to_completion"
                        ],
                        "prev_avg_acc": sample["metadata"]["win_sample"][i - 1][
                            "overall_accuracy"
                        ],
                        "prev_avg_depth": sample["metadata"]["win_sample"][i - 1][
                            "steps_to_completion"
                        ],
                    }
                    rejected_sample_traj = (
                        rejected_sample_traj
                        + sample["metadata"]["lose_sample"][i]["messages"]
                    )
                    rejected_sample_info = {
                        "trajectory": rejected_sample_traj,
                        "current_avg_acc": sample["metadata"]["lose_sample"][i][
                            "overall_accuracy"
                        ],
                        "current_avg_depth": sample["metadata"]["lose_sample"][i][
                            "steps_to_completion"
                        ],
                        "prev_avg_acc": sample["metadata"]["win_sample"][i - 1][
                            "overall_accuracy"
                        ],
                        "prev_avg_depth": sample["metadata"]["win_sample"][i - 1][
                            "steps_to_completion"
                        ],
                    }

                samples.append(
                    {"chosen": chosen_sample_info, "rejected": rejected_sample_info}
                )

    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"Error reading {filepath}: {e}")
        return []

    chosen_sample_scores = []
    rejected_sample_scores = []

    chosen_sample_score_diffs = []
    rejected_sample_score_diffs = []

    for sample in samples:
        chosen_sample_info = sample["chosen"]
        rejected_sample_info = sample["rejected"]

        chosen_sample_scores.append(chosen_sample_info["current_avg_acc"])
        rejected_sample_scores.append(rejected_sample_info["current_avg_acc"])

        chosen_sample_score_diffs.append(
            (chosen_sample_info["current_avg_acc"] - chosen_sample_info["prev_avg_acc"])
            * 4
        )
        rejected_sample_score_diffs.append(
            (
                rejected_sample_info["current_avg_acc"]
                - rejected_sample_info["prev_avg_acc"]
            )
            * 4
        )

    return samples


if __name__ == "__main__":
    target_files = [
        "/annotated_data_dir/annotation_raw_trajectory.json",
    ]
    print(f"Found {len(target_files)} files to check:")

    annotated_samples = []

    for file_path in target_files:
        print(f"  - {file_path}")
        file_samples = extract_samples(file_path, filtering=False)

        if len(file_samples) == 0:
            print(f"Skipping - {file_path}")

        annotated_samples.extend(file_samples)

    with open(
        "/annotated_data_dir/annotated_sample.json",
        "w",
    ) as f:
        json.dump(annotated_samples, f)

    print("finished!")
