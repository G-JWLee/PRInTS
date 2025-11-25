import argparse
import json
import os
import re

import pandas as pd
from datasets import Dataset

OURS_PROMPT_FORMAT = """Instruction:
You are given a user information seeking problem. Your task is to act as an impartial judge and evaluate how well the "Current Reasoning Step" contributes to solving the user's problem based on the problem description and historical reasoning context. It is OK that the current step does not contain any tool call response.

REASONING EVALUATION RULES:
- As you evaluate, develop and refine your assessment criteria based on the specific requirements of this problem type and reasoning context. Think carefully about how to assess the quality of the current reasoning step. Your thinking should include your evaluation criteria, explaining how the step aligns with or deviates from your expectations.
- Finally, assign the reasoning step a score from -4 to 4, using either an integer or a decimal with up to 0.1 precision. A higher score should indicate a higher-quality response.

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
1. Criteria Development: [Identify the key evaluation criteria relevant for evaluating this reasoning step. Consider factors such as: logical validity and coherence of the step, tool call appropriateness and argument quality (whether too general or too narrow), consistency with user problem, historical reasoning trace summary, and previous tool response, informative/progress toward final answer, confidence and uncertainty expression, etc. Briefly explain why your selected criteria are critical for this particular evaluation.]
2. Analysis: [Always provide a step-by-step analysis here. First, briefly state the goal of the current reasoning step. Second, systematically evaluate the step against each of your identified criteria above. For each criterion, assess how well the step performs and explain your reasoning. If errors or deficiencies are found, clearly explain what is wrong and why. If the step performs well on a criterion, explain why it succeeds.]
3. Final Judgment: [Provide the final judgment within \\boxed_score{{}}. Examples: \\boxed_score{{-3.0}} or \\boxed_score{{3.5}}.]
"""  # noqa: E501


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)

    return tokenizer


def chat_messages_to_prompt(messages) -> str:
    formatted_lines = []
    for msg in messages:
        role = msg['role'].capitalize()
        if isinstance(msg['content'], str):
            formatted_lines.append(f"**{role}**: {msg['content'].strip()}")
        else:
            formatted_lines.append(f"**{role}**:")
            for content in msg['content']:
                type_name = content['type']
                formatted_lines.append(f"   - {type_name}: {content[type_name].strip()}")

        # If a tool call exists, include it in a structured way
        if "tool_calls" in msg and msg['tool_calls'] is not None:
            for tool_call in msg['tool_calls']:
                formatted_lines.append(f"   - tool_call: {tool_call}")

    # join all formatted lines into a single string
    return "\n".join(formatted_lines).strip()


def process_information_gain(example, label, split, idx, tokenizer, max_prompt_length, use_scoring=True, use_comparison=True):
    problem = chat_messages_to_prompt(example['trajectory'][1:2])
    historical_summary = example['historical_summary']
    previous_tool_response = chat_messages_to_prompt("" if len(example['trajectory']) <= 4 else example['trajectory'][-3:-2])
    current_reasoning = chat_messages_to_prompt(example['trajectory'][-2:-1])
    
    input_prompt = OURS_PROMPT_FORMAT.format(problem=problem, historical_summary=historical_summary, previous_tool_response=previous_tool_response, current_reasoning=current_reasoning)
    input_prompt = [{"role": "user", "content": input_prompt,},]

    prompt_length = len(tokenizer.apply_chat_template(
        input_prompt, add_generation_prompt=True,
    ))

    if prompt_length > max_prompt_length:
        return None
    else:
        if use_scoring:
            M = 4
            score = (example['current_avg_acc'] - example['prev_avg_acc']) * M
            assert score <= 4.0 and score >= -4

        else:
            score = None

        if not use_comparison:
            label = None

        data = {
            "data_source": "webagent_annotation",
            "prompt": input_prompt,
            "ability": "scoring",
            "reward_model": {"style": "rule", "ground_truth": label, "score": score, "current_summary": example['current_summary']['content'][-1]['text']},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": input_prompt,
                "need_tools_kwargs": False,
                "interaction_kwargs": {
                    "query": input_prompt,
                    "current_avg_acc": example['current_avg_acc'],
                    "prev_avg_acc": example['prev_avg_acc'],
                    "current_avg_depth": example['current_avg_depth'],
                    "prev_avg_depth": example['prev_avg_depth'],
                },
            },
        }

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--local_dir", type=str, default=".")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--use_scoring", action='store_true')
    parser.add_argument("--use_comparison", action='store_true')
    args = parser.parse_args()

    raw_dataset = json.load(open(args.data_path))

    num_dataset = len(raw_dataset)
    train_dataset = raw_dataset[:int(num_dataset * 0.9)]
    test_dataset = raw_dataset[int(num_dataset * 0.9):]

    tokenizer = load_tokenizer(args.tokenizer_path)
    max_prompt_length = args.max_prompt_length
    
    train_data_list = []
    for idx, example in enumerate(train_dataset):
        processed_chosen_data = process_information_gain(example['chosen'], label='chosen', split='train', idx=idx, tokenizer=tokenizer, max_prompt_length=max_prompt_length, use_scoring=args.use_scoring, use_comparison=args.use_comparison)

        processed_rejected_data = process_information_gain(example['rejected'], label='rejected', split='train', idx=idx, tokenizer=tokenizer, max_prompt_length=max_prompt_length, use_scoring=args.use_scoring, use_comparison=args.use_comparison)

        if processed_chosen_data is None or processed_rejected_data is None:
            continue

        chosen_score = processed_chosen_data['reward_model']['score']
        rejected_score = processed_rejected_data['reward_model']['score']

        weight = (chosen_score - rejected_score) / 8
        processed_chosen_data['reward_model']['weight'] = weight
        processed_rejected_data['reward_model']['weight'] = weight

        train_data_list.append(processed_chosen_data)
        train_data_list.append(processed_rejected_data)

    test_data_list = []
    for idx, example in enumerate(test_dataset):
        processed_chosen_data = process_information_gain(example['chosen'], label='chosen', split='test', idx=idx, tokenizer=tokenizer, max_prompt_length=max_prompt_length, use_scoring=args.use_scoring, use_comparison=args.use_comparison)

        processed_rejected_data = process_information_gain(example['rejected'], label='rejected', split='test', idx=idx, tokenizer=tokenizer, max_prompt_length=max_prompt_length, use_scoring=args.use_scoring, use_comparison=args.use_comparison)

        if processed_chosen_data is None or processed_rejected_data is None:
            continue

        chosen_score = processed_chosen_data['reward_model']['score']
        rejected_score = processed_rejected_data['reward_model']['score']

        weight = (chosen_score - rejected_score) / 8
        processed_chosen_data['reward_model']['weight'] = weight
        processed_rejected_data['reward_model']['weight'] = weight

        test_data_list.append(processed_chosen_data)
        test_data_list.append(processed_rejected_data)

    # Convert the list of samples to a pandas DataFrame
    train_dataset = pd.DataFrame(train_data_list)
    train_dataset = Dataset.from_pandas(train_dataset)

    test_dataset = pd.DataFrame(test_data_list)
    test_dataset = Dataset.from_pandas(test_dataset)

    local_dir = args.local_dir
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))


