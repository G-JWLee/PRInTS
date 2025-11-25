import argparse
import json
import os
import re

import pandas as pd
from datasets import Dataset

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


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)

    return tokenizer

def extract_summary(solution_str):

    # this also tests the formatting of the model
    solutions = re.findall(r"\\boxed_summary\{(.*?)\}", solution_str, re.DOTALL)
    if len(solutions) > 0 and solutions[-1].strip() != "":
        return solutions[-1].strip()

    match = re.search(r"2\.\s*\*{0,2}Updated Summary\*{0,2}:\s*(.*)(?=\\boxed_summary\{|$)", solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""

def extract_analysis(solution_str):

    match = re.search(r"1\.\s*\*{0,2}Analysis\*{0,2}:\s*(.*?)(?=\s*2\.\s*\*{0,2}Updated Summary\*{0,2}:|\\n2\.\s*\*{0,2}Updated Summary\*{0,2}:|\Z)", solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"1\.\s*Analysis:\s*(.*?)(?=\s*2\.\s*\*\*Updated Summary\*\*:|\s*2\.\s*Updated Summary:|\\n2\.\s*\*\*Updated Summary\*\*:|\\n2\.\s*Updated Summary:|\Z)", solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""


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


def process_trajacetory_summary(example, split, idx, tokenizer, max_prompt_length):
    if 'historical_summary' not in example:
        return None
    problem = chat_messages_to_prompt(example['trajectory'][1:2])
    idx = len(example['trajectory']) // 2 - 2
    if idx == 0:
        historical_summary = ""
        previous_tool_response = ""
        current_reasoning = chat_messages_to_prompt(example['trajectory'][2 * idx + 2 : 2 * idx + 3])
    else:
        historical_summary = example['historical_summary']
        if historical_summary == "":
            return None
        previous_tool_response = chat_messages_to_prompt(example['trajectory'][2 * idx + 1 : 2 * idx + 2])
        current_reasoning = chat_messages_to_prompt(example['trajectory'][2 * idx + 2 : 2 * idx + 3])

    question = SUMMARY_GENERATION_FORMAT.format(problem=problem, historical_summary=historical_summary, previous_tool_response=previous_tool_response, current_reasoning=current_reasoning)

    valid_current_summary = extract_summary(example['current_summary']['content'][-1]['text'])
    if valid_current_summary == "":
        return None
    valid_current_analysis = extract_analysis(example['current_summary']['content'][-1]['text'])
    if valid_current_analysis == "":
        return None

    # Make unified format to do training.
    valid_output = f"1. Analysis: {valid_current_analysis} \n\n2. Updated Summary:  \n\\boxed_summary{{  \n{valid_current_summary}  \n}}"

    answer = f"<think>\n{example['current_summary']['content'][0]['reasoning']}\n</think>\n\n{valid_output}"
    prompt_chat = [{"role": "user", "content": question,},]
    prompt_chat_str = tokenizer.apply_chat_template(
        prompt_chat, add_generation_prompt=True, add_special_tokens=False, tokenize=False
    )
    response_chat_str = answer + tokenizer.eos_token

    prompt_length = len(tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0])
    response_length = len(tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0])

    if prompt_length + response_length > max_prompt_length:
        return None
    else:
        data = {
            "data_source": "summary_annotation",
            "prompt": prompt_chat,
            "ability": "summary_generation",
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question,
                "answer": answer,
            },
        }

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--local_dir", type=str, default=".")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max_prompt_length", type=int, default=4096)

    args = parser.parse_args()

    dataset = json.load(open(args.data_path, 'r'))

    num_dataset = len(dataset)
    train_dataset = dataset[:int(num_dataset * 0.9)]
    test_dataset = dataset[int(num_dataset * 0.9):]

    tokenizer = load_tokenizer(args.tokenizer_path)
    max_prompt_length = args.max_prompt_length
    
    train_data_list = []
    for idx, example in enumerate(train_dataset):

        # Now that the summaries generated for both chosen and rejected samples, we need both.
        processed_chosen_data = process_trajacetory_summary(example['chosen'], split='train', idx=idx, tokenizer=tokenizer, max_prompt_length=max_prompt_length)

        if processed_chosen_data is not None:
            train_data_list.append(processed_chosen_data)

        processed_rejected_data = process_trajacetory_summary(example['rejected'], split='train', idx=idx, tokenizer=tokenizer, max_prompt_length=max_prompt_length)

        if processed_rejected_data is not None:
            train_data_list.append(processed_rejected_data)

    test_data_list = []
    for idx, example in enumerate(test_dataset):
        
        processed_chosen_data = process_trajacetory_summary(example['chosen'], split='test', idx=idx, tokenizer=tokenizer, max_prompt_length=max_prompt_length)

        if processed_chosen_data is not None:
            test_data_list.append(processed_chosen_data)
            
        processed_rejected_data = process_trajacetory_summary(example['rejected'], split='test', idx=idx, tokenizer=tokenizer, max_prompt_length=max_prompt_length)

        if processed_rejected_data is not None:
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


