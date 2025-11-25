# [PRInTS: Rewarding Agents for Long-Horizon Information Seeking](TBD)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](TBD)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Jaewoo Lee](https://g-jwlee.github.io/) | [Archiki Prasad](https://archiki.github.io/) | [Justin Chih-Yao Chen](https://dinobby.github.io/) | [Zaid Khan](https://zaidkhan.me/) | [Elias Stengel-Eskin](https://esteng.github.io/) | [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

## Overview
Long-horizon information-seeking tasks require agents to gather and synthesize information across multiple reasoning steps and tool interactions.
While process reward models (PRMs) can guide agents by ranking candidate steps at test-time, existing PRMs cannot capture richer dimensions of information-seeking steps nor handle the rapidly growing context in long-horizon tasks.
We propose PRInTS (Process Reward via Information gain scoring and Trjeactory Summary), a generative PRM jointly trained with two key abilities for fine-grained guidance under the challenge of context accumulation.

<center><img src="assets/PRInTS_overview.png" alt="Teaser" width="100%"></center>
<p>

🎯 **PRInTS as a scorer**: evaluates agent's multiple candidate next trajectory steps based on the summarized context and current tool response, and outputs dense scores based on the PRM's reasoning across multiple step quality dimensions (e.g., interpretation of tool outputs, tool call informativeness)<br>
📝 **PRInTS as a summarizer**: recursively updates a compact information-seeking trajectory summary to keep input length bounded and preserve key information for its subsequent score evaluation.

</p>

## Install
Please follow the installation instructions from [verl](https://github.com/volcengine/verl).

## Data annotation
Our data annotation pipeline is based on Inspect Eval evaluation framework.
Please follow the installation isntructions from [Inspect Eval](https://github.com/UKGovernmentBEIS/inspect_evals).
Download the QA corpus from [MiroVerse](https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1) and [webagent families](https://github.com/Alibaba-NLP/DeepResearch), and store them in /webagent_corpus_directory directory.

For scoring annotation, run
```shell
cd inspect_evals
inspect eval inspect_evals/webagent 
```
Save the score annotation logs into /annotated_data_dir/annotation_raw_trajectory.json, and run
```shell
python preprocess_trajectory.py
```

For summary annotation, run
```shell
inspect eval inspect_evals/summary_generator
```
Save the summary annotation logs into /annotated_data_dir/annotation_raw_trajectory_summary.json, and run
```shell
python preprocess_trajectory_summary.py
```

Now construct datasets for both GRPO and SFT
```shell
cd ..
python examples/data_preprocess/prints_grpo_dataset.py --data_path /annotated_data_dir/annotated_sample_summary.json --local_dir benchmarks/PRInTS_infogain_annotation --tokenizer_path Qwen/Qwen3-4B --max_prompt_length 6144 --use_scoring --use_comparison
python examples/data_preprocess/prints_sftdataset.py --data_path /annotated_data_dir/annotated_sample_summary.json --local_dir benchmarks/PRInTS_summary_annotation --tokenizer_path Qwen/Qwen3-4B --max_prompt_length 8192
```

## Training
We train PRInTS on Qwen3-4B with our alternating SFT-GRPO training schedule.
```shell
bash examples/grpo_trainer/run_qwen3-4b_PRInTS_iterative_lr1e6.sh
```

## Evaluation
For evaluation we use the [Inspect Eval](https://github.com/UKGovernmentBEIS/inspect_evals) evaluation pipeline and implement [FRAMES](https://huggingface.co/datasets/inclusionAI/ASearcher-test-data), [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA), and [WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA) on top of the framework.

## Bibtex
```
@article{lee2024prints,
      title={PRInTS: Reward Modeling for Long-Horizon Information Seeking},
      author={Jaewoo Lee and Archiki Prasad and Justin Chih-Yao Chen and Zaid Khan and Elias Stengel-Eskin and Mohit Bansal},
      year={2025},
      journal={arXiv preprint arXiv:2511.19314},
      url={https://arxiv.org/abs/2511.19314},
}
```
