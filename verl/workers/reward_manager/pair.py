# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

_SOLUTION_CLIP_CHARS = 300


def extract_score(solution_str, reward_min):

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    # this also tests the formatting of the model
    solutions = re.findall(r"\\boxed\{(.*?)\}", solution_str)
    solutions_score = re.findall(r"\\boxed_score\{(.*?)\}", solution_str)

    if len(solutions) == 0 and len(solutions_score) == 0:
        final_answer = reward_min
    elif len(solutions) != 0:
        # take the last solution
        final_answer = solutions[-1]
        try:
            final_answer = float(final_answer)
        except ValueError:
            final_answer = reward_min
    else:
        # take the last solution
        final_answer = solutions_score[-1]
        try:
            final_answer = float(final_answer)
        except ValueError:
            final_answer = reward_min

    return final_answer


@register("pair_weighted")
class PairWeightedRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, n_rollouts, reward_score_range, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.n_rollouts = n_rollouts  # the number of rollouts per sample
        self.reward_score_range = reward_score_range
        self.reward_score_length = abs(reward_score_range[0] - reward_score_range[1]) 
        self.reward_min = min(reward_score_range[0], reward_score_range[1])

    def extract_all_predicted_scores(self, data: DataProto):
        """
        Extract predicted scores from all rollouts and organize by sample.
        
        Input format: [sample1_chosen_1, ..., sample1_chosen_n, sample1_rejected_1, ..., sample1_rejected_n, 
                      sample2_chosen_1, ..., sample2_chosen_n, sample2_rejected_1, ..., sample2_rejected_n, ...]
        
        Returns:
            Dict with structure:
            {
                sample_idx: {
                    'chosen_scores': [score_dict1, score_dict2, ...],  # length n_rollouts
                    'rejected_scores': [score_dict1, score_dict2, ...],  # length n_rollouts
                    'chosen_indices': [idx1, idx2, ...],  # original indices in data
                    'rejected_indices': [idx1, idx2, ...]  # original indices in data
                }
            }
        """
        total_rollouts_per_sample = 2 * self.n_rollouts  # n chosen + n rejected
        num_samples = len(data) // total_rollouts_per_sample
        
        all_scores = {}
        
        if 'global_idx' in data.batch.keys():
            global_indices = data.batch['global_idx']
        else:
            global_indices = torch.arange(len(data.batch['prompts']))
        inverse_global_indices = torch.argsort(global_indices)

        for sample_idx in range(num_samples):
            start_idx = sample_idx * total_rollouts_per_sample
            
            # Initialize sample data
            all_scores[sample_idx] = {
                'chosen_scores': [],
                'rejected_scores': [],
                'chosen_ground_truth_scores': [],
                'rejected_ground_truth_scores': [],
                'chosen_indices': [],
                'rejected_indices': [],
                'chosen_prompt': [],
                'rejected_prompt': [],
                'chosen_response': [],
                'rejected_response': [],
                'chosen_comparison_weight': [],
                'rejected_comparison_weight': [],
                'do_compare': True,
            }
            
            # Process chosen rollouts (first n rollouts)
            chosen_start = start_idx
            chosen_end = chosen_start + self.n_rollouts
            
            idx_chosen_indices = inverse_global_indices[chosen_start:chosen_end]
            idx_chosen_indices = idx_chosen_indices.tolist()

            for i in idx_chosen_indices:
                score, label, ground_truth_score, prompt, response, comparison_weight = self._extract_score_from_rollout(data[i])
                if label is not None:
                    assert label == 'chosen'
                else:
                    all_scores[sample_idx]['do_compare'] = False
                all_scores[sample_idx]['chosen_scores'].append(score)
                all_scores[sample_idx]['chosen_ground_truth_scores'].append(ground_truth_score)
                all_scores[sample_idx]['chosen_indices'].append(i)
                all_scores[sample_idx]['chosen_prompt'].append(prompt)
                all_scores[sample_idx]['chosen_response'].append(response)
                all_scores[sample_idx]['chosen_comparison_weight'].append(comparison_weight)

            # Process rejected rollouts (next n rollouts)
            rejected_start = chosen_end
            rejected_end = rejected_start + self.n_rollouts

            idx_rejected_indices = inverse_global_indices[rejected_start:rejected_end]
            idx_rejected_indices = idx_rejected_indices.tolist()
            
            for i in idx_rejected_indices:
                score, label, ground_truth_score, prompt, response, comparison_weight = self._extract_score_from_rollout(data[i])
                if label is not None:
                    assert label == 'rejected'
                else:
                    all_scores[sample_idx]['do_compare'] = False
                all_scores[sample_idx]['rejected_scores'].append(score)
                all_scores[sample_idx]['rejected_ground_truth_scores'].append(ground_truth_score)
                all_scores[sample_idx]['rejected_indices'].append(i)
                all_scores[sample_idx]['rejected_prompt'].append(prompt)
                all_scores[sample_idx]['rejected_response'].append(response)
                all_scores[sample_idx]['rejected_comparison_weight'].append(comparison_weight)

        return all_scores


    def _extract_score_from_rollout(self, data_item):
        """
        Extract predicted scores from a single rollout.
        
        Returns:
            Dict containing predicted 'accuracy' and 'depth' scores
        """
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        
        # Decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        
        ground_truth = data_item.non_tensor_batch["reward_model"].get("ground_truth", None)
        ground_truth_score = data_item.non_tensor_batch["reward_model"].get("score", None)
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        comparison_weight = data_item.non_tensor_batch["reward_model"].get("weight", 1)

        extra_info["num_turns"] = num_turns
        
        score_result = extract_score(response_str, self.reward_min)

        return score_result, ground_truth, ground_truth_score, prompt_str, response_str, comparison_weight


    def compute_average_pairwise_reward(self, chosen_scores, rejected_scores, rollout_type='chosen', do_compare=True):
        """
        Compute average pairwise reward for a rollout against all counterpart rollouts.
        
        Args:
            chosen_scores: List of score dicts for chosen rollouts
            rejected_scores: List of score dicts for rejected rollouts  
            rollout_type: 'chosen' or 'rejected' - which type this rollout belongs to
            
        Returns:
            Average reward across all pairwise comparisons
        """
        if do_compare:
            if rollout_type == 'chosen':
                # For a chosen rollout, compare against all rejected rollouts
                target_scores = chosen_scores
                comparison_scores = rejected_scores
                is_chosen = True
            else:
                # For a rejected rollout, compare against all chosen rollouts  
                target_scores = rejected_scores
                comparison_scores = chosen_scores
                is_chosen = False
                
            total_rewards = []
            
            for target_score in target_scores:
                rollout_rewards = []
                
                for comp_score in comparison_scores:
                    if is_chosen:
                        # Chosen vs Rejected comparison
                        better = target_score > comp_score
                        reward = 1.0 if better else -1.0
                    else:
                        # Rejected vs Chosen comparison
                        worse = target_score < comp_score
                        reward = 1.0 if worse else -1.0
                    
                    rollout_rewards.append(reward)
                
                # Average reward for this rollout across all comparisons
                avg_reward = sum(rollout_rewards) / len(rollout_rewards)
                total_rewards.append(avg_reward)
        else:
            total_rewards = torch.zeros(len(chosen_scores),)
            total_rewards = total_rewards.tolist()
            
        return total_rewards


    def compute_score_reward(self, predicted_scores, ground_truth_scores):

        if ground_truth_scores[0] is not None:
            predicted_scores = torch.tensor(predicted_scores)
            ground_truth_scores = torch.tensor(ground_truth_scores)

            diff = torch.abs(predicted_scores - ground_truth_scores) / self.reward_score_length
            reward = 1 - diff

            reward = reward.tolist()
        else:
            reward = torch.zeros(len(predicted_scores),)
            reward = reward.tolist()

        return reward


    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Prepare output tensors
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        # Step 1: Extract all predicted scores first
        all_scores = self.extract_all_predicted_scores(data)

        # Step 2: Compute average rewards for each rollout
        for sample_idx, sample_data in all_scores.items():
            chosen_scores = sample_data['chosen_scores']
            rejected_scores = sample_data['rejected_scores'] 
            chosen_indices = sample_data['chosen_indices']
            rejected_indices = sample_data['rejected_indices']

            do_compare = sample_data['do_compare']
            
            # Compute average rewards for chosen rollouts
            chosen_pair_rewards = self.compute_average_pairwise_reward(
                chosen_scores, rejected_scores, rollout_type='chosen', do_compare=do_compare,
            )
            
            # Compute average rewards for rejected rollouts
            rejected_pair_rewards = self.compute_average_pairwise_reward(
                chosen_scores, rejected_scores, rollout_type='rejected', do_compare=do_compare,
            )

            # score reward
            chosen_ground_scores = sample_data['chosen_ground_truth_scores']
            rejected_ground_scores = sample_data['rejected_ground_truth_scores']

            chosen_score_rewards = self.compute_score_reward(chosen_scores, chosen_ground_scores)
            rejected_score_rewards = self.compute_score_reward(rejected_scores, rejected_ground_scores)
            
            chosen_weights = sample_data['chosen_comparison_weight']
            rejected_weights = sample_data['rejected_comparison_weight']

            # Assign rewards to tensor at correct positions
            for i, (orig_idx, pair_reward, score_reward, weight) in enumerate(zip(chosen_indices, chosen_pair_rewards, chosen_score_rewards, chosen_weights)):
                data_item = data[orig_idx]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                reward_tensor[orig_idx, valid_response_length - 1] = weight * pair_reward + score_reward

                # Store extra info
                reward_extra_info['sample_idx'].append(sample_idx)
                reward_extra_info['rollout_type'].append('chosen')
                reward_extra_info['rollout_idx'].append(i)
                reward_extra_info['pair_reward'].append(pair_reward)
                reward_extra_info['score_reward'].append(score_reward)
                
            for i, (orig_idx, pair_reward, score_reward, weight) in enumerate(zip(rejected_indices, rejected_pair_rewards, rejected_score_rewards, rejected_weights)):
                data_item = data[orig_idx]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                reward_tensor[orig_idx, valid_response_length - 1] = weight * pair_reward + score_reward

                # Store extra info
                reward_extra_info['sample_idx'].append(sample_idx)
                reward_extra_info['rollout_type'].append('rejected')
                reward_extra_info['rollout_idx'].append(i)
                reward_extra_info['pair_reward'].append(pair_reward)
                reward_extra_info['score_reward'].append(score_reward)

            # Debug printing
            data_source = data[chosen_indices[0]].non_tensor_batch[self.reward_fn_key]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n[Sample {sample_idx}] Data source: {data_source}")
                print(f"Overall chosen avg: {sum(chosen_pair_rewards) / len(chosen_pair_rewards):.3f}")
                print(f"Overall rejected avg: {sum(rejected_pair_rewards) / len(rejected_pair_rewards):.3f}")
                
                print("[chosen_prompt]", sample_data['chosen_prompt'])
                print("[chosen_response]", sample_data['chosen_response'])
                if chosen_ground_scores[0] is not None:
                    print("[chosen_ground_truth]", sample_data['chosen_ground_truth_scores'])

                print("[rejected_prompt]", sample_data['rejected_prompt'])
                print("[rejected_response]", sample_data['rejected_response'])
                if rejected_ground_scores[0] is not None:
                    print("[rejected_ground_truth]", sample_data['rejected_ground_truth_scores'])
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
