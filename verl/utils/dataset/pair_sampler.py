import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler


class PairRandomSampler(Sampler[int]):
    r"""Samples pairs of elements randomly for winning/losing preference predictin.
    
    This sampler ensures that data is sampled in pairs [0,1], [2,3], ..., [n-1,n]
    while maintaining the randomization capabilities of RandomSampler.
    
    Args:
        data_source (Dataset): dataset to sample from (must have even length)
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. Must be even.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool
    drop_last_unpaired: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

        # Validate num_samples
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )
        if self.num_samples % 2 != 0:
            raise ValueError(
                f"num_samples must be even for pair sampling, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples


    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        num_pairs = n // 2
        
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            # Sample pairs with replacement
            num_samples_pairs = self.num_samples // 2
            
            for _ in range(num_samples_pairs // 32):
                pair_indices = torch.randint(
                    high=num_pairs, size=(32,), dtype=torch.int64, generator=generator
                )
                for pair_idx in pair_indices.tolist():
                    yield pair_idx * 2      # winning sample
                    yield pair_idx * 2 + 1  # losing sample
            
            remaining_pairs = num_samples_pairs % 32
            if remaining_pairs > 0:
                pair_indices = torch.randint(
                    high=num_pairs, size=(remaining_pairs,), dtype=torch.int64, generator=generator
                )
                for pair_idx in pair_indices.tolist():
                    yield pair_idx * 2      # winning sample
                    yield pair_idx * 2 + 1  # losing sample
        else:
            # Sample pairs without replacement
            num_samples_pairs = self.num_samples // 2
            
            for _ in range(num_samples_pairs // num_pairs):
                # Generate random permutation of pair indices
                pair_perm = torch.randperm(num_pairs, generator=generator)
                for pair_idx in pair_perm.tolist():
                    yield pair_idx * 2      # winning sample
                    yield pair_idx * 2 + 1  # losing sample
            
            remaining_pairs = num_samples_pairs % num_pairs
            if remaining_pairs > 0:
                pair_perm = torch.randperm(num_pairs, generator=generator)
                for pair_idx in pair_perm.tolist()[:remaining_pairs]:
                    yield pair_idx * 2      # winning sample
                    yield pair_idx * 2 + 1  # losing sample

    def __len__(self) -> int:
        return self.num_samples