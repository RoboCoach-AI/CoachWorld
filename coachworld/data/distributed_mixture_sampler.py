"""Deterministic, resumable distributed sampling for mixed dataset roots."""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


def _largest_remainder_counts(
    probabilities: Sequence[float],
    total: int,
) -> list[int]:
    """Allocate an integer epoch budget while preserving the requested mix."""

    if int(total) <= 0:
        raise ValueError(f"total must be positive, got {total}")
    values = [float(value) for value in probabilities]
    if not values or any(value < 0.0 for value in values):
        raise ValueError(f"probabilities must be non-negative, got {values}")
    probability_sum = sum(values)
    if probability_sum <= 0.0:
        raise ValueError("at least one sampling probability must be positive")
    normalized = [value / probability_sum for value in values]
    raw = [value * int(total) for value in normalized]
    counts = [math.floor(value) for value in raw]
    remainder = int(total) - sum(counts)
    order = sorted(
        range(len(values)),
        key=lambda index: (raw[index] - counts[index], normalized[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    missing = [
        index
        for index, (probability, count) in enumerate(zip(normalized, counts))
        if probability > 0.0 and count == 0
    ]
    if missing:
        raise ValueError(
            "samples_per_epoch is too small to represent every positive-probability "
            f"root; zero-budget roots={missing}"
        )
    return counts


class ResumableDistributedMixtureSampler(Sampler[tuple[int, int]]):
    """Sample root-local windows with deterministic coverage and resume offsets.

    The sampler separates two concerns that the previous fixed expanded index
    conflated:

    * ``probabilities`` determine how many draws each root receives per epoch.
    * Each root owns a continuous stream of shuffled, without-replacement
      permutations, so a low-probability root rotates through all of its source
      windows instead of exposing one permanent subset.

    The global schedule is deterministic, then sharded across ranks. A resumed
    run derives its sampler epoch and rank-local offset directly from the saved
    training micro-step; no serialized Python RNG state is required.
    """

    def __init__(
        self,
        *,
        root_sizes: Sequence[int],
        probabilities: Sequence[float],
        num_replicas: int,
        rank: int,
        batch_size: int,
        start_micro_step: int = 0,
        seed: int = 0,
        samples_per_epoch: int = 0,
    ) -> None:
        self.root_sizes = [int(value) for value in root_sizes]
        if not self.root_sizes or any(value <= 0 for value in self.root_sizes):
            raise ValueError(f"root_sizes must be positive, got {self.root_sizes}")
        if len(probabilities) != len(self.root_sizes):
            raise ValueError(
                "probabilities/root_sizes length mismatch: "
                f"{len(probabilities)} != {len(self.root_sizes)}"
            )
        probability_sum = sum(float(value) for value in probabilities)
        if probability_sum <= 0.0:
            raise ValueError("sampling probabilities must sum to a positive value")
        self.probabilities = [float(value) / probability_sum for value in probabilities]
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {self.num_replicas}")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0,{self.num_replicas}), got {self.rank}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if int(start_micro_step) < 0:
            raise ValueError(f"start_micro_step must be non-negative, got {start_micro_step}")

        requested = int(samples_per_epoch) or sum(self.root_sizes)
        global_batch = self.num_replicas * self.batch_size
        self.global_samples_per_epoch = (
            (requested + global_batch - 1) // global_batch * global_batch
        )
        self.root_draws_per_epoch = _largest_remainder_counts(
            self.probabilities,
            self.global_samples_per_epoch,
        )
        self.samples_per_rank = self.global_samples_per_epoch // self.num_replicas
        if self.samples_per_rank % self.batch_size != 0:
            raise RuntimeError(
                "internal sampler epoch is not divisible by rank-local batch size"
            )
        self.micro_steps_per_epoch = self.samples_per_rank // self.batch_size

        consumed_rank_samples = int(start_micro_step) * self.batch_size
        self.start_epoch = consumed_rank_samples // self.samples_per_rank
        self.start_offset = consumed_rank_samples % self.samples_per_rank
        self.epoch = self.start_epoch

    @staticmethod
    def _generator(seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(seed) % (2**63 - 1))
        return generator

    def _root_stream(self, root_index: int, epoch: int, count: int) -> torch.Tensor:
        """Return the next root-local indices from a continuous permutation stream."""

        root_size = self.root_sizes[root_index]
        stream_start = int(epoch) * int(count)
        output = torch.empty(int(count), dtype=torch.int64)
        written = 0
        position = stream_start
        while written < count:
            cycle, cycle_offset = divmod(position, root_size)
            cycle_seed = self.seed + 1_000_003 * (root_index + 1) + 97_409 * cycle
            permutation = torch.randperm(
                root_size,
                generator=self._generator(cycle_seed),
                dtype=torch.int64,
            )
            take = min(count - written, root_size - cycle_offset)
            output[written : written + take] = permutation[
                cycle_offset : cycle_offset + take
            ]
            written += take
            position += take
        return output

    def _global_schedule(self, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        root_ids = torch.empty(self.global_samples_per_epoch, dtype=torch.int16)
        sample_ids = torch.empty(self.global_samples_per_epoch, dtype=torch.int64)
        cursor = 0
        for root_index, count in enumerate(self.root_draws_per_epoch):
            end = cursor + count
            root_ids[cursor:end] = root_index
            sample_ids[cursor:end] = self._root_stream(root_index, epoch, count)
            cursor = end
        if cursor != self.global_samples_per_epoch:
            raise RuntimeError(
                f"sampler schedule length mismatch: {cursor} != {self.global_samples_per_epoch}"
            )
        schedule_seed = self.seed + 2_147_483_647 + 65_537 * int(epoch)
        order = torch.randperm(
            self.global_samples_per_epoch,
            generator=self._generator(schedule_seed),
            dtype=torch.int64,
        )
        return root_ids[order], sample_ids[order]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        root_ids, sample_ids = self._global_schedule(self.epoch)
        rank_root_ids = root_ids[self.rank :: self.num_replicas]
        rank_sample_ids = sample_ids[self.rank :: self.num_replicas]
        offset = self.start_offset if self.epoch == self.start_epoch else 0
        return (
            (int(root_index), int(sample_index))
            for root_index, sample_index in zip(
                rank_root_ids[offset:].tolist(),
                rank_sample_ids[offset:].tolist(),
            )
        )

    def __len__(self) -> int:
        if self.epoch == self.start_epoch:
            return self.samples_per_rank - self.start_offset
        return self.samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def describe(self) -> dict[str, object]:
        return {
            "kind": "resumable_distributed_mixture_sampler",
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "batch_size": self.batch_size,
            "global_samples_per_epoch": self.global_samples_per_epoch,
            "samples_per_rank": self.samples_per_rank,
            "micro_steps_per_epoch": self.micro_steps_per_epoch,
            "root_sizes": list(self.root_sizes),
            "probabilities": list(self.probabilities),
            "root_draws_per_epoch": list(self.root_draws_per_epoch),
            "start_epoch": self.start_epoch,
            "start_offset": self.start_offset,
        }
