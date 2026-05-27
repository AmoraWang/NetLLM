"""Shared mahimahi start pointers for fair ABR evaluation across algorithms."""

from __future__ import annotations

import time
from typing import List, Sequence

import numpy as np


def derive_round_numpy_seed(round_idx: int, num_test_rounds: int, base_seed: int) -> int:
    """Per-round numpy seed; matches ``plm_special.test.test_on_env`` multi-round derivation."""
    if num_test_rounds <= 1:
        return int(base_seed) % (2**32)
    tns = int(time.time_ns())
    env_seed = (tns + round_idx * 1_000_003) ^ (int(base_seed) & 0xFFFFFFFF) ^ (round_idx * 0x9E3779B9)
    return int(env_seed % (2**32))


def build_mahimahi_ptrs_by_trace_index(
    all_cooked_bw: Sequence[Sequence[float]],
    *,
    seed: int,
    fixed: bool = True,
) -> List[int]:
    """
    Sample start pointers with the same rules as ``baseline_special.env.Environment``.

    Returns a list of length ``len(all_cooked_bw)`` indexed by **trace file index** ``i``,
    suitable for ``Environment`` when ``all_mahimahi_ptrs`` is pre-filled (reindexed by
    ``all_trace_indices`` inside the env).
    """
    n = len(all_cooked_bw)
    if n == 0:
        return []

    np.random.seed(int(seed) % (2**32))
    all_trace_indices = list(range(n))
    if not fixed:
        np.random.shuffle(all_trace_indices)

    ptr_by_file = [1] * n
    for idx in all_trace_indices:
        ln = len(all_cooked_bw[idx])
        if ln > 1:
            ptr_by_file[idx] = int(np.random.randint(1, ln))
        else:
            ptr_by_file[idx] = 1
    return ptr_by_file


def apply_test_round_mahimahi_ptrs(
    env_settings: dict,
    *,
    base_seed: int,
    round_idx: int = 0,
    num_test_rounds: int = 1,
) -> int:
    """
    Fill ``env_settings['all_mahimahi_ptrs']`` for one test round.

    Returns the numpy seed used for this round.
    """
    round_seed = derive_round_numpy_seed(round_idx, num_test_rounds, base_seed)
    ptrs = build_mahimahi_ptrs_by_trace_index(
        env_settings["all_cooked_bw"],
        seed=round_seed,
        fixed=bool(env_settings.get("fixed", True)),
    )
    env_settings["all_mahimahi_ptrs"] = ptrs
    return round_seed


def _clamp_ptr(ptr: int, trace_len: int) -> int:
    if trace_len <= 1:
        return 1
    return max(1, min(int(ptr), trace_len - 1))


def merina_apply_mahimahi_ptr(env, ptrs_by_trace_index: Sequence[int]) -> None:
    """Set Merina env start pointer for ``env.trace_idx`` (after trace switch)."""
    idx = int(env.trace_idx)
    n = len(env.cooked_bw)
    if idx < len(ptrs_by_trace_index):
        ptr = _clamp_ptr(ptrs_by_trace_index[idx], n)
    else:
        ptr = 1
    env.mahimahi_ptr = ptr
    env.last_mahimahi_time = env.cooked_time[env.mahimahi_ptr - 1]


def wrap_merina_env_with_shared_ptrs(merina_env_class):
    """Return a Merina Environment subclass that uses a fixed per-trace pointer table."""

    class SharedPtrMerinaEnv(merina_env_class):
        def __init__(self, *args, mahimahi_ptrs_by_trace_index=None, **kwargs):
            kwargs.pop("random_seed", None)
            self._mahimahi_ptrs = list(mahimahi_ptrs_by_trace_index or [])
            super().__init__(*args, **kwargs)
            merina_apply_mahimahi_ptr(self, self._mahimahi_ptrs)

        def get_video_chunk(self, quality):
            out = super().get_video_chunk(quality)
            if out[6]:
                merina_apply_mahimahi_ptr(self, self._mahimahi_ptrs)
            return out

    SharedPtrMerinaEnv.__name__ = f"SharedPtr{merina_env_class.__name__}"
    return SharedPtrMerinaEnv
