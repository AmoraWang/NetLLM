"""Merina 式 ABR 状态 (11 × hist) 的逐步更新，供 ABRLLM_v3 / test / evaluate 使用。"""
from __future__ import annotations

import torch

from baseline_special.utils.constants import (
    BITRATE_LEVELS,
    BUFFER_NORM_FACTOR,
    CHUNK_TIL_VIDEO_END_CAP,
    M_IN_K,
    MAX_VIDEO_BIT_RATE,
    VIDEO_BIT_RATE,
)


def merina_abr_apply_state_step_torch(
    state: torch.Tensor,
    bit_rate: int,
    delay: float,
    buffer_size: float,
    video_chunk_size: float,
    next_video_chunk_sizes,
    video_chunk_remain: float,
) -> torch.Tensor:
    """
    对 ``state`` 做 ``roll`` 后按 Merina ``test_v5`` 行序写入当前步观测。

    行语义：0 吞吐、1 缓冲、2 上一档码率、3 剩余分片比例、4–9 各档下一分片大小、10 下载时间。
    ``state`` 形状末尾为 ``(..., 11, H)``，``H`` 为历史长度（如 6）。
    """
    state = torch.roll(state, -1, dims=-1)
    d = torch.as_tensor(delay, device=state.device, dtype=torch.float32).clamp(min=1e-6)
    state[..., 0, -1] = float(video_chunk_size) / d / M_IN_K
    state[..., 1, -1] = buffer_size / BUFFER_NORM_FACTOR
    state[..., 2, -1] = VIDEO_BIT_RATE[int(bit_rate)] / MAX_VIDEO_BIT_RATE
    state[..., 3, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
    ns = torch.as_tensor(next_video_chunk_sizes, device=state.device, dtype=torch.float32) / M_IN_K / M_IN_K
    for i in range(BITRATE_LEVELS):
        state[..., 4 + i, -1] = ns[i]
    state[..., 10, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
    return state
