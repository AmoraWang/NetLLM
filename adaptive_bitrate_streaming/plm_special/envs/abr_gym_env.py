#!/usr/bin/env python3
"""Gym-style ABR environment wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from baseline_special.env import Environment
from baseline_special.utils.constants import (
    A_DIM,
    BITRATE_LEVELS,
    BUFFER_NORM_FACTOR,
    CHUNK_TIL_VIDEO_END_CAP,
    DEFAULT_QUALITY,
    M_IN_K,
    MAX_VIDEO_BIT_RATE,
    REBUF_PENALTY,
    S_INFO,
    S_LEN,
    SMOOTH_PENALTY,
    VIDEO_BIT_RATE,
)
from plm_special.utils.utils import action2bitrate

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


@dataclass
class AbrStepInfo:
    trace_idx: int
    bitrate: int
    delay_ms: float
    sleep_time_ms: float
    buffer_s: float
    rebuffer_s: float
    chunk_size_bytes: int
    video_chunk_remain: float


class ABRGymEnv(gym.Env if gym is not None else object):
    """ABR RL environment with Gymnasium-compatible reset/step API."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_settings: Dict[str, Any],
        *,
        action_mode: str = "bitrate",
        skip_first_reward: bool = True,
    ):
        if action_mode not in ("bitrate", "jump"):
            raise ValueError("action_mode must be 'bitrate' or 'jump'")

        self.env_settings = dict(env_settings)
        self.action_mode = action_mode
        self.skip_first_reward = skip_first_reward

        self._env: Optional[Environment] = None
        self._state = np.zeros((S_INFO, S_LEN), dtype=np.float32)
        self._last_bit_rate = DEFAULT_QUALITY
        self._bit_rate = DEFAULT_QUALITY
        self._timestep = 0

        if spaces is not None:
            if action_mode == "bitrate":
                self.action_space = spaces.Discrete(BITRATE_LEVELS)
            else:
                self.action_space = spaces.Discrete(A_DIM)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(S_INFO, S_LEN),
                dtype=np.float32,
            )

    def _build_env(self) -> Environment:
        return Environment(**self.env_settings)

    def _obs(self) -> np.ndarray:
        return self._state.copy()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        del options
        if seed is not None:
            np.random.seed(seed)

        self._env = self._build_env()
        self._state.fill(0.0)
        self._last_bit_rate = DEFAULT_QUALITY
        self._bit_rate = DEFAULT_QUALITY
        self._timestep = 0

        info = {
            "trace_idx": self._env.trace_idx,
            "action_mode": self.action_mode,
        }
        return self._obs(), info

    def _decode_action(self, action: int) -> int:
        if self.action_mode == "bitrate":
            bit_rate = int(action)
            if bit_rate < 0 or bit_rate >= BITRATE_LEVELS:
                raise ValueError(f"bitrate action out of range: {bit_rate}")
            return bit_rate

        jump_action = int(action)
        if jump_action < 0 or jump_action >= A_DIM:
            raise ValueError(f"jump action out of range: {jump_action}")
        return action2bitrate(jump_action, self._last_bit_rate)

    def step(self, action: int):
        if self._env is None:
            raise RuntimeError("Environment must be reset before step().")

        self._bit_rate = self._decode_action(action)

        (
            delay,
            sleep_time,
            buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = self._env.get_video_chunk(self._bit_rate)

        reward = (
            VIDEO_BIT_RATE[self._bit_rate] / M_IN_K
            - REBUF_PENALTY * rebuf
            - SMOOTH_PENALTY
            * abs(VIDEO_BIT_RATE[self._bit_rate] - VIDEO_BIT_RATE[self._last_bit_rate])
            / M_IN_K
        )

        self._state = np.roll(self._state, -1, axis=1)
        safe_delay = max(float(delay), 1e-6)
        self._state[0, -1] = VIDEO_BIT_RATE[self._bit_rate] / float(MAX_VIDEO_BIT_RATE)
        self._state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        self._state[2, -1] = float(video_chunk_size) / safe_delay / M_IN_K
        self._state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        self._state[4, :BITRATE_LEVELS] = (
            np.array(next_video_chunk_sizes, dtype=np.float32) / M_IN_K / M_IN_K
        )
        self._state[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP
        )

        if self.skip_first_reward and self._timestep == 0:
            reward = 0.0

        info = AbrStepInfo(
            trace_idx=self._env.trace_idx,
            bitrate=int(self._bit_rate),
            delay_ms=float(delay),
            sleep_time_ms=float(sleep_time),
            buffer_s=float(buffer_size),
            rebuffer_s=float(rebuf),
            chunk_size_bytes=int(video_chunk_size),
            video_chunk_remain=float(video_chunk_remain),
        ).__dict__

        self._last_bit_rate = self._bit_rate
        self._timestep += 1

        terminated = bool(end_of_video)
        truncated = False

        if terminated:
            self._last_bit_rate = DEFAULT_QUALITY
            self._bit_rate = DEFAULT_QUALITY
            self._state.fill(0.0)
            self._timestep = 0

        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self._env = None
