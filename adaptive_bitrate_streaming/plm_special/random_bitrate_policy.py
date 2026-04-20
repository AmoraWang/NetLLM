"""
均匀随机码率策略：每步独立从 {0, …, BITRATE_LEVELS-1} 中均匀采样，与 ABRLLM.sample 接口兼容。
"""
import random

from baseline_special.utils.constants import BITRATE_LEVELS


class RandomUniformBitratePolicy:
    """每步随机选一个码率档位；忽略 state / target_return / timestep。"""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def sample(self, _state, _target_return, _timestep, **_kwargs):
        return self._rng.randint(0, BITRATE_LEVELS - 1)
