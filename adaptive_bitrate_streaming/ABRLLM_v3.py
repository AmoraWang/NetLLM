"""
ABRLLM v3：NetLLM (6, 6) 状态 + state/action 语义模板（供 LLM 对齐与对比学习）。

状态行语义（与 ``generate_exp_pool`` / ``ABRLLM_v2.StateEncoder`` 一致）：
  0  上一档码率（相对 MAX_VIDEO_BIT_RATE）
  1  播放缓冲（秒，已除以 BUFFER_NORM_FACTOR）
  2  历史吞吐（size/delay/M_IN_K，各列为过去时刻）
  3  历史下载时间（归一化）
  4  各码率下一分片大小（列 0–5 对应 6 档）
  5  剩余分片（相对 CHUNK_TIL_VIDEO_END_CAP）

对比损失（``abrllm_v3_core.ABRLLM``）：
  ``state_aligned`` = alignment_layer 输出（未加 timestep）；
  文本 = 冻结 PLM 对 ``render_joint_state_action_text`` 的均值池化；
  ``contrast_proj_*`` = Linear → LeakyReLU → Linear 映射到对比空间；
  InfoNCE 在 ``plm_special/utils/losses.py``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from baseline_special.utils.constants import (
    BITRATE_LEVELS,
    BUFFER_NORM_FACTOR,
    M_IN_K,
    MAX_VIDEO_BIT_RATE,
    S_INFO,
    S_LEN,
    VIDEO_BIT_RATE,
)

# ---------------------------------------------------------------------------
# 语义模板：state / action 分离；joint 用于后续 (state_embed, text) 对比
# ---------------------------------------------------------------------------

STATE_ROW_LAST_BITRATE = 0
STATE_ROW_BUFFER = 1
STATE_ROW_THROUGHPUT = 2
STATE_ROW_DOWNLOAD = 3
STATE_ROW_NEXT_CHUNK = 4
STATE_ROW_REMAIN = 5


@dataclass(frozen=True)
class SemanticTemplateConfig:
    """控制自然语言粒度；训练时可固定以便预计算 text_embed。"""

    language: str = "en"
    include_history: bool = True
    throughput_trend_window: int = 3


DEFAULT_SEMANTIC_CONFIG = SemanticTemplateConfig()


def _as_state_matrix(state: Any) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float64)
    if arr.shape != (S_INFO, S_LEN):
        raise ValueError(f"state 形状应为 ({S_INFO}, {S_LEN})，得到 {arr.shape}")
    return arr


def _latest(state: np.ndarray, row: int) -> float:
    return float(state[row, -1])


def _history(state: np.ndarray, row: int) -> np.ndarray:
    return state[row, :].astype(np.float64)


def _buffer_seconds(norm_val: float) -> float:
    return float(norm_val) * float(BUFFER_NORM_FACTOR)


def _bitrate_kbps(norm_val: float) -> float:
    return float(norm_val) * float(MAX_VIDEO_BIT_RATE)


def _throughput_mbps(norm_val: float) -> float:
    """行 2：与仿真中 size/delay/M_IN_K 一致 (Mbps)。"""
    return max(0.0, float(norm_val))


def _download_ms(norm_val: float) -> float:
    """行 3 反归一化为毫秒量级下载时间。"""
    return float(norm_val) * float(M_IN_K) * float(BUFFER_NORM_FACTOR)


def _bucket_buffer(sec: float) -> str:
    if sec < 2.0:
        return "critically low (high stall risk)"
    if sec < 6.0:
        return "moderate"
    if sec < 15.0:
        return "comfortable"
    return "very high"


def _bucket_throughput_level(mbps: float) -> str:
    if mbps < 0.35:
        return "very poor"
    if mbps < 0.8:
        return "limited"
    if mbps < 1.8:
        return "moderate"
    if mbps < 3.5:
        return "good"
    return "excellent"


def _describe_throughput_trend(hist: np.ndarray, cfg: SemanticTemplateConfig) -> str:
    if not cfg.include_history or hist.size < 2:
        return ""
    w = min(cfg.throughput_trend_window, hist.size // 2)
    if w < 1:
        return ""
    early = np.mean(hist[:w])
    late = np.mean(hist[-w:])
    early_m = _throughput_mbps(early)
    late_m = _throughput_mbps(late)
    delta = late_m - early_m
    if abs(delta) < 0.08:
        return " Throughput has been stable recently."
    if delta > 0:
        return f" Throughput is rising (about {early_m:.2f} → {late_m:.2f} Mbps)."
    return f" Throughput is falling (about {early_m:.2f} → {late_m:.2f} Mbps)."


def _describe_next_chunk_sizes(state: np.ndarray) -> str:
    """行 4 各列 → 各档下一分片相对大小（已归一化）。"""
    sizes = [float(state[STATE_ROW_NEXT_CHUNK, i]) for i in range(BITRATE_LEVELS)]
    best = int(np.argmax(sizes))
    kbps = VIDEO_BIT_RATE[best]
    return (
        f" The next segment is largest at level {best} ({kbps} kbps) "
        f"relative to other bitrate options."
    )


def render_state_semantic_text(
    state: Any,
    *,
    config: SemanticTemplateConfig | None = None,
) -> str:
    """
    仅描述网络/播放器观测（不含专家动作）。
    用于 state 侧文本锚点或拼接前的 state 模态。
    """
    cfg = config or DEFAULT_SEMANTIC_CONFIG
    s = _as_state_matrix(state)

    br_kbps = _bitrate_kbps(_latest(s, STATE_ROW_LAST_BITRATE))
    buf_s = _buffer_seconds(_latest(s, STATE_ROW_BUFFER))
    thr_m = _throughput_mbps(_latest(s, STATE_ROW_THROUGHPUT))
    dl_ms = _download_ms(_latest(s, STATE_ROW_DOWNLOAD))
    remain_pct = 100.0 * _latest(s, STATE_ROW_REMAIN)
    thr_hist = _history(s, STATE_ROW_THROUGHPUT)

    parts = [
        "The network and player state is as follows.",
        f" The player is currently using about {br_kbps:.0f} kbps.",
        f" Playback buffer is {_bucket_buffer(buf_s)} at about {buf_s:.1f} s.",
        f" Recent throughput is {_bucket_throughput_level(thr_m)} at about {thr_m:.2f} Mbps.",
        f" The last chunk download took about {dl_ms:.0f} ms.",
        f" Roughly {remain_pct:.0f}% of the video remains to be delivered.",
    ]
    parts.append(_describe_throughput_trend(thr_hist, cfg))
    parts.append(_describe_next_chunk_sizes(s))
    return "".join(parts)


def _action_index(action: Any) -> int:
    if isinstance(action, (int, np.integer)):
        a = int(action)
    else:
        a = int(np.asarray(action).reshape(-1)[0])
    if not 0 <= a < BITRATE_LEVELS:
        raise ValueError(f"action 应在 [0, {BITRATE_LEVELS - 1}]，得到 {a}")
    return a


def render_action_semantic_text(
    action: Any,
    *,
    prev_action: int | None = None,
    config: SemanticTemplateConfig | None = None,
) -> str:
    """
    仅描述专家码率决策（不含 state）。
    ``prev_action`` 可选，用于说明切换幅度。
    """
    _ = config or DEFAULT_SEMANTIC_CONFIG
    a = _action_index(action)
    kbps = VIDEO_BIT_RATE[a]

    base = (
        f"The expert selects bitrate level {a} ({kbps} kbps) for the next video chunk."
    )
    if prev_action is None or prev_action == a:
        return base

    prev_action = int(prev_action)
    prev_kbps = VIDEO_BIT_RATE[prev_action]
    delta = kbps - prev_kbps
    if delta > 0:
        switch = f"an increase from level {prev_action} ({prev_kbps} kbps)"
    elif delta < 0:
        switch = f"a decrease from level {prev_action} ({prev_kbps} kbps)"
    else:
        switch = f"the same level as before ({prev_kbps} kbps)"
    return f"{base} This is {switch}."


def render_joint_state_action_text(
    state: Any,
    action: Any,
    *,
    prev_action: int | None = None,
    config: SemanticTemplateConfig | None = None,
) -> str:
    """
    联合模板：显式因果句式，供 (state_embed, text) 跨模态对齐。
    """
    cfg = config or DEFAULT_SEMANTIC_CONFIG
    state_txt = render_state_semantic_text(state, config=cfg)
    action_txt = render_action_semantic_text(action, prev_action=prev_action, config=cfg)
    return (
        f"{state_txt} "
        f"Given this network condition, {action_txt[0].lower() + action_txt[1:]}"
    )


def render_contrastive_action_texts(
    state: Any,
    expert_action: Any,
    *,
    negative_actions: Iterable[int] | None = None,
    prev_action: int | None = None,
    config: SemanticTemplateConfig | None = None,
) -> dict[str, str]:
    """
    生成对比学习用文本：``positive`` 为专家动作；``negative_*`` 为错动作（默认其余 5 档）。
    """
    cfg = config or DEFAULT_SEMANTIC_CONFIG
    expert = _action_index(expert_action)
    negs = list(negative_actions) if negative_actions is not None else [
        i for i in range(BITRATE_LEVELS) if i != expert
    ]
    out = {
        "positive": render_joint_state_action_text(
            state, expert, prev_action=prev_action, config=cfg
        ),
        "state_only": render_state_semantic_text(state, config=cfg),
        "action_positive": render_action_semantic_text(
            expert, prev_action=prev_action, config=cfg
        ),
    }
    for na in negs:
        out[f"negative_{na}"] = render_joint_state_action_text(
            state, na, prev_action=prev_action, config=cfg
        )
    return out


def batch_render_joint_texts(
    states: Sequence[Any],
    actions: Sequence[Any],
    *,
    prev_actions: Sequence[int | None] | None = None,
    config: SemanticTemplateConfig | None = None,
) -> list[str]:
    """批量生成联合描述（例如写回经验池或预计算 PLM embedding）。"""
    cfg = config or DEFAULT_SEMANTIC_CONFIG
    n = len(states)
    if len(actions) != n:
        raise ValueError("states 与 actions 长度须一致")
    if prev_actions is not None and len(prev_actions) != n:
        raise ValueError("prev_actions 长度须与 states 一致")
    texts: list[str] = []
    for i in range(n):
        pa = None if prev_actions is None else prev_actions[i]
        texts.append(
            render_joint_state_action_text(states[i], actions[i], prev_action=pa, config=cfg)
        )
    return texts


# ---------------------------------------------------------------------------
# 模型：abrllm_v3_core（自 v2 复制，不 import ABRLLM_v2）+ 对比对齐
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    if name == 'ABRLLM':
        from abrllm_v3_core import ABRLLM as _ABRLLM

        return _ABRLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ABRLLM",
    "SemanticTemplateConfig",
    "DEFAULT_SEMANTIC_CONFIG",
    "render_state_semantic_text",
    "render_action_semantic_text",
    "render_joint_state_action_text",
    "render_contrastive_action_texts",
    "batch_render_joint_texts",
    "S_INFO",
    "S_LEN",
    "STATE_ROW_LAST_BITRATE",
    "STATE_ROW_BUFFER",
    "STATE_ROW_THROUGHPUT",
    "STATE_ROW_DOWNLOAD",
    "STATE_ROW_NEXT_CHUNK",
    "STATE_ROW_REMAIN",
]
