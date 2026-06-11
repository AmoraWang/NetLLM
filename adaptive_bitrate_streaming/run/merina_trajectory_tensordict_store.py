"""Merina 单条带宽 trace 决策轨迹的 TensorDict .pt 存取约定。"""
from __future__ import annotations

import os
import re
from typing import Any

import torch
from tensordict import TensorDict

BUFFER_SIZE_KEY = "buffer_size"
ACTION_KEY = "action"
ACTION_PROB_KEY = "action_prob"
BELIEF_MU_KEY = "belief_mu"
BELIEF_LOGVAR_KEY = "belief_logvar"
BELIEF_LATENT_KEY = "belief_latent"
REWARD_KEY = "reward"
REBUFFER_KEY = "rebuffer"
DONE_KEY = "done"

TRAJECTORY_TD_KEYS: tuple[str, ...] = (
    BUFFER_SIZE_KEY,
    ACTION_KEY,
    ACTION_PROB_KEY,
    BELIEF_MU_KEY,
    BELIEF_LOGVAR_KEY,
    BELIEF_LATENT_KEY,
    REWARD_KEY,
    REBUFFER_KEY,
    DONE_KEY,
)


def sanitize_trace_stem(trace_name: str, *, max_len: int = 120) -> str:
    """将 trace 文件名转为安全的路径 stem。"""
    stem = re.sub(r"[^\w.\-]+", "_", str(trace_name).strip())
    stem = stem.strip("._") or "trace"
    return stem[:max_len]


def make_trace_pt_path(output_dir: str, trace_index: int, trace_name: str) -> str:
    stem = sanitize_trace_stem(trace_name)
    return os.path.join(output_dir, f"{int(trace_index):04d}_{stem}.pt")


def build_merina_trajectory_tensordict(
    *,
    buffer_size: torch.Tensor,
    action: torch.Tensor,
    action_prob: torch.Tensor,
    belief_mu: torch.Tensor,
    belief_logvar: torch.Tensor,
    belief_latent: torch.Tensor,
    reward: torch.Tensor,
    rebuffer: torch.Tensor,
    done: torch.Tensor,
) -> TensorDict:
    """
    构造单条 trace 的轨迹 TensorDict，batch 维为时间步 T。

    - ``buffer_size`` (T,) 秒，chunk 下载后的播放器缓冲
    - ``action`` (T,) int64，Actor 在本步选出的下一 chunk 码率档位 0–5
    - ``action_prob`` (T, 6) Actor Softmax
    - ``belief_mu`` / ``belief_logvar`` (T, D) Merina β-VAE 后验参数
    - ``belief_latent`` (T, D) 送入 Actor 的 latent z（``get_latent`` 采样结果）
    """
    tensors = {
        BUFFER_SIZE_KEY: buffer_size.reshape(-1).to(torch.float32),
        ACTION_KEY: action.reshape(-1).to(torch.int64),
        ACTION_PROB_KEY: action_prob.reshape(-1, 6).to(torch.float32),
        BELIEF_MU_KEY: belief_mu.reshape(-1, belief_mu.shape[-1]).to(torch.float32),
        BELIEF_LOGVAR_KEY: belief_logvar.reshape(-1, belief_logvar.shape[-1]).to(torch.float32),
        BELIEF_LATENT_KEY: belief_latent.reshape(-1, belief_latent.shape[-1]).to(torch.float32),
        REWARD_KEY: reward.reshape(-1).to(torch.float32),
        REBUFFER_KEY: rebuffer.reshape(-1).to(torch.float32),
        DONE_KEY: done.reshape(-1).to(torch.bool),
    }
    num_steps = int(tensors[ACTION_KEY].shape[0])
    if num_steps <= 0:
        raise ValueError("轨迹步数须 > 0")
    for key, tensor in tensors.items():
        if int(tensor.shape[0]) != num_steps:
            raise ValueError(f"{key} 长度 {tensor.shape[0]} != T={num_steps}")
    return TensorDict(tensors, batch_size=[num_steps])


def save_merina_trajectory_pt(
    path: str,
    trajectory: TensorDict,
    meta: dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"trajectory": trajectory, "meta": meta or {}}
    torch.save(payload, path)


def load_merina_trajectory_pt(path: str) -> tuple[TensorDict, dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "trajectory" not in obj:
        raise TypeError(f"{path!r} 格式无效，需要 dict 含 'trajectory'")
    trajectory = obj["trajectory"]
    if not isinstance(trajectory, TensorDict):
        raise TypeError(f"{path!r} trajectory 不是 TensorDict，得到 {type(trajectory)}")
    for key in TRAJECTORY_TD_KEYS:
        if key not in trajectory.keys():
            raise KeyError(f"{path!r} 缺少字段 {key!r}")
    meta = obj.get("meta") or {}
    if not isinstance(meta, dict):
        raise TypeError(f"{path!r} meta 应为 dict，得到 {type(meta)}")
    return trajectory, meta
