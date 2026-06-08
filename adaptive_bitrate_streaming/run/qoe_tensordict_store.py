"""ABR 测试结果 per-trace mean QoE 的 TensorDict .pt 存取约定。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
from tensordict import TensorDict

AVG_QOE_KEY = "avg_qoe"
QUALITY_KEY = "quality"
REBUFFER_PENALTY_KEY = "rebuffer_penalty"
SMOOTHNESS_PENALTY_KEY = "smoothness_penalty"

QOE_TD_KEYS: tuple[str, ...] = (
    AVG_QOE_KEY,
    QUALITY_KEY,
    REBUFFER_PENALTY_KEY,
    SMOOTHNESS_PENALTY_KEY,
)


@dataclass
class TraceBatchQoe:
    """每条轨迹一组标量（与 TensorDict batch 维对齐）。"""

    avg_qoe: list[float]
    quality: list[float]
    rebuffer_penalty: list[float]
    smoothness_penalty: list[float]

    def __post_init__(self) -> None:
        n = len(self.avg_qoe)
        if not all(
            len(getattr(self, k)) == n
            for k in ("quality", "rebuffer_penalty", "smoothness_penalty")
        ):
            raise ValueError("TraceBatchQoe 各字段长度须一致")

    @property
    def num_select(self) -> int:
        return len(self.avg_qoe)


def make_qoe_key(trace_slug: str, algo: str) -> str:
    """结果文件 stem：``{数据集}_{算法}``，如 ``fcc-test_mpc``。"""
    return f"{trace_slug}_{algo}"


def make_result_pt_path(output_dir: str, trace_slug: str, algo: str) -> str:
    return os.path.join(output_dir, f"{make_qoe_key(trace_slug, algo)}.pt")


def parse_qoe_key(stem: str) -> tuple[str, str]:
    """``fcc-test_mpc`` → (``fcc-test``, ``mpc``)。"""
    if "_" not in stem:
        raise ValueError(f"无效的 QoE 结果名: {stem!r}")
    trace_slug, algo = stem.rsplit("_", 1)
    return trace_slug, algo


def values_to_tensor(values: Iterable[float], *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=dtype)


def build_qoe_tensordict(batch: TraceBatchQoe) -> TensorDict:
    """
    ::

        TensorDict(
            {
                "avg_qoe": ...,
                "quality": ...,
                "rebuffer_penalty": ...,
                "smoothness_penalty": ...,
            },
            batch_size=[num_select],
        )
    """
    num_select = batch.num_select
    if num_select <= 0:
        raise ValueError("batch 不能为空")
    data = {
        AVG_QOE_KEY: values_to_tensor(batch.avg_qoe),
        QUALITY_KEY: values_to_tensor(batch.quality),
        REBUFFER_PENALTY_KEY: values_to_tensor(batch.rebuffer_penalty),
        SMOOTHNESS_PENALTY_KEY: values_to_tensor(batch.smoothness_penalty),
    }
    for key, tensor in data.items():
        if int(tensor.numel()) != num_select:
            raise ValueError(f"{key} 长度 {tensor.numel()} != num_select {num_select}")
    return TensorDict(data, batch_size=[num_select])


def save_qoe_result_pt(path: str, batch: TraceBatchQoe) -> TensorDict:
    """保存单个 (测试集, 算法) 的 .pt 文件。"""
    qoe_td = build_qoe_tensordict(batch)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(qoe_td, path)
    return qoe_td


def load_qoe_result_pt(path: str) -> TensorDict:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, TensorDict):
        raise TypeError(f"{path!r} 不是 TensorDict，得到 {type(obj)}")
    for key in QOE_TD_KEYS:
        if key not in obj.keys():
            raise KeyError(f"{path!r} 缺少字段 {key!r}")
    return obj


def avg_qoe_numpy(path: str):
    """从 .pt 读取 ``avg_qoe`` 为 numpy 1D 数组。"""
    import numpy as np

    td = load_qoe_result_pt(path)
    return td[AVG_QOE_KEY].detach().cpu().numpy().astype(np.float64, copy=False)


def load_component_means_from_pt(path: str) -> dict[str, float]:
    """
    从单个 .pt 读取各分量在 trace 维上的均值（与柱状图四组指标对应）。

    ``quality`` 即平均码率（Mbps）；惩罚项与 ``avg_qoe`` 满足
    avg_qoe ≈ quality - rebuffer_penalty - smoothness_penalty（逐 trace 精确）。
    """
    td = load_qoe_result_pt(path)
    num_traces = int(td.batch_size[0])
    return {
        "qoe": float(td[AVG_QOE_KEY].mean().item()),
        "bitrate_mbps": float(td[QUALITY_KEY].mean().item()),
        "rebuf_penalty": float(td[REBUFFER_PENALTY_KEY].mean().item()),
        "smooth_penalty": float(td[SMOOTHNESS_PENALTY_KEY].mean().item()),
        "num_traces": num_traces,
    }


def load_trace_algo_arrays_from_dir(output_dir: str, trace_slug: str) -> dict[str, "np.ndarray"]:
    """
    扫描目录下 ``{trace_slug}_{algo}.pt``，返回 ``{algo: avg_qoe ndarray}``。
    """
    if not os.path.isdir(output_dir):
        return {}
    prefix = f"{trace_slug}_"
    out: dict[str, "np.ndarray"] = {}
    for fn in sorted(os.listdir(output_dir)):
        if not fn.endswith(".pt") or fn.endswith("_meta.pt"):
            continue
        if not fn.startswith(prefix):
            continue
        stem = fn[: -len(".pt")]
        algo = stem[len(prefix) :]
        if not algo:
            continue
        out[algo] = avg_qoe_numpy(os.path.join(output_dir, fn))
    return out
