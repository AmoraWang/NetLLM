"""默认绘图算法列表与图例显示名（各 plot_*.py 共用）。"""

from __future__ import annotations

# 内部目录键名（与 build_preset_paths 一致）
DEFAULT_PLOT_ALGOS: tuple[str, ...] = (
    "ours",
    "BOLA",
    "RobustMPC",
    "Pensieve",
    "Comyco",
    "Oracle",
)

# 图上图例显示名
ALGO_DISPLAY_NAMES: dict[str, str] = {
    "BOLA": "BOLA",
    "RobustMPC": "RobustMPC",
    "Pensieve": "Pensieve",
    "MERINA": "Merina",
    "Comyco": "Comyco",
    "Oracle": "Oracle",
    "ours": "Ours",
    "Fugu": "Fugu",
    "BayesMPC": "BayesMPC",
}

# Ours 相对对比的 baseline（不含 Ours 自身）
DEFAULT_BASELINE_ALGOS: tuple[str, ...] = tuple(
    k for k in DEFAULT_PLOT_ALGOS if k != "ours"
)


def algo_display_name(key: str) -> str:
    return ALGO_DISPLAY_NAMES.get(key, key)


def is_plot_excluded_algo(name: str) -> bool:
    """绘图时跳过 Merina（collect 仍可保留 merina .pt）。"""
    return name.lower() == "merina"


def filter_plot_algos(data: dict) -> dict:
    """去掉不参与绘图的算法。"""
    return {k: v for k, v in data.items() if not is_plot_excluded_algo(k)}


def order_algo_dict(data: dict, *, order: tuple[str, ...] = DEFAULT_PLOT_ALGOS) -> dict:
    """按默认顺序重排，未列出的键附在末尾。"""
    out: dict = {}
    for k in order:
        if k in data:
            out[k] = data[k]
    for k, v in data.items():
        if k not in out:
            out[k] = v
    return out


def filter_algo_dirs(
    algo_dirs: dict[str, str],
    *,
    allowed: tuple[str, ...] = DEFAULT_PLOT_ALGOS,
) -> dict[str, str]:
    return {k: v for k, v in algo_dirs.items() if k in allowed}
