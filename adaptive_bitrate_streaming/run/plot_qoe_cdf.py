#!/usr/bin/env python3
"""
绘制测试集上各 ABR 算法的 QoE CDF（累积分布函数）。

默认指标（--metric trace_mean）：
  每条带宽轨迹上完整播完一个视频 → 去掉第 1 个 chunk 的 QoE 后，对剩余块求平均
  → 得到每条 trace 一个标量 → 对这些 trace 均值做 CDF（纵轴为 trace 占比）。

可选（--metric per_chunk）：
  横轴为每个 chunk 的 QoE，纵轴为 chunk 占比。

数据来源（自动递归扫描目录）：
  - NetLLM ``plm_special/test`` 格式：``result_sim_abr_*``，8 列 tab，第 8 列 reward
  - Merina 基线格式：``log_test_*``，空格/制表符分隔，最后一列 reward

默认横轴范围 [-2, 3]：该范围外的样本不参与 CDF（可用 --xmin / --xmax 调整）。

示例：
  cd adaptive_bitrate_streaming
  bash bash/run_plot_cdf.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)

import matplotlib.pyplot as plt
import numpy as np

from config import cfg

from run.plot_algo_defaults import (
    DEFAULT_PLOT_ALGOS,
    algo_display_name,
    filter_algo_dirs,
    order_algo_dict,
)

# 默认顺序：Ours 首位，其后 BOLA … Oracle；Ours 单独配色
DEFAULT_LINE_STYLES = ["--", ":", "-.", "--", ":", "-.", "-"]
DEFAULT_COLORS = [
    "#1f77b4",  # BOLA
    "#ff7f0e",  # RobustMPC
    "#2ca02c",  # Pensieve
    "#e377c2",  # Merina
    "#d62728",  # Comyco
    "#9467bd",  # Oracle
]

OURS_LINE_STYLE = "-"
OURS_COLOR = "#c44e52"  # Ours

LOG_NAME_HINTS = ("result_sim_abr_", "log_test_")
TRACE_KEY_PREFIXES = (
    "result_sim_abr_",
    "log_test_comyco_merina_",
    "log_test_merina_",
)
DEFAULT_NUM_BINS = 500


def trace_key_from_log_path(path: str) -> str:
    """从日志文件名提取 trace 标识，用于跨算法按同一条轨迹对齐。"""
    base = os.path.basename(path)
    for prefix in TRACE_KEY_PREFIXES:
        if base.startswith(prefix):
            return base[len(prefix) :]
    return base


def _parse_algo_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--algo 需为 NAME=PATH 形式，收到: {spec!r}")
    name, path = spec.split("=", 1)
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise ValueError(f"--algo 需为 NAME=PATH 形式，收到: {spec!r}")
    return name, path


def find_log_files(root: str, name_hints: Iterable[str] = LOG_NAME_HINTS) -> list[str]:
    """递归查找测试结果日志文件。"""
    if not os.path.isdir(root):
        return []
    hints = tuple(name_hints)
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("."):
                continue
            if any(h in fn for h in hints):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def read_all_chunk_rewards(path: str) -> list[float]:
    """从单条 trace 日志读取全部 chunk 的 QoE（不做首尾裁剪）。"""
    rewards: list[float] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                if len(parts) >= 8:
                    rewards.append(float(parts[7]))
                else:
                    rewards.append(float(parts[-1]))
            except ValueError:
                continue
    return rewards


def read_chunk_rewards_from_file(
    path: str,
    *,
    skip_first: bool = True,
    skip_last: bool = True,
) -> list[float]:
    """
    从单条 trace 日志读取每块 QoE，可选去掉首尾块。
    NetLLM 8 列 → parse[7]；Merina 7 列 → 最后一列。
    """
    rewards = read_all_chunk_rewards(path)
    if skip_first and rewards:
        rewards = rewards[1:]
    if skip_last and rewards:
        rewards = rewards[:-1]
    return rewards


def trace_mean_qoe_from_file(
    path: str,
    *,
    skip_first_chunk: bool = True,
    skip_last_chunk: bool = False,
) -> float | None:
    """单条 trace：完整视频在一条轨迹上，除去首块（及可选末块）后的平均 QoE。"""
    rewards = read_all_chunk_rewards(path)
    if skip_first_chunk and rewards:
        rewards = rewards[1:]
    if skip_last_chunk and rewards:
        rewards = rewards[:-1]
    if not rewards:
        return None
    return float(np.mean(rewards))


def collect_trace_mean_qoe_by_key(
    log_root: str,
    *,
    skip_first_chunk: bool = True,
    skip_last_chunk: bool = False,
) -> dict[str, float]:
    """每条 trace 一个均值 QoE，键为 ``trace_key_from_log_path``。"""
    files = find_log_files(log_root)
    if not files:
        raise FileNotFoundError(
            f"在 {log_root!r} 下未找到测试结果日志（需含 {LOG_NAME_HINTS} 的文件名）"
        )
    out: dict[str, float] = {}
    for fp in files:
        m = trace_mean_qoe_from_file(
            fp,
            skip_first_chunk=skip_first_chunk,
            skip_last_chunk=skip_last_chunk,
        )
        if m is not None:
            out[trace_key_from_log_path(fp)] = m
    if not out:
        raise ValueError(f"{log_root!r} 中无法计算 trace 平均 QoE")
    return out


def collect_trace_mean_qoe(
    log_root: str,
    *,
    skip_first_chunk: bool = True,
    skip_last_chunk: bool = False,
) -> np.ndarray:
    """每条 trace 一个均值（去掉首块后的平均 QoE），再汇总为数组。"""
    files = find_log_files(log_root)
    if not files:
        raise FileNotFoundError(
            f"在 {log_root!r} 下未找到测试结果日志（需含 {LOG_NAME_HINTS} 的文件名）"
        )
    means: list[float] = []
    for fp in files:
        m = trace_mean_qoe_from_file(
            fp,
            skip_first_chunk=skip_first_chunk,
            skip_last_chunk=skip_last_chunk,
        )
        if m is not None:
            means.append(m)
    if not means:
        raise ValueError(f"{log_root!r} 中无法计算 trace 平均 QoE")
    return np.asarray(means, dtype=np.float64)


def collect_chunk_qoe(
    log_root: str,
    *,
    skip_first: bool = True,
    skip_last: bool = True,
) -> np.ndarray:
    """汇总某算法目录下全部 trace 的中间块 QoE。"""
    files = find_log_files(log_root)
    if not files:
        raise FileNotFoundError(
            f"在 {log_root!r} 下未找到测试结果日志（需含 {LOG_NAME_HINTS} 的文件名）"
        )
    pooled: list[float] = []
    for fp in files:
        pooled.extend(
            read_chunk_rewards_from_file(fp, skip_first=skip_first, skip_last=skip_last)
        )
    if not pooled:
        raise ValueError(f"{log_root!r} 中日志无有效 QoE 行")
    return np.asarray(pooled, dtype=np.float64)


def compute_hist_cdf(
    values: np.ndarray,
    xmin: float,
    xmax: float,
    num_bins: int = DEFAULT_NUM_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    直方图 CDF（与 merina/utils/plt_v2.py 一致）：先分箱再累积，曲线平滑。
    返回 (bin_edges, cumulative)，长度均为 num_bins + 1。
    """
    plot_values = truncate_qoe(values, xmin, xmax)
    if plot_values.size == 0:
        return np.array([]), np.array([])
    counts, base = np.histogram(plot_values, bins=num_bins, range=(xmin, xmax))
    cumulative = np.cumsum(counts, dtype=np.float64) / float(plot_values.size)
    cumulative = np.insert(cumulative, 0, 0.0)
    return base, cumulative


def truncate_qoe(values: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """保留 [xmin, xmax] 范围内的 QoE，范围外的样本丢弃。"""
    if xmin > xmax:
        raise ValueError(f"xmin ({xmin}) 不能大于 xmax ({xmax})")
    return values[(values >= xmin) & (values <= xmax)]


def build_preset_paths(
    preset: str,
    *,
    trace: str,
    video: str,
    trace_num: int,
    fixed_order: bool,
    seed: int,
    test_rounds: int,
    ours_dir: str | None = None,
) -> dict[str, str]:
    """根据仓库测试脚本约定，构造常见算法日志目录。"""
    rounds_tag = f"seed_{seed}_rounds_{test_rounds}"
    fixed = str(fixed_order)
    base = os.path.join(cfg.results_dir, f"{trace}_{video}")

    if preset == "netllm_baselines":
        trace_tag = f"trace_num_{trace_num}_fixed_{fixed}"
        root = os.path.join(base, trace_tag)
        paths = {
            "BOLA": os.path.join(root, "rule_baseline", "bba", rounds_tag),
            "RobustMPC": os.path.join(root, "rule_baseline", "mpc", rounds_tag),
            "Pensieve": os.path.join(root, "rl_baseline", "genet", rounds_tag),
            "MERINA": os.path.join(base, f"trace_num_{trace_num}_merina", rounds_tag),
            "Comyco": os.path.join(base, f"trace_num_{trace_num}_comyco_merina", rounds_tag),
            "Oracle": os.path.join(root, "DP", rounds_tag),
        }
        if ours_dir:
            paths["ours"] = ours_dir
        return paths

    if preset == "paper_seven":
        paths = build_preset_paths(
            "netllm_baselines",
            trace=trace,
            video=video,
            trace_num=trace_num,
            fixed_order=fixed_order,
            seed=seed,
            test_rounds=test_rounds,
            ours_dir=ours_dir,
        )
        paths["Fugu"] = os.path.join(base, f"trace_num_{trace_num}_fugu_merina", rounds_tag)
        paths["BayesMPC"] = os.path.join(base, f"trace_num_{trace_num}_bayes_merina", rounds_tag)
        return paths

    raise ValueError(f"未知 preset: {preset!r}")


def plot_qoe_cdf(
    algo_qoe: dict[str, np.ndarray],
    *,
    output_path: str | None = None,
    title: str | None = None,
    xlabel: str = "Average Values of Chunk's QoE",
    ylabel: str = "CDF (Perc. of sessions)",
    xlim: tuple[float, float] = (-2.0, 3.0),
    ylim: tuple[float, float] = (0.0, 1.03),
    num_bins: int = DEFAULT_NUM_BINS,
    line_styles: list[str] | None = None,
    colors: list[str] | None = None,
    show_better_arrow: bool = True,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> plt.Figure:
    line_styles = line_styles or DEFAULT_LINE_STYLES
    colors = colors or DEFAULT_COLORS
    xmin, xmax = xlim
    ymin, ymax = ylim

    fig, ax = plt.subplots(figsize=figsize)

    baseline_idx = 0
    for name, values in algo_qoe.items():
        x, y = compute_hist_cdf(values, xmin, xmax, num_bins=num_bins)
        if x.size == 0:
            print(f"  警告: {algo_display_name(name)} 在 [{xmin}, {xmax}] 内无数据，跳过曲线")
            continue
        is_ours = name.lower() == "ours"
        if is_ours:
            ls, color = OURS_LINE_STYLE, OURS_COLOR
        else:
            ls = line_styles[baseline_idx % len(line_styles)]
            color = colors[baseline_idx % len(colors)]
            baseline_idx += 1
        ax.plot(
            x, y, linestyle=ls, linewidth=2.6, color=color,
            label=algo_display_name(name),
        )

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    legend = ax.legend(loc="upper left", fontsize=12, framealpha=0.0)
    if legend.get_frame() is not None:
        legend.get_frame().set_facecolor("none")

    if show_better_arrow:
        ax.text(
            0.72,
            0.17,
            "Better ==>",
            transform=ax.transAxes,
            fontsize=13,
            ha="left",
            va="bottom",
            rotation=-30,
        )

    if title:
        ax.set_title(title, fontsize=14)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"CDF 图已保存: {output_path}")

    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="绘制 ABR 测试 QoE CDF（默认：每条 trace 去首块后的平均 QoE）"
    )
    parser.add_argument(
        "--metric",
        choices=("trace_mean", "per_chunk"),
        default="trace_mean",
        help="trace_mean=每条轨迹去首块后平均 QoE 的 CDF（默认）；per_chunk=每块 QoE 的 CDF",
    )
    parser.add_argument(
        "--algo",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="算法显示名与日志根目录，可重复多次",
    )
    parser.add_argument(
        "--preset",
        choices=("netllm_baselines", "paper_seven"),
        default="netllm_baselines",
        help="内置路径模板；默认 netllm_baselines（七算法）；paper_seven 额外含 Fugu/BayesMPC",
    )
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--video", default="video1")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--fixed-order", action="store_true", default=True)
    parser.add_argument("--no-fixed-order", action="store_false", dest="fixed_order")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--test-rounds", type=int, default=1)
    parser.add_argument(
        "--ours-dir",
        default=None,
        help="ABRLLM 测试结果目录，图例标注为 Ours（可与 --preset 联用）",
    )
    parser.add_argument("--output", "-o", default="artifacts/figures/qoe_cdf.png")
    parser.add_argument("--stats-json", default=None, help="可选：输出各算法统计 JSON")
    parser.add_argument("--skip-first-chunk", action="store_true", default=True)
    parser.add_argument("--no-skip-first-chunk", action="store_false", dest="skip_first_chunk")
    parser.add_argument("--skip-last-chunk", action="store_true", default=False)
    parser.add_argument("--no-skip-last-chunk", action="store_false", dest="skip_last_chunk")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xmin", type=float, default=-2.0, help="横轴下限；范围外的 chunk QoE 不参与 CDF")
    parser.add_argument("--xmax", type=float, default=3.0, help="横轴上限；范围外的样本不参与 CDF")
    parser.add_argument("--ymax", type=float, default=1.03, help="纵轴上限（略大于 1.0 以免曲线贴顶被裁切）")
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS, help="直方图分箱数（越大曲线越平滑）")
    parser.add_argument("--show", action="store_true", help="保存后弹出窗口")
    parser.add_argument("--no-better-arrow", action="store_true")
    args = parser.parse_args(argv)

    algo_dirs: dict[str, str] = {}
    if args.preset:
        algo_dirs.update(
            build_preset_paths(
                args.preset,
                trace=args.trace,
                video=args.video,
                trace_num=args.trace_num,
                fixed_order=args.fixed_order,
                seed=args.seed,
                test_rounds=args.test_rounds,
                ours_dir=args.ours_dir,
            )
        )
    for spec in args.algo:
        name, path = _parse_algo_spec(spec)
        algo_dirs[name] = path

    if args.ours_dir and "ours" not in algo_dirs:
        algo_dirs["ours"] = args.ours_dir

    if args.preset != "paper_seven":
        algo_dirs = filter_algo_dirs(algo_dirs)

    if not algo_dirs:
        parser.error("请至少指定 --algo NAME=DIR 或 --preset")

    algo_qoe: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}
    missing: list[str] = []

    for name, log_root in algo_dirs.items():
        abs_root = os.path.abspath(log_root)
        if not os.path.isdir(abs_root):
            missing.append(f"{name}: 目录不存在 {abs_root}")
            continue
        try:
            if args.metric == "trace_mean":
                values = collect_trace_mean_qoe(
                    abs_root,
                    skip_first_chunk=args.skip_first_chunk,
                    skip_last_chunk=args.skip_last_chunk,
                )
            else:
                values = collect_chunk_qoe(
                    abs_root,
                    skip_first=args.skip_first_chunk,
                    skip_last=args.skip_last_chunk,
                )
        except (FileNotFoundError, ValueError) as e:
            missing.append(f"{name}: {e}")
            continue
        algo_qoe[name] = values
        in_range = truncate_qoe(values, args.xmin, args.xmax)
        stat_key = "num_traces" if args.metric == "trace_mean" else "num_chunks"
        stats[name] = {
            "log_root": abs_root,
            "metric": args.metric,
            stat_key: int(values.size),
            f"{stat_key}_in_xlim": int(in_range.size),
            f"{stat_key}_dropped": int(values.size - in_range.size),
            "xlim": [args.xmin, args.xmax],
            "mean_qoe": float(values.mean()),
            "mean_qoe_in_xlim": float(in_range.mean()) if in_range.size else None,
            "std_qoe": float(values.std()),
            "min_qoe": float(values.min()),
            "max_qoe": float(values.max()),
            "median_qoe": float(np.median(values)),
        }
        print(
            f"  {name}: n={values.size} traces, mean={values.mean():.4f}, "
            f"median={np.median(values):.4f} (in [{args.xmin},{args.xmax}]: {in_range.size}), "
            f"dir={abs_root}"
            if args.metric == "trace_mean"
            else
            f"  {name}: n={values.size} chunks, mean={values.mean():.4f}, "
            f"median={np.median(values):.4f} (in [{args.xmin},{args.xmax}]: {in_range.size}), "
            f"dir={abs_root}"
        )

    if missing:
        print("\n警告：以下算法未纳入绘图：")
        for m in missing:
            print(f"  - {m}")

    if not algo_qoe:
        print("错误：没有可用的 QoE 数据，请先运行 baseline 测试或检查 --algo 路径。")
        return 1

    algo_qoe = order_algo_dict(algo_qoe)

    ylabel = (
        "CDF (Perc. of sessions)"
        if args.metric == "trace_mean"
        else "CDF (Perc. of chunks)"
    )
    plot_qoe_cdf(
        algo_qoe,
        output_path=args.output,
        title=args.title,
        xlabel="Average Values of Chunk's QoE",
        ylabel=ylabel,
        xlim=(args.xmin, args.xmax),
        ylim=(0.0, args.ymax),
        num_bins=args.num_bins,
        show_better_arrow=not args.no_better_arrow,
    )

    if args.stats_json:
        out_json = os.path.abspath(args.stats_json)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"统计已写入: {out_json}")

    if args.show:
        plt.show()
    else:
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
