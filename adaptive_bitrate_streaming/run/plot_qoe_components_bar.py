#!/usr/bin/env python3
"""
绘制各 ABR 方法的 QoE 及其分量分组柱状图（风格参考论文 QoE 分解图）。

四组指标（x 轴分组标签含 ↑/↓）：
  - QoE ↑：每条 trace 去首块后的 chunk 均值，再对 trace 求平均
  - Bitrate ↑：平均码率（Mbps），对应 .pt 中 quality
  - Rebuffering ↓：平均重缓冲惩罚项 REBUF_PENALTY * rebuf
  - Smoothness ↓：平均平滑度惩罚项 SMOOTH_PENALTY * |Δbitrate|

数据来源：
  - ``--qoe-pt-dir``：``collect_qoe_cdf_columns.py`` 输出的 ``{trace}_{algo}.pt``
  - 默认（无 ``--qoe-pt-dir``）：递归扫描测试结果日志目录

纵轴：Average value

示例：
  cd adaptive_bitrate_streaming
  bash bash/run_plot_components_bar.sh
  python run/plot_qoe_components_bar.py \\
    --qoe-pt-dir artifacts/qoe_cdf_columns/video1 --trace fcc-test --video video1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)

import matplotlib.pyplot as plt
import numpy as np

from baseline_special.utils.constants import M_IN_K, REBUF_PENALTY, SMOOTH_PENALTY
from run.plot_algo_defaults import (
    DEFAULT_PLOT_ALGOS,
    algo_display_name,
    filter_algo_dirs,
    filter_plot_algos,
    is_plot_excluded_algo,
)
from run.plot_qoe_cdf import (
    LOG_NAME_HINTS,
    _parse_algo_spec,
    build_preset_paths,
    find_log_files,
)

DEFAULT_BAR_ALGOS = DEFAULT_PLOT_ALGOS

# 颜色 + hatch
BAR_STYLES: dict[str, dict] = {
    "BOLA": {"color": "#aec7e8", "hatch": "//", "edgecolor": "#4a6fa5"},
    "RobustMPC": {"color": "#5ab4c5", "hatch": "--", "edgecolor": "#2d6a78"},
    "Pensieve": {"color": "#3d4f5f", "hatch": "xx", "edgecolor": "#1a252f"},
    "MERINA": {"color": "#e377c2", "hatch": "|||", "edgecolor": "#9467bd"},
    "Comyco": {"color": "#c49c94", "hatch": "++", "edgecolor": "#8c564b"},
    "Oracle": {"color": "#bdbdbd", "hatch": "xx", "edgecolor": "#636363"},
    "ours": {"color": "#c44e52", "hatch": "", "edgecolor": "#8b2e32"},
}

METRIC_GROUPS = (
    ("QoE ↑", "qoe"),
    ("Bitrate ↑", "bitrate_mbps"),
    ("Rebuffering ↓", "rebuf_penalty"),
    ("Smoothness ↓", "smooth_penalty"),
)


@dataclass
class ChunkMetrics:
    qoe: float
    bitrate_mbps: float
    rebuf_penalty: float
    smooth_penalty: float


def _parse_log_line(parts: list[str]) -> tuple[float, float, float, float | None, float] | None:
    """
    返回 (bitrate_kbps, rebuf_sec, smooth_diff_mbps_or_none, reward)。
    8 列：含 smoothness；7 列：无 smoothness 列。
    """
    if len(parts) < 7:
        return None
    try:
        bitrate_kbps = float(parts[1])
        rebuf_sec = float(parts[3])
        if len(parts) >= 8:
            smooth_diff = float(parts[6])
            reward = float(parts[7])
            return bitrate_kbps, rebuf_sec, smooth_diff, reward
        reward = float(parts[-1])
        return bitrate_kbps, rebuf_sec, None, reward
    except ValueError:
        return None


def read_chunk_metrics_from_file(
    path: str,
    *,
    skip_first: bool = True,
    skip_last: bool = False,
) -> list[ChunkMetrics]:
    chunks: list[ChunkMetrics] = []
    last_bitrate_kbps: float | None = None

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            parsed = _parse_log_line(parts)
            if parsed is None:
                continue
            bitrate_kbps, rebuf_sec, smooth_diff, reward = parsed

            if smooth_diff is None:
                if last_bitrate_kbps is None:
                    smooth_diff = 0.0
                else:
                    smooth_diff = abs(bitrate_kbps - last_bitrate_kbps) / M_IN_K
            last_bitrate_kbps = bitrate_kbps

            rebuf_pen = REBUF_PENALTY * rebuf_sec
            smooth_pen = SMOOTH_PENALTY * smooth_diff
            chunks.append(
                ChunkMetrics(
                    qoe=reward,
                    bitrate_mbps=bitrate_kbps / M_IN_K,
                    rebuf_penalty=rebuf_pen,
                    smooth_penalty=smooth_pen,
                )
            )

    if skip_first and chunks:
        chunks = chunks[1:]
    if skip_last and chunks:
        chunks = chunks[:-1]
    return chunks


def collect_algo_component_means(
    log_root: str,
    *,
    skip_first: bool = True,
    skip_last: bool = False,
) -> dict[str, float]:
    files = find_log_files(log_root)
    if not files:
        raise FileNotFoundError(
            f"在 {log_root!r} 下未找到测试结果日志（需含 {LOG_NAME_HINTS} 的文件名）"
        )

    qoes: list[float] = []
    bitrates: list[float] = []
    rebufs: list[float] = []
    smooths: list[float] = []

    for fp in files:
        for c in read_chunk_metrics_from_file(fp, skip_first=skip_first, skip_last=skip_last):
            qoes.append(c.qoe)
            bitrates.append(c.bitrate_mbps)
            rebufs.append(c.rebuf_penalty)
            smooths.append(c.smooth_penalty)

    if not qoes:
        raise ValueError(f"{log_root!r} 中无法解析 QoE 分量")

    return {
        "qoe": float(np.mean(qoes)),
        "bitrate_mbps": float(np.mean(bitrates)),
        "rebuf_penalty": float(np.mean(rebufs)),
        "smooth_penalty": float(np.mean(smooths)),
        "num_chunks": len(qoes),
    }


def collect_algo_component_means_from_pt(path: str) -> dict[str, float]:
    """从 ``{trace}_{algo}.pt`` 读取四组柱状图指标（trace 维均值）。"""
    from run.qoe_tensordict_store import load_component_means_from_pt

    means = load_component_means_from_pt(path)
    return {
        "qoe": means["qoe"],
        "bitrate_mbps": means["bitrate_mbps"],
        "rebuf_penalty": means["rebuf_penalty"],
        "smooth_penalty": means["smooth_penalty"],
        "num_traces": means["num_traces"],
    }


def _print_algo_means(name: str, means: dict[str, float]) -> None:
    count_key = "num_traces" if "num_traces" in means else "num_chunks"
    count = means[count_key]
    print(
        f"  {name}: {count_key}={count}, "
        f"QoE={means['qoe']:.4f}, bitrate={means['bitrate_mbps']:.4f} Mbps, "
        f"rebuf_pen={means['rebuf_penalty']:.4f}, smooth_pen={means['smooth_penalty']:.4f}"
    )


def plot_qoe_components_bar(
    algo_metrics: dict[str, dict[str, float]],
    *,
    algo_order: tuple[str, ...] = DEFAULT_BAR_ALGOS,
    output_path: str | None = None,
    title: str | None = None,
    ylabel: str = "Average value",
    figsize: tuple[float, float] = (11, 5),
    dpi: int = 150,
) -> plt.Figure:
    present = [a for a in algo_order if a in algo_metrics]
    if not present:
        raise ValueError("没有可绘制的算法数据")

    n_groups = len(METRIC_GROUPS)
    n_algos = len(present)
    group_x = np.arange(n_groups, dtype=float)
    bar_width = min(0.12, 0.85 / max(n_algos, 1))
    offsets = (np.arange(n_algos) - (n_algos - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=figsize)

    for i, algo in enumerate(present):
        vals = [algo_metrics[algo][key] for _, key in METRIC_GROUPS]
        style = BAR_STYLES.get(algo, {"color": "#cccccc", "hatch": "", "edgecolor": "#333333"})
        label = algo_display_name(algo)
        ax.bar(
            group_x + offsets[i],
            vals,
            width=bar_width * 0.92,
            label=label,
            color=style["color"],
            hatch=style.get("hatch", ""),
            edgecolor=style.get("edgecolor", "#333333"),
            linewidth=0.8,
            zorder=3,
        )

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(group_x)
    ax.set_xticklabels([g[0] for g in METRIC_GROUPS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)

    ymax = max(algo_metrics[a][k] for a in present for _, k in METRIC_GROUPS)
    ax.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=min(n_algos, 7),
        fontsize=12,
        framealpha=0.0,
    )
    if legend.get_frame() is not None:
        legend.get_frame().set_facecolor("none")

    if title:
        ax.set_title(title, fontsize=14)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"QoE 分量柱状图已保存: {output_path}")

    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="绘制 QoE 及分量分组柱状图")
    parser.add_argument("--algo", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument(
        "--preset",
        choices=("netllm_baselines", "paper_seven"),
        default="netllm_baselines",
    )
    parser.add_argument(
        "--bar-algos",
        default=",".join(DEFAULT_BAR_ALGOS),
        help="逗号分隔的算法名及顺序（默认七算法含 Ours）",
    )
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--video", default="video1")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--fixed-order", action="store_true", default=True)
    parser.add_argument("--no-fixed-order", action="store_false", dest="fixed_order")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--test-rounds", type=int, default=1)
    parser.add_argument("--ours-dir", default=None)
    parser.add_argument(
        "--qoe-pt-dir",
        "--qoe-tensordict",
        dest="qoe_pt_dir",
        default=None,
        help="collect_qoe_cdf_columns.py 输出目录（读取 {trace}_{algo}.pt）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="artifacts/figures/qoe_components_bar.png",
    )
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--skip-first-chunk", action="store_true", default=True)
    parser.add_argument("--no-skip-first-chunk", action="store_false", dest="skip_first_chunk")
    parser.add_argument("--skip-last-chunk", action="store_true", default=False)
    parser.add_argument("--title", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    bar_order = tuple(s.strip() for s in args.bar_algos.split(",") if s.strip())

    algo_metrics: dict[str, dict[str, float]] = {}
    stats: dict[str, dict] = {}
    missing: list[str] = []

    if args.qoe_pt_dir:
        from run.qoe_result_paths import (
            OURS_PT_ALGO,
            normalize_trace_key,
            plot_key_to_pt_algo,
            trace_output_slug,
        )
        from run.qoe_tensordict_store import make_result_pt_path

        pt_dir = os.path.abspath(args.qoe_pt_dir)
        trace_slug = trace_output_slug(normalize_trace_key(args.trace))
        plot_targets = bar_order or tuple(DEFAULT_BAR_ALGOS)

        for plot_key in plot_targets:
            if is_plot_excluded_algo(plot_key):
                continue
            pt_algo = plot_key_to_pt_algo(plot_key)
            if pt_algo is None:
                missing.append(f"{plot_key}: 无对应 .pt 算法名")
                continue
            pt_path = make_result_pt_path(pt_dir, trace_slug, pt_algo)
            if not os.path.isfile(pt_path):
                if plot_key.lower() == OURS_PT_ALGO and args.ours_dir:
                    continue
                missing.append(f"{plot_key}: 缺少 {pt_path}")
                continue
            try:
                means = collect_algo_component_means_from_pt(pt_path)
                store_key = "ours" if plot_key.lower() == OURS_PT_ALGO else plot_key
                algo_metrics[store_key] = means
                stats[store_key] = {
                    "source": "pt_tensordict",
                    "file": pt_path,
                    "trace_slug": trace_slug,
                    "pt_algo": pt_algo,
                    **means,
                }
                _print_algo_means(store_key, means)
            except (FileNotFoundError, ValueError, TypeError, KeyError) as e:
                missing.append(f"{plot_key}: {e}")

        if "ours" not in algo_metrics and args.ours_dir:
            abs_root = os.path.abspath(args.ours_dir)
            if os.path.isdir(abs_root):
                try:
                    means = collect_algo_component_means(
                        abs_root,
                        skip_first=args.skip_first_chunk,
                        skip_last=args.skip_last_chunk,
                    )
                    algo_metrics["ours"] = means
                    stats["ours"] = {"source": "log_dir", "log_root": abs_root, **means}
                    _print_algo_means("ours", means)
                except (FileNotFoundError, ValueError) as e:
                    missing.append(f"ours: {e}")
            else:
                missing.append(f"ours: 目录不存在 {abs_root}")
    else:
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

        for name, log_root in algo_dirs.items():
            if is_plot_excluded_algo(name):
                continue
            if name not in bar_order and not args.algo:
                continue
            abs_root = os.path.abspath(log_root)
            if not os.path.isdir(abs_root):
                missing.append(f"{name}: 目录不存在 {abs_root}")
                continue
            try:
                means = collect_algo_component_means(
                    abs_root,
                    skip_first=args.skip_first_chunk,
                    skip_last=args.skip_last_chunk,
                )
                algo_metrics[name] = means
                stats[name] = {"source": "log_dir", "log_root": abs_root, **means}
                _print_algo_means(name, means)
            except (FileNotFoundError, ValueError) as e:
                missing.append(f"{name}: {e}")

    if missing:
        print("\n警告：")
        for m in missing:
            print(f"  - {m}")

    algo_metrics = filter_plot_algos(algo_metrics)
    stats = filter_plot_algos(stats)

    plot_order = tuple(a for a in bar_order if a in algo_metrics)
    if not plot_order:
        plot_order = tuple(a for a in DEFAULT_BAR_ALGOS if a in algo_metrics)

    if not plot_order:
        print("错误：没有可用的分量数据。")
        return 1

    out_path = args.output
    if args.output == parser.get_default("output"):
        out_path = f"artifacts/figures/qoe_components_bar_{args.trace}_{args.video}.png"

    plot_qoe_components_bar(
        algo_metrics,
        algo_order=plot_order,
        output_path=out_path,
        title=args.title,
    )

    if args.stats_json:
        out_json = os.path.abspath(args.stats_json)
    elif args.output == parser.get_default("output"):
        out_json = out_path.replace(".png", ".json")
    else:
        out_json = None

    if out_json:
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
