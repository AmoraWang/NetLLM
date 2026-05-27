#!/usr/bin/env python3
"""
绘制 Ours 相对各 baseline 的平均 QoE 改进 CDF。

每条 trace：改进量 = ours_trace_mean_qoe - baseline_trace_mean_qoe（同一条轨迹对齐）。
横轴：Avg. QoE Improvement，默认范围 [-2, 2]；在 x=0 处绘制竖直黑线。

示例：
  cd adaptive_bitrate_streaming
  bash bash/run_plot_improvement_cdf.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)

import matplotlib.pyplot as plt
import numpy as np

from run.plot_algo_defaults import (
    DEFAULT_BASELINE_ALGOS,
    algo_display_name,
    filter_algo_dirs,
)
from run.plot_qoe_cdf import (
    DEFAULT_COLORS,
    DEFAULT_LINE_STYLES,
    DEFAULT_NUM_BINS,
    _parse_algo_spec,
    build_preset_paths,
    collect_trace_mean_qoe_by_key,
    compute_hist_cdf,
    truncate_qoe,
)

SKIP_BASELINES = frozenset({"ours"})
BASELINE_PLOT_ORDER = DEFAULT_BASELINE_ALGOS


def paired_qoe_improvements(
    ours_by_key: dict[str, float],
    baseline_by_key: dict[str, float],
    *,
    xmin: float,
    xmax: float,
) -> tuple[np.ndarray, int, int]:
    """
    返回 (改进量数组, 配对 trace 数, 落入 xlim 的样本数)。
    仅使用两算法均存在的 trace。
    """
    common = sorted(set(ours_by_key) & set(baseline_by_key))
    if not common:
        return np.array([]), 0, 0
    diffs = np.asarray(
        [ours_by_key[k] - baseline_by_key[k] for k in common],
        dtype=np.float64,
    )
    in_range = truncate_qoe(diffs, xmin, xmax)
    return in_range, len(common), int(in_range.size)


def plot_qoe_improvement_cdf(
    baseline_improvements: dict[str, np.ndarray],
    *,
    output_path: str | None = None,
    title: str | None = None,
    xlabel: str = "Avg. QoE Improvement",
    ylabel: str = "CDF",
    xlim: tuple[float, float] = (-2.0, 2.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    num_bins: int = DEFAULT_NUM_BINS,
    line_styles: list[str] | None = None,
    colors: list[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> plt.Figure:
    line_styles = line_styles or DEFAULT_LINE_STYLES
    colors = colors or DEFAULT_COLORS
    xmin, xmax = xlim

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color="black", linewidth=1.8, linestyle="-", zorder=1)

    plot_idx = 0
    for name in BASELINE_PLOT_ORDER:
        if name not in baseline_improvements:
            continue
        values = baseline_improvements[name]
        x, y = compute_hist_cdf(values, xmin, xmax, num_bins=num_bins)
        if x.size == 0:
            print(f"  警告: Ours vs {algo_display_name(name)} 在 [{xmin}, {xmax}] 内无数据，跳过曲线")
            continue
        ls = line_styles[plot_idx % len(line_styles)]
        color = colors[plot_idx % len(colors)]
        ax.plot(
            x, y, linestyle=ls, linewidth=2.6, color=color,
            label=algo_display_name(name), zorder=2,
        )
        plot_idx += 1

    for name, values in baseline_improvements.items():
        if name in BASELINE_PLOT_ORDER:
            continue
        x, y = compute_hist_cdf(values, xmin, xmax, num_bins=num_bins)
        if x.size == 0:
            continue
        ls = line_styles[plot_idx % len(line_styles)]
        color = colors[plot_idx % len(colors)]
        ax.plot(
            x, y, linestyle=ls, linewidth=2.6, color=color,
            label=algo_display_name(name), zorder=2,
        )
        plot_idx += 1

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    legend = ax.legend(loc="upper left", fontsize=12, framealpha=0.0)
    if legend.get_frame() is not None:
        legend.get_frame().set_facecolor("none")

    if title:
        ax.set_title(title, fontsize=14)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"QoE 改进 CDF 图已保存: {output_path}")

    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="绘制 Ours 相对各 baseline 的 QoE 改进 CDF（按 trace 配对做差）"
    )
    parser.add_argument(
        "--algo",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="baseline 显示名与日志根目录，可重复；不含 Ours",
    )
    parser.add_argument(
        "--preset",
        choices=("netllm_baselines", "paper_seven"),
        default="netllm_baselines",
        help="内置路径模板（默认 netllm_baselines，七算法）",
    )
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--video", default="video1")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--fixed-order", action="store_true", default=True)
    parser.add_argument("--no-fixed-order", action="store_false", dest="fixed_order")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--test-rounds", type=int, default=1)
    parser.add_argument("--ours-dir", required=False, default=None, help="Ours 测试结果目录")
    parser.add_argument(
        "--output",
        "-o",
        default="artifacts/figures/qoe_improvement_cdf.png",
    )
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--skip-first-chunk", action="store_true", default=True)
    parser.add_argument("--no-skip-first-chunk", action="store_false", dest="skip_first_chunk")
    parser.add_argument("--skip-last-chunk", action="store_true", default=False)
    parser.add_argument("--no-skip-last-chunk", action="store_false", dest="skip_last_chunk")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xmin", type=float, default=-2.0)
    parser.add_argument("--xmax", type=float, default=2.0)
    parser.add_argument("--ymax", type=float, default=1.0)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--show", action="store_true")
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

    ours_dir = algo_dirs.pop("ours", None)
    if not ours_dir or not os.path.isdir(os.path.abspath(ours_dir)):
        parser.error("请通过 --ours-dir 或 --preset + 有效 ours 路径指定 Ours 测试结果目录")

    ours_abs = os.path.abspath(ours_dir)
    try:
        ours_by_key = collect_trace_mean_qoe_by_key(
            ours_abs,
            skip_first_chunk=args.skip_first_chunk,
            skip_last_chunk=args.skip_last_chunk,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"错误：无法读取 Ours 日志 {ours_abs}: {e}")
        return 1

    print(f"Ours: {len(ours_by_key)} traces, dir={ours_abs}")

    baseline_improvements: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {"ours": {"log_root": ours_abs, "num_traces": len(ours_by_key)}}
    missing: list[str] = []

    for name, log_root in algo_dirs.items():
        if name.lower() in SKIP_BASELINES:
            continue
        abs_root = os.path.abspath(log_root)
        if not os.path.isdir(abs_root):
            missing.append(f"{name}: 目录不存在 {abs_root}")
            continue
        try:
            base_by_key = collect_trace_mean_qoe_by_key(
                abs_root,
                skip_first_chunk=args.skip_first_chunk,
                skip_last_chunk=args.skip_last_chunk,
            )
            diffs, n_paired, n_in = paired_qoe_improvements(
                ours_by_key,
                base_by_key,
                xmin=args.xmin,
                xmax=args.xmax,
            )
            if diffs.size == 0:
                missing.append(f"{name}: 无配对 trace 或改进量均落在 [{args.xmin},{args.xmax}] 外")
                continue
            baseline_improvements[name] = diffs
            stats[name] = {
                "log_root": abs_root,
                "num_traces_paired": n_paired,
                "num_in_xlim": n_in,
                "xlim": [args.xmin, args.xmax],
                "mean_improvement": float(diffs.mean()),
                "median_improvement": float(np.median(diffs)),
                "fraction_positive": float((diffs > 0).mean()),
            }
            print(
                f"  Ours vs {algo_display_name(name)}: paired={n_paired}, in_xlim={n_in}, "
                f"mean_improve={diffs.mean():.4f}, frac(>0)={(diffs > 0).mean():.3f}, "
                f"dir={abs_root}"
            )
        except (FileNotFoundError, ValueError) as e:
            missing.append(f"{name}: {e}")

    if missing:
        print("\n警告：以下 baseline 未纳入绘图：")
        for m in missing:
            print(f"  - {m}")

    if not baseline_improvements:
        print("错误：没有可用的改进量数据。")
        return 1

    out_path = args.output
    if args.output == parser.get_default("output"):
        out_path = f"artifacts/figures/qoe_improvement_cdf_{args.trace}_{args.video}.png"

    plot_qoe_improvement_cdf(
        baseline_improvements,
        output_path=out_path,
        title=args.title,
        xlim=(args.xmin, args.xmax),
        ylim=(0.0, args.ymax),
        num_bins=args.num_bins,
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
