#!/usr/bin/env python3
"""
从 baseline / ABRLLM (Ours) 测试日志收集 per-trace mean QoE，按 (测试集, 算法) 写入 .pt。

每个文件命名 ``{数据集}_{算法}.pt``，内容为::

    TensorDict(
        {
            "avg_qoe": ...,
            "quality": ...,
            "rebuffer_penalty": ...,
            "smoothness_penalty": ...,
        },
        batch_size=[num_select],
    )

算法键名：baseline 为 mpc/bba/genet/...；Ours 为 ``ours`` → ``fcc-test_ours.pt``。

示例：
  cd adaptive_bitrate_streaming
  python run/collect_qoe_cdf_columns.py
  python run/collect_qoe_cdf_columns.py --check-only
  python run/collect_qoe_cdf_columns.py --output-dir artifacts/qoe_cdf_columns/video1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)

from run.plot_qoe_cdf import collect_trace_mean_qoe_components_by_key
from run.qoe_result_paths import (
    DEFAULT_ABRLLM_COLLECT,
    DEFAULT_COLLECT_ALGOS,
    DEFAULT_CDF_TRACES,
    OURS_PT_ALGO,
    algo_run_command_hint,
    build_algo_log_dir,
    normalize_trace_key,
    resolve_trace_num,
    trace_output_slug,
)
from run.qoe_tensordict_store import (
    TraceBatchQoe,
    make_qoe_key,
    make_result_pt_path,
    save_qoe_result_pt,
)


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def collect_batch_for_algo(
    log_root: str,
    *,
    skip_first_chunk: bool,
    skip_last_chunk: bool,
) -> TraceBatchQoe:
    by_key = collect_trace_mean_qoe_components_by_key(
        log_root,
        skip_first_chunk=skip_first_chunk,
        skip_last_chunk=skip_last_chunk,
    )
    avg_qoe: list[float] = []
    quality: list[float] = []
    rebuffer_penalty: list[float] = []
    smoothness_penalty: list[float] = []
    for _key in sorted(by_key.keys()):
        q, r_pen, s_pen, reward = by_key[_key]
        avg_qoe.append(reward)
        quality.append(q)
        rebuffer_penalty.append(r_pen)
        smoothness_penalty.append(s_pen)
    return TraceBatchQoe(
        avg_qoe=avg_qoe,
        quality=quality,
        rebuffer_penalty=rebuffer_penalty,
        smoothness_penalty=smoothness_penalty,
    )


def _resolve_collect_algos(raw: str, *, include_ours: bool) -> list[str]:
    algos = _parse_csv_list(raw)
    if include_ours:
        if OURS_PT_ALGO not in algos:
            algos = [*algos, OURS_PT_ALGO]
    else:
        algos = [a for a in algos if a.lower() != OURS_PT_ALGO]
    return algos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="收集各算法在各测试集上的 per-trace mean QoE → .pt (TensorDict)"
    )
    parser.add_argument(
        "--traces",
        default=",".join(trace_output_slug(t) for t in DEFAULT_CDF_TRACES),
        help="逗号分隔测试集（fcc-test / norway3G-test / hsr-test / oboe-test 等）",
    )
    parser.add_argument(
        "--algos",
        default=",".join(DEFAULT_COLLECT_ALGOS),
        help="逗号分隔算法：mpc,bba,genet,comyco,merina,DP,ours",
    )
    parser.add_argument(
        "--no-ours",
        action="store_true",
        help="不收集 Ours (ABRLLM) 的 .pt",
    )
    parser.add_argument("--video", default="video1")
    parser.add_argument(
        "--trace-num",
        type=int,
        default=-1,
        help="每条测试集 trace 数；-1=用 trace 目录内全部文件数",
    )
    parser.add_argument("--fixed-order", action="store_true", default=True)
    parser.add_argument("--no-fixed-order", action="store_false", dest="fixed_order")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--test-rounds", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="artifacts/qoe_cdf_columns/video1",
        help="输出目录；每个 (测试集, 算法) 一个 {trace}_{algo}.pt",
    )
    parser.add_argument("--skip-first-chunk", action="store_true", default=True)
    parser.add_argument("--no-skip-first-chunk", action="store_false", dest="skip_first_chunk")
    parser.add_argument("--skip-last-chunk", action="store_true", default=False)
    parser.add_argument("--no-skip-last-chunk", action="store_false", dest="skip_last_chunk")
    parser.add_argument(
        "--manifest",
        default=None,
        help="JSON 汇总路径（默认 output-dir/manifest.json）",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查日志目录，不写 .pt",
    )
    abr = DEFAULT_ABRLLM_COLLECT
    parser.add_argument("--plm-type", default=str(abr["plm_type"]))
    parser.add_argument("--plm-size", default=str(abr["plm_size"]))
    parser.add_argument("--abr-llm-version", default=str(abr["abr_llm_version"]))
    parser.add_argument("--rank", type=int, default=int(abr["rank"]))
    parser.add_argument("--w", type=int, default=int(abr["w"]))
    parser.add_argument("--gamma", type=float, default=float(abr["gamma"]))
    parser.add_argument(
        "--target-return-scale",
        type=float,
        default=float(abr["target_return_scale"]),
    )
    args = parser.parse_args(argv)

    trace_inputs = _parse_csv_list(args.traces)
    algos = _resolve_collect_algos(args.algos, include_ours=not args.no_ours)
    out_dir = os.path.abspath(args.output_dir)
    manifest_path = args.manifest
    if manifest_path is None and not args.check_only:
        manifest_path = os.path.join(out_dir, "manifest.json")

    manifest: dict = {
        "video": args.video,
        "output_dir": out_dir,
        "format": "pt_tensordict_qoe_components",
        "tensor_layout": (
            "TensorDict({avg_qoe, quality, rebuffer_penalty, smoothness_penalty}, "
            "batch_size=[num_select])"
        ),
        "ours_log_dir_template": (
            "artifacts/results/{trace_cfg_key}_{video}/trace_num_{N}_fixed_{fixed}/"
            "{plm_type}_{plm_size}/abrllm_{version}_rank_{rank}_w_{w}_gamma_{gamma}"
            "_tgt_scale_{tgt}_seed_{seed}/"
        ),
        "ours_collect_params": {
            "plm_type": args.plm_type,
            "plm_size": args.plm_size,
            "abr_llm_version": args.abr_llm_version,
            "rank": args.rank,
            "w": args.w,
            "gamma": args.gamma,
            "target_return_scale": args.target_return_scale,
            "seed": args.seed,
            "fixed_order": args.fixed_order,
        },
        "entries": [],
        "missing": [],
    }
    ok_count = 0
    missing_count = 0

    log_dir_kwargs = dict(
        video=args.video,
        fixed_order=args.fixed_order,
        seed=args.seed,
        test_rounds=args.test_rounds,
        plm_type=args.plm_type,
        plm_size=args.plm_size,
        abr_llm_version=args.abr_llm_version,
        rank=args.rank,
        w=args.w,
        gamma=args.gamma,
        target_return_scale=args.target_return_scale,
    )
    hint_kwargs = dict(
        video=args.video,
        seed=args.seed,
        test_rounds=args.test_rounds,
        plm_type=args.plm_type,
        plm_size=args.plm_size,
        abr_llm_version=args.abr_llm_version,
        rank=args.rank,
        w=args.w,
        gamma=args.gamma,
        target_return_scale=args.target_return_scale,
    )

    for trace_in in trace_inputs:
        trace_key = normalize_trace_key(trace_in)
        trace_slug = trace_output_slug(trace_key)
        trace_num = resolve_trace_num(trace_key, args.trace_num)
        print(f"\n[{trace_slug}] trace_num={trace_num} (cfg key: {trace_key})")

        for algo in algos:
            log_root = build_algo_log_dir(
                algo,
                trace=trace_key,
                trace_num=trace_num,
                **log_dir_kwargs,
            )
            abs_root = os.path.abspath(log_root)
            file_key = make_qoe_key(trace_slug, algo)
            pt_path = make_result_pt_path(out_dir, trace_slug, algo)

            if not os.path.isdir(abs_root):
                missing_count += 1
                hint = algo_run_command_hint(algo, trace_key, trace_num, **hint_kwargs)
                print(f"  MISSING {algo:8s}  file={file_key}.pt")
                print(f"           dir={abs_root}")
                print(f"           → {hint}")
                manifest["missing"].append(
                    {
                        "file": f"{file_key}.pt",
                        "trace": trace_slug,
                        "trace_cfg_key": trace_key,
                        "algo": algo,
                        "log_root": abs_root,
                        "run_hint": hint,
                    }
                )
                continue

            try:
                batch = collect_batch_for_algo(
                    abs_root,
                    skip_first_chunk=args.skip_first_chunk,
                    skip_last_chunk=args.skip_last_chunk,
                )
            except (FileNotFoundError, ValueError) as e:
                missing_count += 1
                hint = algo_run_command_hint(algo, trace_key, trace_num, **hint_kwargs)
                print(f"  EMPTY  {algo:8s}  file={file_key}.pt  {e}")
                print(f"           → {hint}")
                manifest["missing"].append(
                    {
                        "file": f"{file_key}.pt",
                        "trace": trace_slug,
                        "algo": algo,
                        "log_root": abs_root,
                        "error": str(e),
                        "run_hint": hint,
                    }
                )
                continue

            num_select = batch.num_select
            mean_val = float(sum(batch.avg_qoe) / num_select) if num_select else None

            if not args.check_only:
                qoe_td = save_qoe_result_pt(pt_path, batch)
                assert int(qoe_td.batch_size[0]) == num_select

            manifest["entries"].append(
                {
                    "file": f"{file_key}.pt",
                    "trace": trace_slug,
                    "trace_cfg_key": trace_key,
                    "algo": algo,
                    "log_root": abs_root,
                    "output_path": pt_path,
                    "num_select": num_select,
                    "mean_of_means": mean_val,
                    "mean_quality": float(sum(batch.quality) / num_select),
                    "mean_rebuffer_penalty": float(sum(batch.rebuffer_penalty) / num_select),
                    "mean_smoothness_penalty": float(sum(batch.smoothness_penalty) / num_select),
                }
            )
            ok_count += 1
            print(
                f"  OK     {algo:8s}  file={file_key}.pt  "
                f"num_select={num_select}  avg={mean_val:.4f}"
            )

    print(f"\n汇总: 成功 {ok_count}，缺失/空 {missing_count}，合计 {ok_count + missing_count}")

    if manifest_path and not args.check_only:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)) or ".", exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"manifest: {manifest_path}")
        print(f"输出目录: {out_dir}")

    return 0 if missing_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
