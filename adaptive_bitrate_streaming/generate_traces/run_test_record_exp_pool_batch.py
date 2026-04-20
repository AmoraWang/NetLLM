"""
在所有 trace 数据集与所有 video 上依次测试，每次完成后立即保存 pkl（wm_traces/model/{trace}_{video}.pkl），
并在控制台打印保存路径与规模。某一组失败不会中断整体（打印错误后继续）。

扫描目录：adaptive_bitrate_streaming/data/traces/{train,valid,test}/<trace_name>/
视频：config 中的 video1, video2, video3（对应 video*_sizes）。

用法（在 adaptive_bitrate_streaming 目录下）::
    python generate_traces/run_test_record_exp_pool_batch.py [与单测相同的超参/模型参数]
"""
import os
import traceback

import _bootstrap  # noqa: F401

import torch
from pprint import pprint

from config import cfg

from run_test_record_exp_pool_core import (
    DEFAULT_RECORD_EXP_POOL_DIR,
    PROJECT_ROOT,
    build_model_and_load,
    build_parser,
    discover_trace_datasets,
    load_exp_dataset_info,
    normalize_args,
    run_single_record_test,
)


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_args(args)

    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes

    traces = discover_trace_datasets(PROJECT_ROOT)
    if not traces:
        raise SystemExit("未发现任何 trace 数据集，请检查 data/traces/{train,valid,test}/ 下子目录。")

    videos = sorted(cfg.video_size_dirs.keys())
    exp_dataset_info = load_exp_dataset_info(args)
    print("Experience dataset info:")
    pprint(exp_dataset_info)

    abrllm_model, model_dir = build_model_and_load(args)
    torch.backends.cudnn.benchmark = True

    os.makedirs(DEFAULT_RECORD_EXP_POOL_DIR, exist_ok=True)

    print("Arguments:")
    pprint(args)
    print("Model dir:", model_dir)
    print(f"批量测试: {len(traces)} 个 trace × {len(videos)} 个 video = {len(traces) * len(videos)} 组")
    print("输出目录:", os.path.abspath(DEFAULT_RECORD_EXP_POOL_DIR))
    print("-" * 60)

    ok, fail = 0, 0
    for trace_key, trace_dir in traces:
        for video_key in videos:
            video_size_dir = cfg.video_size_dirs[video_key]
            safe_trace = trace_key.replace(os.sep, "_").replace("/", "_")
            safe_video = video_key.replace(os.sep, "_")
            exp_pool_out = os.path.join(
                DEFAULT_RECORD_EXP_POOL_DIR,
                f"{safe_trace}_{safe_video}.pkl",
            )
            label = f"{trace_key} + {video_key}"
            try:
                results = run_single_record_test(
                    args,
                    exp_dataset_info,
                    abrllm_model,
                    trace_key,
                    trace_dir,
                    video_key,
                    video_size_dir,
                    exp_pool_out,
                )
                n = results.get("exp_pool_size", "?")
                ok += 1
                print(
                    f"[OK] {label}\n"
                    f"     已保存: {os.path.abspath(exp_pool_out)}  (transitions={n}, mean_qoe={results.get('mean_qoe')})"
                )
            except Exception as e:
                fail += 1
                print(f"[FAIL] {label}")
                print(f"       错误: {e!r}")
                traceback.print_exc()
            print("-" * 60)

    print(f"批量结束: 成功 {ok}, 失败 {fail}, 合计 {ok + fail}")


if __name__ == "__main__":
    main()
