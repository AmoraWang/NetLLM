"""
独立测试入口：加载与 run_abr 相同的模型与配置，调用 test_on_env_with_exp_pool，
将 transition 写入 ExperiencePool 并保存为 pkl。不修改 run_abr.py。

在仓库根目录 adaptive_bitrate_streaming 下执行：
  python generate_traces/run_test_record_exp_pool.py [参数]

默认输出的测试经验池 pkl 路径：wm_traces/model/test_record_exp_pool.pkl（可用 --exp-pool-output 覆盖）。

批量测试（所有 trace × 所有 video）：请使用 generate_traces/run_test_record_exp_pool_batch.py
"""
import os

import _bootstrap  # noqa: F401 — cwd = 项目根

import torch
from pprint import pprint

from config import cfg

from run_test_record_exp_pool_core import (
    DEFAULT_RECORD_EXP_POOL_DIR,
    DEFAULT_RECORD_EXP_POOL_NAME,
    PROJECT_ROOT,
    build_model_and_load,
    build_parser,
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
    assert args.trace in cfg.trace_dirs.keys()
    assert args.video in cfg.video_size_dirs.keys()

    exp_dataset_info = load_exp_dataset_info(args)
    print("Experience dataset info:")
    pprint(exp_dataset_info)

    abrllm_model, model_dir = build_model_and_load(args)

    trace_dir = cfg.trace_dirs[args.trace]
    video_size_dir = cfg.video_size_dirs[args.video]

    exp_pool_out = args.exp_pool_output
    if exp_pool_out is None:
        os.makedirs(DEFAULT_RECORD_EXP_POOL_DIR, exist_ok=True)
        exp_pool_out = os.path.join(DEFAULT_RECORD_EXP_POOL_DIR, DEFAULT_RECORD_EXP_POOL_NAME)
    elif not os.path.isabs(exp_pool_out):
        exp_pool_out = os.path.join(PROJECT_ROOT, exp_pool_out)

    torch.backends.cudnn.benchmark = True

    print("Arguments:")
    pprint(args)
    print("Using model_dir:", model_dir)
    print("Experience pool output:", os.path.abspath(exp_pool_out))

    results = run_single_record_test(
        args,
        exp_dataset_info,
        abrllm_model,
        args.trace,
        trace_dir,
        args.video,
        video_size_dir,
        exp_pool_out,
    )

    print(results)
    print("Test time:", results["time"])
    print("QoE Metrics:")
    print("  Mean QoE (per chunk):", results["mean_qoe"])
    print("  Total QoE (all episodes):", results["total_qoe"])
    print("  Episodes count:", results["episodes_count"])
    print("  Total chunks:", results["total_chunks"])
    print("Recorded ExperiencePool:", results.get("exp_pool_path"))


if __name__ == "__main__":
    main()
