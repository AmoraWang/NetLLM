"""
用均匀随机码率策略合成轨迹并保存为 ExperiencePool pkl（不加载 LLM）。

在 adaptive_bitrate_streaming 目录下执行：
  python generate_traces/run_record_random_exp_pool.py --trace fcc-test --video video1
"""
import argparse
import os

import _bootstrap  # noqa: F401

import torch

from config import cfg
from baseline_special.utils.utils import load_traces
from plm_special.record_random_policy_exp_pool import record_random_policy_exp_pool
from plm_special.utils.utils import set_random_seed


def main():
    p = argparse.ArgumentParser(description="Random uniform bitrate rollout -> exp_pool.pkl")
    p.add_argument("--trace", type=str, default="fcc-test", help="cfg.trace_dirs 中的 key")
    p.add_argument("--video", type=str, default="video1", help="cfg.video_size_dirs 中的 key")
    p.add_argument("--trace-num", type=int, default=100)
    p.add_argument("--fixed-order", action="store_true")
    p.add_argument("--seed", type=int, default=100003)
    p.add_argument("--policy-seed", type=int, default=None, help="随机策略专用种子；默认与 --seed 相同")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 pkl 路径；默认 wm_traces/model/random_{trace}_{video}.pkl",
    )
    p.add_argument("--target-return", type=float, default=0.0, help="仅用于统计中的 return 轨迹，策略不使用")
    p.add_argument("--keep-first-transition", action="store_true", help="pkl 中保留每集第一条（默认与训练池一致：去掉首条）")
    args_ns = p.parse_args()

    assert args_ns.trace in cfg.trace_dirs, f"Unknown trace key: {args_ns.trace}"
    assert args_ns.video in cfg.video_size_dirs, f"Unknown video key: {args_ns.video}"

    device = args_ns.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(args_ns.seed)

    trace_dir = cfg.trace_dirs[args_ns.trace]
    video_size_dir = cfg.video_size_dirs[args_ns.video]
    all_cooked_time, all_cooked_bw, all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)
    trace_num = min(args_ns.trace_num, len(all_file_names))
    if args_ns.trace_num == -1:
        trace_num = len(all_file_names)
    fixed_order = args_ns.fixed_order or (trace_num == len(all_file_names))

    env_settings = {
        "all_cooked_time": all_cooked_time,
        "all_cooked_bw": all_cooked_bw,
        "all_file_names": all_file_names,
        "all_mahimahi_ptrs": all_mahimahi_ptrs,
        "video_size_dir": video_size_dir,
        "fixed": fixed_order,
        "trace_num": trace_num,
    }

    out_dir = os.path.join(_bootstrap.PROJECT_ROOT, "wm_traces", "model")
    os.makedirs(out_dir, exist_ok=True)
    if args_ns.output is None:
        exp_pool_path = os.path.join(out_dir, f"random_{args_ns.trace}_{args_ns.video}.pkl")
    else:
        exp_pool_path = args_ns.output
        if not os.path.isabs(exp_pool_path):
            exp_pool_path = os.path.join(_bootstrap.PROJECT_ROOT, exp_pool_path)

    results_dir = os.path.join(
        cfg.results_dir,
        f"random_{args_ns.trace}_{args_ns.video}",
        f"trace_num_{trace_num}_fixed_{fixed_order}_seed_{args_ns.seed}",
    )
    os.makedirs(results_dir, exist_ok=True)

    pol_seed = args_ns.policy_seed if args_ns.policy_seed is not None else args_ns.seed

    log = record_random_policy_exp_pool(
        device=device,
        seed=args_ns.seed,
        env_settings=env_settings,
        results_dir=results_dir,
        exp_pool_path=exp_pool_path,
        max_ep_num=trace_num,
        process_reward_fn=None,
        target_return=args_ns.target_return,
        skip_first_transition_per_episode=not args_ns.keep_first_transition,
        policy_seed=pol_seed,
    )
    print(log)


if __name__ == "__main__":
    main()
