"""
对种子 seed ∈ [seed_start, seed_end]（默认 1–50）各跑一遍「全 trace × 全 video」的随机策略仿真。

每个种子的产物放在：
  adaptive_bitrate_streaming/wm_traces/random{seed}/
    random_{trace}_{video}.pkl   # 仅 pkl，不写 result_sim_abr_* 文本

某一 (seed, trace, video) 失败不会中断其它任务。

用法（在 adaptive_bitrate_streaming 目录下）::
 python generate_traces/run_record_random_exp_pool_seeds.py --seed-start 1 --seed-end 50
"""
import argparse
import os
import traceback

import _bootstrap  # noqa: F401

import torch

from config import cfg
from baseline_special.utils.utils import load_traces
from plm_special.record_random_policy_exp_pool import record_random_policy_exp_pool
from plm_special.utils.utils import set_random_seed
from run_test_record_exp_pool_core import PROJECT_ROOT, discover_trace_datasets


def seed_run_root(seed: int) -> str:
    """wm_traces/random1, random2, …"""
    return os.path.join(PROJECT_ROOT, "wm_traces", f"random{int(seed)}")


def main():
    p = argparse.ArgumentParser(description="Random policy: batch seeds 1..50 -> wm_traces/random{seed}/")
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--seed-end", type=int, default=50)
    p.add_argument("--trace-num", type=int, default=100)
    p.add_argument("--fixed-order", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--target-return", type=float, default=0.0)
    p.add_argument("--keep-first-transition", action="store_true")
    p.add_argument(
        "--same-policy-seed",
        action="store_true",
        help="若设置，则各 trace×video 共用 policy_seed=run_seed（随机动作序列起点相同；通常不建议）",
    )
    args_ns = p.parse_args()

    if args_ns.seed_start > args_ns.seed_end:
        raise SystemExit("seed-start 不能大于 seed-end")

    device = args_ns.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    traces = discover_trace_datasets(PROJECT_ROOT)
    if not traces:
        raise SystemExit("未发现 trace 子目录（data/traces/{train,valid,test}/）")

    videos = sorted(cfg.video_size_dirs.keys())
    total_runs = (args_ns.seed_end - args_ns.seed_start + 1) * len(traces) * len(videos)
    print(
        f"种子范围: {args_ns.seed_start}..{args_ns.seed_end}（共 {args_ns.seed_end - args_ns.seed_start + 1} 个）\n"
        f"每组目录: {seed_run_root(args_ns.seed_start)} … random{args_ns.seed_end}\n"
        f"总任务数: {total_runs}"
    )
    print("=" * 60)

    ok, fail = 0, 0
    for run_seed in range(args_ns.seed_start, args_ns.seed_end + 1):
        seed_dir = seed_run_root(run_seed)
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n>>> 种子 {run_seed} 输出根目录: {os.path.abspath(seed_dir)}")

        for ti, (trace_key, trace_dir) in enumerate(traces):
            for vi, video_key in enumerate(videos):
                label = f"seed={run_seed}  {trace_key} + {video_key}"
                safe_trace = trace_key.replace(os.sep, "_").replace("/", "_")
                exp_pool_path = os.path.join(seed_dir, f"random_{safe_trace}_{video_key}.pkl")

                try:
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
                        "video_size_dir": cfg.video_size_dirs[video_key],
                        "fixed": fixed_order,
                        "trace_num": trace_num,
                    }
                    pol_seed = run_seed if args_ns.same_policy_seed else run_seed + ti * 100 + vi

                    set_random_seed(run_seed)
                    results = record_random_policy_exp_pool(
                        device=device,
                        seed=run_seed,
                        env_settings=env_settings,
                        results_dir=seed_dir,
                        exp_pool_path=exp_pool_path,
                        max_ep_num=trace_num,
                        process_reward_fn=None,
                        target_return=args_ns.target_return,
                        skip_first_transition_per_episode=not args_ns.keep_first_transition,
                        policy_seed=pol_seed,
                        write_sim_logs=False,
                    )
                    ok += 1
                    print(
                        f"[OK] {label}\n"
                        f"     pkl: {results['exp_pool_path']}  (n={results['exp_pool_size']})"
                    )
                except Exception as e:
                    fail += 1
                    print(f"[FAIL] {label}: {e!r}")
                    traceback.print_exc()
                print("-" * 60)

    print(f"\n全部结束: 成功 {ok}, 失败 {fail}, 合计 {ok + fail}")


if __name__ == "__main__":
    main()
