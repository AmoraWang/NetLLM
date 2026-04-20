"""
对 data/traces/{train,valid,test} 下每个 trace 与 video1–3，用均匀随机策略各生成一个 ExperiencePool pkl。
输出：wm_traces/model/random_{trace}_{video}.pkl；不写 result_sim_abr_* 文本。
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
from run_test_record_exp_pool_core import DEFAULT_RECORD_EXP_POOL_DIR, PROJECT_ROOT, discover_trace_datasets


def main():
    p = argparse.ArgumentParser(description="Batch random-policy exp pool generation")
    p.add_argument("--trace-num", type=int, default=100)
    p.add_argument("--fixed-order", action="store_true")
    p.add_argument("--seed", type=int, default=100003)
    p.add_argument("--policy-seed", type=int, default=None, help="每组可复用同一策略种子；默认每组用 trace+video 派生")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--target-return", type=float, default=0.0)
    p.add_argument("--keep-first-transition", action="store_true")
    args_ns = p.parse_args()

    device = args_ns.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    traces = discover_trace_datasets(PROJECT_ROOT)
    if not traces:
        raise SystemExit("未发现 trace 子目录")

    videos = sorted(cfg.video_size_dirs.keys())
    os.makedirs(DEFAULT_RECORD_EXP_POOL_DIR, exist_ok=True)

    ok, fail = 0, 0
    for ti, (trace_key, trace_dir) in enumerate(traces):
        for vi, video_key in enumerate(videos):
            label = f"{trace_key} + {video_key}"
            exp_pool_path = os.path.join(
                DEFAULT_RECORD_EXP_POOL_DIR,
                f"random_{trace_key.replace(os.sep, '_')}_{video_key}.pkl",
            )
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
                pol_seed = args_ns.policy_seed
                if pol_seed is None:
                    pol_seed = args_ns.seed + ti * 100 + vi

                set_random_seed(args_ns.seed)
                results = record_random_policy_exp_pool(
                    device=device,
                    seed=args_ns.seed,
                    env_settings=env_settings,
                    results_dir=DEFAULT_RECORD_EXP_POOL_DIR,
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
                    f"     已保存: {results['exp_pool_path']}  (n={results['exp_pool_size']})"
                )
            except Exception as e:
                fail += 1
                print(f"[FAIL] {label}: {e!r}")
                traceback.print_exc()
            print("-" * 60)
    print(f"完成: 成功 {ok}, 失败 {fail}")


if __name__ == "__main__":
    main()
