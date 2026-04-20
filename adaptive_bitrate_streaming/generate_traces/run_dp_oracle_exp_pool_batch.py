"""
对 data/traces/{train,valid,test} 下每个 trace 数据集目录、以及每个 video（video1–3），
调用 DP.py 中的动态规划 Oracle 生成经验池并保存为单个 .pkl。

每个 pkl 包含该目录下**所有 trace 文件**依次求解 DP 后拼接的 ExperiencePool（与 DP.generate_oracle_exp_pool 行为一致）。

输出目录：wm_traces/DP/{trace_key}_{video}.pkl

用法（在 adaptive_bitrate_streaming 根目录下）::

    python generate_traces/run_dp_oracle_exp_pool_batch.py
    python generate_traces/run_dp_oracle_exp_pool_batch.py --seed 42 --trace-limit 10
"""
import argparse
import os
import traceback

import _bootstrap  # noqa: F401

from config import cfg

import DP
from run_test_record_exp_pool_core import PROJECT_ROOT, discover_trace_datasets


def main():
    p = argparse.ArgumentParser(description="Batch DP Oracle ExperiencePool -> wm_traces/DP/")
    p.add_argument("--seed", type=int, default=100003, help="DP.generate_oracle_exp_pool 的 seed")
    p.add_argument(
        "--trace-limit",
        type=int,
        default=-1,
        help="每个 trace 目录内最多使用前 N 个文件；-1 表示全部",
    )
    p.add_argument(
        "--log-dp-text",
        action="store_true",
        help="若为真，则为每个 trace 文件写 DP 回放日志（默认不写，仅 pkl）",
    )
    args_ns = p.parse_args()

    out_root = os.path.join(PROJECT_ROOT, "wm_traces", "DP")
    os.makedirs(out_root, exist_ok=True)

    traces = discover_trace_datasets(PROJECT_ROOT)
    if not traces:
        raise SystemExit("未发现 trace 子目录（data/traces/{train,valid,test}/）")

    videos = sorted(cfg.video_size_dirs.keys())
    total = len(traces) * len(videos)
    print(f"DP Oracle 批量生成: {len(traces)} 个 trace 集 × {len(videos)} 个 video = {total} 个 pkl")
    print(f"输出目录: {os.path.abspath(out_root)}")
    print("=" * 60)

    ok, fail = 0, 0
    for trace_key, trace_dir in traces:
        for video_key in videos:
            safe_trace = trace_key.replace(os.sep, "_").replace("/", "_")
            out_path = os.path.join(out_root, f"{safe_trace}_{video_key}.pkl")
            video_rel = cfg.video_size_dirs[video_key]
            video_size_dir = (
                video_rel
                if os.path.isabs(video_rel)
                else os.path.join(PROJECT_ROOT, video_rel.replace("\\", "/").lstrip("./"))
            )
            label = f"{trace_key} + {video_key}"

            log_dir = None
            if args_ns.log_dp_text:
                log_dir = os.path.join(out_root, "logs", f"{safe_trace}_{video_key}")
                os.makedirs(log_dir, exist_ok=True)

            try:
                DP.generate_oracle_exp_pool(
                    trace_folder=trace_dir,
                    video_size_dir=video_size_dir,
                    output_path=out_path,
                    trace_limit=args_ns.trace_limit,
                    seed=args_ns.seed,
                    log_dir=log_dir,
                )
                ok += 1
                print(f"[OK] {label}\n     已保存: {os.path.abspath(out_path)}")
            except Exception as e:
                fail += 1
                print(f"[FAIL] {label}: {e!r}")
                traceback.print_exc()
            print("-" * 60)

    print(f"完成: 成功 {ok}, 失败 {fail}, 合计 {ok + fail}")


if __name__ == "__main__":
    main()
