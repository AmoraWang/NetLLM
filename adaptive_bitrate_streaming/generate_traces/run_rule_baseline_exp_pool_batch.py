"""
使用与 generate_exp_pool.py 相同的规则类 / 基线算法采集轨迹，批量生成 ExperiencePool pkl。

- **MPC**、**BBA**：纯规则，**不需要 TensorFlow**（依赖 generate_exp_pool 已对 TF 做延迟导入）。
- **genet**、**udr_***：Pensieve TensorFlow Actor，本机需安装 tensorflow 与 ckpt。

对每个 (算法名, trace 数据集目录, video) 生成一个 pkl，路径为::

 wm_traces/{算法名}/{trace_key}_{video}.pkl

扫描 trace 的方式与 run_test_record_exp_pool_core.discover_trace_datasets 一致。

用法（在 adaptive_bitrate_streaming 根目录，无 TF 时只跑规则类）::

    python generate_traces/run_rule_baseline_exp_pool_batch.py --algorithms mpc bba --trace-num -1 --seed 1
"""
import argparse
import os
import pickle
import random
import traceback
from types import SimpleNamespace

import _bootstrap  # noqa: F401

import numpy as np

from config import cfg

import generate_exp_pool as gep
from run_test_record_exp_pool_core import PROJECT_ROOT, discover_trace_datasets

# 与 generate_exp_pool.run 中一致的合法模型名
PENSIEVE_MODEL_NAMES = ("genet", "udr_1", "udr_2", "udr_3", "udr_real")


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_model(model_name: str):
    if model_name == "mpc":
        return gep.MPC, "mpc"
    if model_name == "bba":
        return gep.BBA, "bba"
    if model_name in PENSIEVE_MODEL_NAMES:
        if model_name not in cfg.baseline_model_paths:
            raise ValueError(f"config 中缺少 baseline 权重: {model_name}")
        return gep.PENSIEVE, model_name
    raise ValueError(
        f"未知算法: {model_name}，可选: mpc, bba, {', '.join(PENSIEVE_MODEL_NAMES)}"
    )


def main():
    p = argparse.ArgumentParser(description="Rule/baseline exp pool batch -> wm_traces/{algo}/")
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["mpc", "bba"],
        help="基线名称；默认仅规则类 mpc/bba（无需 TF）。含 genet/udr_* 时需安装 tensorflow。",
    )
    p.add_argument("--trace-num", type=int, default=-1, help="每个 trace 目录内 episode 上限；-1 为全部")
    p.add_argument("--seed", type=int, default=100003)
    p.add_argument("--cuda-id", type=int, default=0)
    p.add_argument("--fixed-order", action="store_true")
    args_ns = p.parse_args()

    for name in args_ns.algorithms:
        resolve_model(name)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_ns.cuda_id)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    _seed_all(args_ns.seed)

    traces = discover_trace_datasets(PROJECT_ROOT)
    if not traces:
        raise SystemExit("未发现 data/traces/{train,valid,test}/ 下的子目录")

    videos = sorted(cfg.video_size_dirs.keys())
    print(f"算法: {args_ns.algorithms}")
    print(f"trace 集: {len(traces)}, videos: {videos}")
    print("=" * 60)

    for algo in args_ns.algorithms:
        model_kind, model_name = resolve_model(algo)
        out_dir = os.path.join(PROJECT_ROOT, "wm_traces", algo)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n>>> 算法 [{algo}] 输出目录: {os.path.abspath(out_dir)}")

        ok, fail = 0, 0

        def run_one_combo(sess, actor_ref):
            nonlocal ok, fail
            for trace_key, trace_dir in traces:
                for video_key in videos:
                    safe_trace = trace_key.replace(os.sep, "_").replace("/", "_")
                    out_path = os.path.join(out_dir, f"{safe_trace}_{video_key}.pkl")
                    label = f"{algo}  {trace_key} + {video_key}"

                    all_cooked_time, all_cooked_bw, all_file_names, all_mahimahi_ptrs = gep.load_traces(
                        trace_dir
                    )
                    trace_num = min(args_ns.trace_num, len(all_file_names))
                    if args_ns.trace_num == -1:
                        trace_num = len(all_file_names)
                    fixed_order = args_ns.fixed_order or (trace_num == len(all_file_names))

                    video_rel = cfg.video_size_dirs[video_key]
                    video_size_dir = (
                        video_rel
                        if os.path.isabs(video_rel)
                        else os.path.join(PROJECT_ROOT, video_rel.replace("\\", "/").lstrip("./"))
                    )

                    env_settings = {
                        "all_cooked_time": all_cooked_time,
                        "all_cooked_bw": all_cooked_bw,
                        "all_file_names": all_file_names,
                        "all_mahimahi_ptrs": all_mahimahi_ptrs,
                        "video_size_dir": video_size_dir,
                        "fixed": fixed_order,
                        "trace_num": trace_num,
                    }

                    fake_args = SimpleNamespace(
                        video=video_key,
                        trace=trace_key,
                        cuda_id=args_ns.cuda_id,
                        seed=args_ns.seed,
                    )

                    try:
                        states, actions, rewards, dones, new_actor = gep.collect_experience(
                            fake_args,
                            model_kind,
                            model_name,
                            env_settings,
                            trace_num,
                            sess,
                            actor_ref,
                        )
                        actor_ref = new_actor
                        pool = gep.ExperiencePool()
                        for i in range(len(states)):
                            pool.add(
                                state=states[i],
                                action=actions[i],
                                reward=rewards[i],
                                done=dones[i],
                            )
                        with open(out_path, "wb") as f:
                            pickle.dump(pool, f)
                        ok += 1
                        print(f"[OK] {label}\n     已保存: {os.path.abspath(out_path)}  (n={len(pool)})")
                    except Exception as e:
                        fail += 1
                        print(f"[FAIL] {label}: {e!r}")
                        traceback.print_exc()
                    print("-" * 60)
            return actor_ref

        if model_kind == gep.PENSIEVE:
            import tensorflow as tf

            tf.random.set_random_seed(args_ns.seed)
            with tf.Session() as sess:
                actor = None
                run_one_combo(sess, actor)
        else:
            run_one_combo(None, None)

        print(f"算法 [{algo}] 完成: 成功 {ok}, 失败 {fail}")

    print("\n全部算法跑完。")


if __name__ == "__main__":
    main()
