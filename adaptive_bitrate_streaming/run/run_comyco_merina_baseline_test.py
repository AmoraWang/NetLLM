#!/usr/bin/env python3
"""
在 NetLLM 的带宽 trace 与各码率 video_size_* 上评测 Merina 仓库内复现的 Comyco（TensorFlow 1.x + tflearn）。

实现与 ``merina/baselines/comyco.py`` 一致：
  - 环境：基于 ``merina/envs/fixed_env_log.Environment`` 的子类，使用 ``baseline_special.utils.mahimahi_start`` 共享起点表
  - 网络：``merina/baselines/libcomyco_lin.libcomyco``，决策为 ``argmax(action_prob)``（与 comyco.py 主循环一致，忽略 predict 内的随机档位）
  - 状态：S_INFO=6, S_LEN=8（与 Pensieve/NetLLM baseline 同构）

默认权重前缀（TensorFlow checkpoint）：
  ``merina/models/baselines/comyco/lin/lin/nn_model_ep_580.ckpt``

平均 QoE：与 ``run_merina_baseline_test.py`` / ``run_rule_baselines_test`` / ``calc_mean_reward`` 一致——每条 trace 去掉首尾块 reward，只取中间块再全局平均。
多轮（``--test-rounds``>1）：**mean_qoe 为各轮 mean_qoe 的算术平均**；每轮 numpy 种子与 ``plm_special.test.test_on_env`` 多轮派生方式一致。

示例：
  cd adaptive_bitrate_streaming
  python run/run_comyco_merina_baseline_test.py --trace fcc-test --video video1 --trace-num 100
  python run/run_comyco_merina_baseline_test.py --trace fcc-test --video video1 --test-rounds 15 --seed 666
  python run/run_comyco_merina_baseline_test.py --cuda-id 0
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)
_MERINA_ROOT = os.path.join(_ABR_ROOT, "merina")
_DEFAULT_CKPT = os.path.join(
    _MERINA_ROOT, "models", "baselines", "comyco", "lin", "lin", "nn_model_ep_580.ckpt"
)

# 与 merina/baselines/comyco.py 一致
S_INFO = 6
S_LEN = 8
A_DIM = 6
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
REBUF_PENALTY_lin = 4.3
REBUF_PENALTY_log = 2.66
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1


def _count_video_chunks(video_size_dir: str) -> int:
    path0 = os.path.join(video_size_dir, "video_size_0")
    with open(path0, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _merina_video_size_prefix(video_size_dir: str) -> str:
    d = os.path.abspath(video_size_dir)
    if not d.endswith(os.sep):
        d = d + os.sep
    return os.path.join(d, "video_size_")


def _aggregate_mean_qoe_mid_chunks(log_dir: str, name_tag: str) -> tuple[float, list[float]]:
    """每条 trace 取全部 reward 行后 ``[1:-1]``（去掉首尾块），再全局平均（与 calc_mean_reward 一致）。"""
    pooled: list[float] = []
    per_trace: list[float] = []
    if not os.path.isdir(log_dir):
        return 0.0, per_trace
    for fname in sorted(os.listdir(log_dir)):
        if name_tag not in fname or fname.startswith("."):
            continue
        fpath = os.path.join(log_dir, fname)
        if not os.path.isfile(fpath):
            continue
        file_rewards: list[float] = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) <= 1:
                    continue
                try:
                    val = float(parts[-1])
                except (ValueError, IndexError):
                    continue
                file_rewards.append(val)
        mid = file_rewards[1:-1] if len(file_rewards) > 2 else []
        if mid:
            per_trace.append(float(np.mean(mid)))
            pooled.extend(mid)
    if not pooled:
        return 0.0, per_trace
    return float(np.mean(pooled)), per_trace


def _clear_merina_round_subdirs(parent: str) -> None:
    if not os.path.isdir(parent):
        return
    for name in os.listdir(parent):
        if name.startswith("round_") and os.path.isdir(os.path.join(parent, name)):
            shutil.rmtree(os.path.join(parent, name), ignore_errors=True)


def _patch_sklearn_externals_for_tf15_compat() -> None:
    import types

    try:
        import six
    except ImportError:
        return
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return
    ext = getattr(sklearn, "externals", None)
    if ext is not None and hasattr(ext, "six"):
        return
    if ext is None:
        ext = types.ModuleType("sklearn.externals")
        sys.modules["sklearn.externals"] = ext
        setattr(sklearn, "externals", ext)
    sys.modules["sklearn.externals.six"] = six
    ext.six = six  # type: ignore[attr-defined]


def _ckpt_prefix_exists(prefix: str) -> bool:
    if os.path.isfile(prefix):
        return True
    if os.path.isfile(prefix + ".index"):
        return True
    import glob

    if glob.glob(prefix + ".meta"):
        return True
    if glob.glob(prefix + ".data-*"):
        return True
    return False


def run_comyco_merina_eval(args_ns: argparse.Namespace) -> dict:
    cid = getattr(args_ns, "cuda_id", None)
    if cid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if int(cid) < 0 else str(int(cid))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    _patch_sklearn_externals_for_tf15_compat()

    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "Merina Comyco 基线需要 TensorFlow 1.x（与 merina/baselines/comyco.py 一致）。"
        ) from e

    sys.path.insert(0, _MERINA_ROOT)
    sys.path.insert(0, os.path.join(_MERINA_ROOT, "baselines"))
    import libcomyco_lin as libcomyco  # noqa: E402
    import envs.fixed_env_log as merina_fixed_env  # noqa: E402

    from baseline_special.utils.mahimahi_start import (  # noqa: E402
        build_mahimahi_ptrs_by_trace_index,
        derive_round_numpy_seed,
        wrap_merina_env_with_shared_ptrs,
    )
    from baseline_special.utils.utils import load_traces  # noqa: E402
    from config import cfg  # noqa: E402

    trace_dir = cfg.trace_dirs[args_ns.trace]
    video_size_dir = cfg.video_size_dirs[args_ns.video]
    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(f"trace 目录不存在: {trace_dir}")
    if not os.path.isdir(video_size_dir):
        raise FileNotFoundError(f"video 目录不存在: {video_size_dir}")

    ckpt = os.path.abspath(args_ns.checkpoint)
    if not _ckpt_prefix_exists(ckpt):
        raise FileNotFoundError(
            f"未找到 TensorFlow checkpoint 前缀: {ckpt}\n"
            "请确认 merina/models/baselines/comyco/lin/lin/ 下存在 nn_model_ep_580.ckpt.*"
        )

    all_cooked_time, all_cooked_bw, all_file_names, _ = load_traces(trace_dir)
    n_avail = len(all_file_names)
    trace_num = args_ns.trace_num
    if trace_num == -1:
        trace_num = n_avail
    trace_num = min(trace_num, n_avail)
    if trace_num <= 0:
        raise ValueError("trace_num 必须为正或 -1")

    fixed_order = bool(getattr(args_ns, "fixed_order", False))
    if trace_num == n_avail:
        fixed_order = True

    all_cooked_bw_full = all_cooked_bw
    all_cooked_time = all_cooked_time[:trace_num]
    all_cooked_bw = all_cooked_bw[:trace_num]
    all_file_names = all_file_names[:trace_num]

    SharedPtrMerinaEnv = wrap_merina_env_with_shared_ptrs(merina_fixed_env.Environment)

    video_prefix = _merina_video_size_prefix(video_size_dir)
    for br in range(A_DIM):
        p = video_prefix + str(br)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少 video_size 文件: {p}")

    total_chunk_num = _count_video_chunks(video_size_dir)
    chunk_cap = float(total_chunk_num)

    test_rounds = max(1, int(getattr(args_ns, "test_rounds", 1)))
    multi = test_rounds > 1
    rebuff_p = REBUF_PENALTY_log if args_ns.log else REBUF_PENALTY_lin
    rebuffer_penalty = rebuff_p
    smooth_penalty = SMOOTH_PENALTY
    bitrate_versions = VIDEO_BIT_RATE

    results_dir = os.path.join(
        cfg.results_dir,
        f"{args_ns.trace}_{args_ns.video}",
        f"trace_num_{trace_num}_comyco_merina",
        f"seed_{args_ns.seed}_rounds_{test_rounds}",
    )
    os.makedirs(results_dir, exist_ok=True)
    if multi:
        _clear_merina_round_subdirs(results_dir)

    log_tag = "log_test_comyco_merina"

    if hasattr(tf, "set_random_seed"):
        tf.set_random_seed(args_ns.seed)
    elif hasattr(tf.random, "set_random_seed"):
        tf.random.set_random_seed(args_ns.seed)

    round_mean_qoes: list[float] = []
    last_per_trace: list[float] = []

    with tf.Session() as sess:
        actor = libcomyco.libcomyco(sess, S_INFO, S_LEN, A_DIM, LR_RATE=1e-4)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)

        for r in range(test_rounds):
            round_seed = derive_round_numpy_seed(r, test_rounds, args_ns.seed)
            ptrs_full = build_mahimahi_ptrs_by_trace_index(
                all_cooked_bw_full,
                seed=round_seed,
                fixed=fixed_order,
            )
            mahimahi_ptrs = ptrs_full[:trace_num]
            round_out = os.path.join(results_dir, f"round_{r:03d}") if multi else results_dir
            os.makedirs(round_out, exist_ok=True)
            log_path_ini = os.path.join(round_out, log_tag)

            test_env = SharedPtrMerinaEnv(
                all_cooked_time=all_cooked_time,
                all_cooked_bw=all_cooked_bw,
                all_file_names=all_file_names,
                video_size_file=video_prefix,
                mahimahi_ptrs_by_trace_index=mahimahi_ptrs,
            )
            test_env.set_env_info(0, 0, 0, int(total_chunk_num), VIDEO_BIT_RATE, 1, rebuff_p, SMOOTH_PENALTY, 0)

            log_path = log_path_ini + "_" + all_file_names[test_env.trace_idx]
            log_file = open(log_path, "w", encoding="utf-8")

            time_stamp = 0.0
            bit_rate = DEFAULT_QUALITY
            last_bit_rate = DEFAULT_QUALITY

            s_batch: list[np.ndarray] = [np.zeros((S_INFO, S_LEN))]
            video_count = 0

            while True:
                delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain, _ = test_env.get_video_chunk(int(bit_rate))

                time_stamp += delay
                time_stamp += sleep_time

                if args_ns.log:
                    log_bit_rate = np.log(bitrate_versions[bit_rate] / float(bitrate_versions[0]))
                    log_last_bit_rate = np.log(bitrate_versions[last_bit_rate] / float(bitrate_versions[0]))
                    reward = (
                        log_bit_rate
                        - rebuffer_penalty * rebuf
                        - smooth_penalty * np.abs(log_bit_rate - log_last_bit_rate)
                    )
                else:
                    reward = (
                        bitrate_versions[bit_rate] / M_IN_K
                        - rebuffer_penalty * rebuf
                        - smooth_penalty
                        * np.abs(bitrate_versions[bit_rate] - bitrate_versions[last_bit_rate])
                        / M_IN_K
                    )
                last_bit_rate = bit_rate

                log_file.write(
                    f"{time_stamp / M_IN_K}\t{VIDEO_BIT_RATE[bit_rate]}\t{buffer_size}\t{rebuf}\t"
                    f"{video_chunk_size}\t{delay}\t{reward}\n"
                )
                log_file.flush()

                state = np.array(s_batch[-1], copy=True)
                state = np.roll(state, -1, axis=1)
                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
                state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
                state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
                state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
                state[5, -1] = np.minimum(video_chunk_remain, chunk_cap) / chunk_cap

                action_prob, _ = actor.predict(np.reshape(state, (-1, S_INFO, S_LEN)))
                bit_rate = int(np.argmax(action_prob[0]))

                s_batch.append(state)

                if end_of_video:
                    log_file.write("\n")
                    log_file.close()

                    bit_rate = DEFAULT_QUALITY
                    last_bit_rate = DEFAULT_QUALITY
                    del s_batch[:]
                    s_batch.append(np.zeros((S_INFO, S_LEN)))

                    video_count += 1
                    if video_count >= len(all_file_names):
                        break

                    log_path = log_path_ini + "_" + all_file_names[test_env.trace_idx]
                    log_file = open(log_path, "w", encoding="utf-8")

            mean_qoe_r, per_trace = _aggregate_mean_qoe_mid_chunks(round_out, log_tag)
            round_mean_qoes.append(mean_qoe_r)
            last_per_trace = per_trace

    mean_qoe = float(np.mean(round_mean_qoes)) if round_mean_qoes else 0.0
    std_qoe_across_rounds = float(np.std(round_mean_qoes)) if len(round_mean_qoes) > 1 else 0.0

    if multi:
        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write(f"num_test_rounds\t{test_rounds}\n")
            sf.write(f"mean_qoe\t{mean_qoe:.6f}\n")
            sf.write(f"std_qoe_across_rounds\t{std_qoe_across_rounds:.6f}\n")
            sf.write(f"mean_qoe_per_round\t{round_mean_qoes}\n")

    return {
        "mean_qoe": mean_qoe,
        "mean_qoe_per_round": round_mean_qoes,
        "std_qoe_across_rounds": std_qoe_across_rounds,
        "per_trace_mean_qoe": last_per_trace,
        "results_dir": results_dir,
        "trace_num": trace_num,
        "total_chunk_num": total_chunk_num,
        "checkpoint": ckpt,
        "num_test_rounds": test_rounds,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Merina 复现 Comyco 基线：NetLLM trace + video_size_*")
    p.add_argument("--trace", type=str, default="fcc-test")
    p.add_argument("--video", type=str, default="video1")
    p.add_argument("--trace-num", type=int, default=100, help="评测 trace 条数；-1 为全部")
    p.add_argument("--seed", type=int, default=42, help="共享 mahimahi 起点表种子")
    p.add_argument("--fixed-order", action="store_true", help="按固定 trace 序号生成起点表（trace_num=全部时自动开启）")
    p.add_argument(
        "--test-rounds",
        type=int,
        default=100,
        help="多轮评测；>1 时 mean_qoe 为各轮 mean_qoe 平均，与 run_rule_baselines_test / test_on_env 一致",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=_DEFAULT_CKPT,
        help="TF checkpoint 前缀路径（默认 lin/lin/nn_model_ep_580.ckpt）",
    )
    p.add_argument("--log", action="store_true", help="对数 QoE（需 log 权重目录下模型）")
    p.add_argument(
        "--cuda-id",
        type=int,
        default=None,
        help=">=0：CUDA_VISIBLE_DEVICES；<0：仅 CPU。省略则不改环境变量。",
    )
    args_ns = p.parse_args()

    from config import cfg

    if args_ns.trace not in cfg.trace_dirs:
        raise SystemExit(f"未知 trace: {args_ns.trace}")
    if args_ns.video not in cfg.video_size_dirs:
        raise SystemExit(f"未知 video: {args_ns.video}")

    out = run_comyco_merina_eval(args_ns)
    print(f"\nmean_qoe（每条 trace 去掉首尾块，仅中间块，全局平均）: {out['mean_qoe']:.6f}")
    if out.get("num_test_rounds", 1) > 1:
        print(f"std_qoe_across_rounds: {out['std_qoe_across_rounds']:.6f}")
        print(f"mean_qoe_per_round: {out['mean_qoe_per_round']}")
    print(f"results_dir: {out['results_dir']}")
    print(f"checkpoint: {out['checkpoint']}")
    print(f"total_chunk_num: {out['total_chunk_num']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
