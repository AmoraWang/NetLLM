#!/usr/bin/env python3
"""
使用与 run_plm.py --test / plm_special.test.test_on_env 相同的仿真与评测流程，
测试规则算法：
  - bba : Buffer-Based Adaptation（缓冲阈值映射码率）
  - mpc : RobustMPC（generate_exp_pool 中与 Pensieve 一致的 harmonic + 误差鲁棒带宽）

以及 Pensieve 架构的强化学习基线（需 TensorFlow 1.x + 权重，与 run_baseline.py / generate_exp_pool.py 一致）：
  - genet（及可选 udr_*）：ActorNetwork + nn_model_ep_*.ckpt，决策逻辑为 generate_exp_pool.pensieve + action2bitrate。

规则类（bba/mpc）仅需 NumPy + Numba；GENET 等需安装 TensorFlow（见仓库 run_baseline.py 说明，常见为 conda abr_tf）。

示例：
  cd adaptive_bitrate_streaming
  python run/run_rule_baselines_test.py --algorithm mpc --trace fcc-test --video video1 --trace-num 100 --test-rounds 15 --seed 666
  python run/run_rule_baselines_test.py --algorithm both --trace fcc-test --video video1 --test-rounds 10
  python run/run_rule_baselines_test.py --algorithm genet --trace fcc-test --video video1 --trace-num 100 --cuda-id 0
  python run/run_rule_baselines_test.py --algorithm all --trace fcc-test --video video1 --pensieve-model genet

RobustMPC 对应仓库中的 mpc(...)（harmonic_bandwidth/(1+max_error)），即注释中的 robustMPC。

mean_qoe：由 ``plm_special.test.test_on_env`` 调 ``calc_mean_reward`` 聚合，**每条 trace 丢弃首块与末块的 reward**，只统计中间块（例如 49 块时 47 块）；与 ``run_merina_baseline_test`` / ``run_comyco_merina_baseline_test`` 对齐。
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from types import SimpleNamespace

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)
_GEN_POOL_DIR = os.path.join(_ABR_ROOT, "generate_exp_pool")
if _GEN_POOL_DIR not in sys.path:
    sys.path.insert(0, _GEN_POOL_DIR)

import numpy as np

# 与 generate_exp_pool / run_baseline 中 RobustMPC + BBA + Pensieve 一致
from generate_exp_pool import (
    bba,
    calculate_jump_action_combo,
    mpc,
    pensieve,
)
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import (
    A_DIM,
    BITRATE_LEVELS,
    BUFFER_NORM_FACTOR,
    CHUNK_TIL_VIDEO_END_CAP,
    DEFAULT_QUALITY,
    S_INFO,
    S_LEN,
    VIDEO_BIT_RATE,
)
from config import cfg
from plm_special.test import test_on_env


def _load_size_video_array(video_size_dir: str) -> np.ndarray:
    rows = []
    for br in range(BITRATE_LEVELS):
        sizes = []
        path = os.path.join(video_size_dir, f"video_size_{br}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                sizes.append(int(line.split()[0]))
        rows.append(sizes)
    return np.array(rows, dtype=np.float64)


class RuleBaselineAdapter:
    """
    模拟 OfflineRLPolicy.sample(state, target_return, timestep) 接口，
    state 为 torch (1,1,S_INFO,S_LEN)，与 plm_special/test 中一致。
    """

    def __init__(self, algorithm: str, video_size_dir: str):
        assert algorithm in ("bba", "mpc")
        self.algorithm = algorithm
        self.size_video_array = _load_size_video_array(video_size_dir)
        self.video_bit_rate = np.array(VIDEO_BIT_RATE, dtype=np.float64)
        self.combo_dict = {
            str(k): calculate_jump_action_combo(k) for k in range(BITRATE_LEVELS)
        }
        self.past_errors: list = []
        self.past_bandwidth_ests: list = []
        self._completed_quality = DEFAULT_QUALITY

    def sample(self, state, target_return, timestep):
        del target_return  # 规则策略不使用回报条件

        s = state.squeeze(0).squeeze(0).detach().cpu().numpy()
        assert s.shape == (S_INFO, S_LEN)

        if timestep == 0:
            self._completed_quality = DEFAULT_QUALITY
            self.past_errors.clear()
            self.past_bandwidth_ests.clear()

        buffer_sec = float(s[1, -1]) * BUFFER_NORM_FACTOR
        vremain = int(round(float(s[5, -1]) * CHUNK_TIL_VIDEO_END_CAP))
        vremain = max(0, min(vremain, int(CHUNK_TIL_VIDEO_END_CAP)))

        if self.algorithm == "bba":
            br = int(bba(buffer_sec))
        else:
            br = int(
                mpc(
                    self.size_video_array,
                    s,
                    self._completed_quality,
                    buffer_sec,
                    float(vremain),
                    self.video_bit_rate,
                    self.past_errors,
                    self.past_bandwidth_ests,
                    self.combo_dict,
                )
            )

        br = max(0, min(BITRATE_LEVELS - 1, br))
        self._completed_quality = br
        return br


PENSIEVE_MODEL_NAMES = ("genet", "udr_1", "udr_2", "udr_3", "udr_real")


class GenetBaselineAdapter:
    """
    Pensieve / GENET Actor：与 run_baseline.py 中 pensieve(actor, state, last_bit_rate) 一致，
    适配 test_on_env 的 model.sample(state, target_return, timestep)。
    """

    def __init__(self, actor):
        self.actor = actor
        self._last_bit_rate = DEFAULT_QUALITY

    def sample(self, state, target_return, timestep):
        del target_return
        if timestep == 0:
            self._last_bit_rate = DEFAULT_QUALITY

        s = state.squeeze(0).squeeze(0).detach().cpu().numpy()
        assert s.shape == (S_INFO, S_LEN)

        br = int(pensieve(self.actor, s, self._last_bit_rate))
        br = max(0, min(BITRATE_LEVELS - 1, br))
        self._last_bit_rate = br
        return br


def build_env_settings(args_ns: argparse.Namespace) -> dict:
    trace_dir = cfg.trace_dirs[args_ns.trace]
    video_size_dir = cfg.video_size_dirs[args_ns.video]

    all_cooked_time, all_cooked_bw, all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)
    trace_num = min(args_ns.trace_num, len(all_file_names))
    if trace_num == -1:
        trace_num = len(all_file_names)
    fixed_order = args_ns.fixed_order
    if trace_num == len(all_file_names):
        fixed_order = True

    return {
        "all_cooked_time": all_cooked_time,
        "all_cooked_bw": all_cooked_bw,
        "all_file_names": all_file_names,
        "all_mahimahi_ptrs": all_mahimahi_ptrs,
        "video_size_dir": video_size_dir,
        "fixed": fixed_order,
        "trace_num": trace_num,
    }, trace_num, fixed_order


def run_one_algorithm(algo: str, args_ns: argparse.Namespace, env_settings: dict, trace_num: int) -> dict:
    video_size_dir = env_settings["video_size_dir"]
    model = RuleBaselineAdapter(algo, video_size_dir)

    results_dir = os.path.join(
        cfg.results_dir,
        f"{args_ns.trace}_{args_ns.video}",
        f"trace_num_{trace_num}_fixed_{env_settings['fixed']}",
        "rule_baseline",
        algo,
        f"seed_{args_ns.seed}_rounds_{args_ns.test_rounds}",
    )
    os.makedirs(results_dir, exist_ok=True)

    test_args = SimpleNamespace(device=args_ns.device, seed=args_ns.seed)

    logs = test_on_env(
        test_args,
        model,
        results_dir,
        env_settings,
        target_return=0.0,
        max_ep_num=trace_num,
        process_reward_fn=lambda x: x,
        seed=args_ns.seed,
        num_test_rounds=args_ns.test_rounds,
    )
    print(f"\n=== {algo.upper()} ===")
    print(f"mean_qoe (聚合输出): {logs['mean_qoe']:.6f}")
    print(f"结果目录: {results_dir}")
    return logs


def run_one_pensieve_rl(model_name: str, args_ns: argparse.Namespace, env_settings: dict, trace_num: int) -> dict:
    """GENET / UDR_*：TensorFlow 1.x Session + a3c.ActorNetwork，评测流程与 test_on_env 对齐。"""
    if model_name not in cfg.baseline_model_paths:
        raise ValueError(f"config 中无 baseline 路径: {model_name}")

    ckpt = cfg.baseline_model_paths[model_name]

    def _ckpt_prefix_exists(prefix: str) -> bool:
        if os.path.isfile(prefix):
            return True
        if os.path.isfile(prefix + ".index"):
            return True
        if glob.glob(prefix + ".meta"):
            return True
        if glob.glob(prefix + ".data-*"):
            return True
        return False

    if not _ckpt_prefix_exists(ckpt):
        raise FileNotFoundError(
            f"未找到 Pensieve 权重文件（TensorFlow checkpoint 前缀）: {ckpt}\n"
            f"请将 nn_model_ep_*.ckpt* 放到 config.baseline_model_paths['{model_name}'] 所指目录。"
        )

    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "运行 GENET / UDR 基线需要安装 TensorFlow（仓库中 run_baseline.py 使用 TF1 API）。"
            "若使用 conda，可尝试论文环境说明中的 tensorflow 1.x 环境。"
        ) from e

    import baseline_special.a3c as a3c

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_ns.cuda_id)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    np.random.seed(args_ns.seed)
    tf.random.set_random_seed(args_ns.seed)
    random.seed(args_ns.seed)

    results_dir = os.path.join(
        cfg.results_dir,
        f"{args_ns.trace}_{args_ns.video}",
        f"trace_num_{trace_num}_fixed_{env_settings['fixed']}",
        "rl_baseline",
        model_name,
        f"seed_{args_ns.seed}_rounds_{args_ns.test_rounds}",
    )
    os.makedirs(results_dir, exist_ok=True)

    test_args = SimpleNamespace(device=args_ns.device, seed=args_ns.seed)

    with tf.Session() as sess:
        actor = a3c.ActorNetwork(
            sess,
            state_dim=[S_INFO, S_LEN],
            action_dim=A_DIM,
            bitrate_dim=BITRATE_LEVELS,
        )
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        print(f"Pensieve 模型已加载: {model_name} <- {ckpt}")

        model = GenetBaselineAdapter(actor)
        logs = test_on_env(
            test_args,
            model,
            results_dir,
            env_settings,
            target_return=0.0,
            max_ep_num=trace_num,
            process_reward_fn=lambda x: x,
            seed=args_ns.seed,
            num_test_rounds=args_ns.test_rounds,
        )

    print(f"\n=== {model_name.upper()} (RL / Pensieve) ===")
    print(f"mean_qoe (聚合输出): {logs['mean_qoe']:.6f}")
    print(f"结果目录: {results_dir}")
    return logs


def main():
    p = argparse.ArgumentParser(description="BBA / RobustMPC / GENET(Pensieve) 基线，流程对齐 test_on_env")
    p.add_argument(
        "--algorithm",
        choices=("bba", "mpc", "genet", "both", "all"),
        default="both",
        help="both=bba+mpc；all=bba+mpc+由 --pensieve-model 指定的 RL 模型",
    )
    p.add_argument(
        "--pensieve-model",
        type=str,
        default="genet",
        choices=PENSIEVE_MODEL_NAMES,
        help="在 --algorithm genet|all 时使用的 checkpoint 名称（对应 config.baseline_model_paths）",
    )
    p.add_argument("--trace", type=str, default="fcc-test")
    p.add_argument("--video", type=str, default="video1")
    p.add_argument("--trace-num", type=int, default=100, help="评测 episode 数；-1 表示目录内全部 trace")
    p.add_argument("--fixed-order", action="store_true", help="按固定顺序遍历 trace（与 run_plm 一致）")
    p.add_argument("--seed", type=int, default=100003)
    p.add_argument("--test-rounds", type=int, default=1, help="多轮随机带宽起点；>1 时 mean_qoe 为各轮平均")
    p.add_argument("--device", type=str, default="cpu", help="state 张量所在设备，规则策略仅在 CPU 上算 numpy")
    p.add_argument("--cuda-id", type=int, default=0, help="GENET/UDR 使用的 GPU（写入 CUDA_VISIBLE_DEVICES，TF1）")
    args_ns = p.parse_args()

    assert args_ns.trace in cfg.trace_dirs, f"未知 trace: {args_ns.trace}"
    assert args_ns.video in cfg.video_size_dirs, f"未知 video: {args_ns.video}"

    env_settings, trace_num, _ = build_env_settings(args_ns)
    env_settings["trace_num"] = trace_num

    if args_ns.algorithm == "both":
        algos = ["bba", "mpc"]
    elif args_ns.algorithm == "all":
        algos = ["bba", "mpc", "__rl__"]
    elif args_ns.algorithm == "genet":
        algos = ["__rl__"]
    else:
        algos = [args_ns.algorithm]

    for algo in algos:
        if algo == "__rl__":
            run_one_pensieve_rl(args_ns.pensieve_model, args_ns, env_settings, trace_num)
        else:
            run_one_algorithm(algo, args_ns, env_settings, trace_num)

    return 0


if __name__ == "__main__":
    sys.exit(main())
