#!/usr/bin/env python3
"""
在 NetLLM 的带宽 trace 与各码率 video_size_* 上评测 Merina（状态与仿真逻辑完全使用 merina 代码）。

- 轨迹：`config.cfg.trace_dirs` + `baseline_special.utils.utils.load_traces`（与 run_rule_baselines_test 一致）。
- 环境：基于 `merina/envs/fixed_env_log.py` 的 `Environment` 子类（随机带宽起点 + 与 `merina/algos/test_v5.py` 中 `evaluation` 相同的状态与奖励）。
- 权重默认：`merina/models/pretrain_policy_lin.model` 与 `pretrain_vae_lin.model`（linear QoE，与 evaluation 中 args.log=False 一致）。
- 平均 QoE：与 `run_rule_baselines_test` / `plm_special.utils.calc_mean_reward` 一致——**每条 trace 丢弃首块与末块的 reward**，只取中间块（例如 49 块时中间 47 块），再对所有 trace 的中间块 reward 做全局平均。
- 多轮（`--test-rounds`>1）：**mean_qoe 为各轮 mean_qoe 的算术平均**；每轮 numpy 种子与 `plm_special.test.test_on_env` 一致。
- 带宽起点：与 `baseline_special.utils.mahimahi_start` 生成的指针表一致（与 Pensieve `Environment` / `test_on_env` 同 seed、同 `--fixed-order`）。

示例：
  cd adaptive_bitrate_streaming
  python run/run_merina_baseline_test.py --trace fcc-test --video video1 --trace-num 100
  python run/run_merina_baseline_test.py --trace fcc-test --video video1 --test-rounds 15 --seed 666
  python run/run_merina_baseline_test.py --trace fcc-test --video video1 --cuda-id 0 --stocha
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)
_MERINA_ROOT = os.path.join(_ABR_ROOT, "merina")
_DEFAULT_POLICY = os.path.join(_MERINA_ROOT, "models", "pretrain_policy_lin.model")
_DEFAULT_VAE = os.path.join(_MERINA_ROOT, "models", "pretrain_vae_lin.model")

# 与 merina/main.py 中 set_env_info 一致（test_v5.evaluation 依赖这些超参）
MERINA_S_INFO = 11
MERINA_S_LEN = 2
MERINA_C_LEN = 8
MERINA_VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
MERINA_REBUF_LIN = 4.3
MERINA_REBUF_LOG = 2.66
MERINA_SMOOTH_PENALTY = 1


def _count_video_chunks(video_size_dir: str) -> int:
    path0 = os.path.join(video_size_dir, "video_size_0")
    with open(path0, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _merina_video_size_prefix(video_size_dir: str) -> str:
    """Merina 使用 video_size_file + str(br) 打开各档文件，需以 video_size_ 结尾。"""
    d = os.path.abspath(video_size_dir)
    if not d.endswith(os.sep):
        d = d + os.sep
    return os.path.join(d, "video_size_")


def _aggregate_mean_qoe_from_logs(log_dir: str, name_tag: str) -> tuple[float, list[float]]:
    """
    与 ``plm_special.utils.calc_mean_reward(..., skip_first_reward=True, skip_last_reward=True)`` 一致：
    每个 trace 日志收集全部 reward 行后取 ``[1:-1]``（去掉首尾块），再拼成全局列表求 ``np.mean``。
    ``per_trace`` 为各文件中间块的均值。
    """
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


def run_merina_eval(args_ns: argparse.Namespace) -> dict:
    # 须在 import torch / merina.algos.test_v5 之前设置，否则 test_v5 内 dtype 与设备不一致
    cid = getattr(args_ns, "cuda_id", None)
    if cid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if int(cid) < 0 else str(int(cid))

    sys.path.insert(0, _MERINA_ROOT)

    import torch
    from algos.beta_vae_v6 import BetaVAE
    from algos.AC_net_v6 import Actor
    from algos import test_v5 as merina_test
    import envs.fixed_env_log as merina_env_test

    from baseline_special.utils.mahimahi_start import (
        build_mahimahi_ptrs_by_trace_index,
        derive_round_numpy_seed,
        wrap_merina_env_with_shared_ptrs,
    )
    from baseline_special.utils.utils import load_traces
    from config import cfg

    trace_dir = cfg.trace_dirs[args_ns.trace]
    video_size_dir = cfg.video_size_dirs[args_ns.video]
    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(f"trace 目录不存在: {trace_dir}")
    if not os.path.isdir(video_size_dir):
        raise FileNotFoundError(f"video 目录不存在: {video_size_dir}")

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

    SharedPtrMerinaEnv = wrap_merina_env_with_shared_ptrs(merina_env_test.Environment)

    video_prefix = _merina_video_size_prefix(video_size_dir)
    for br in range(6):
        p = video_prefix + str(br)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少 Merina 所需文件: {p}")

    total_chunk_num = _count_video_chunks(video_size_dir)

    test_rounds = max(1, int(getattr(args_ns, "test_rounds", 1)))
    multi = test_rounds > 1

    s_info, s_len, c_len = MERINA_S_INFO, MERINA_S_LEN, MERINA_C_LEN
    bitrate_versions = MERINA_VIDEO_BIT_RATE
    a_dim = len(bitrate_versions)
    rebuff_p = MERINA_REBUF_LOG if args_ns.log else MERINA_REBUF_LIN
    latent_dim = args_ns.latent_dim

    policy_path = os.path.abspath(args_ns.policy_model)
    vae_path = os.path.abspath(args_ns.vae_model)
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"未找到 policy 权重: {policy_path}")
    if not os.path.isfile(vae_path):
        raise FileNotFoundError(f"未找到 VAE 权重: {vae_path}")

    map_loc = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_actor = Actor(a_dim, latent_dim, s_info, s_len)
    model_vae = BetaVAE(in_channels=2, hist_dim=c_len, latent_dim=latent_dim)
    model_actor.load_state_dict(torch.load(policy_path, map_location=map_loc))
    model_vae.load_state_dict(torch.load(vae_path, map_location=map_loc))
    model_actor.eval()
    model_vae.eval()
    model_actor = model_actor.to(map_loc)
    model_vae = model_vae.to(map_loc)

    results_dir = os.path.join(
        cfg.results_dir,
        f"{args_ns.trace}_{args_ns.video}",
        f"trace_num_{trace_num}_merina",
        f"seed_{args_ns.seed}_rounds_{test_rounds}",
    )
    os.makedirs(results_dir, exist_ok=True)
    if multi:
        _clear_merina_round_subdirs(results_dir)

    eval_args = SimpleNamespace(log=args_ns.log, stocha=args_ns.stocha)
    round_mean_qoes: list[float] = []
    last_per_trace: list[float] = []

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
        log_path_ini = os.path.join(round_out, "log_test_merina")

        test_env = SharedPtrMerinaEnv(
            all_cooked_time=all_cooked_time,
            all_cooked_bw=all_cooked_bw,
            all_file_names=all_file_names,
            video_size_file=video_prefix,
            mahimahi_ptrs_by_trace_index=mahimahi_ptrs,
        )
        test_env.set_env_info(
            MERINA_S_INFO,
            MERINA_S_LEN,
            MERINA_C_LEN,
            total_chunk_num,
            MERINA_VIDEO_BIT_RATE,
            1,
            rebuff_p,
            MERINA_SMOOTH_PENALTY,
            0,
        )

        merina_test.evaluation(
            model_actor,
            model_vae,
            log_path_ini,
            test_env,
            s_info,
            s_len,
            c_len,
            total_chunk_num,
            bitrate_versions,
            rebuff_p,
            MERINA_SMOOTH_PENALTY,
            a_dim,
            eval_args,
        )

        mean_qoe_r, per_trace = _aggregate_mean_qoe_from_logs(round_out, "log_test_merina")
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
        "num_test_rounds": test_rounds,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Merina 基线：NetLLM trace + video_size_*，输出平均 QoE")
    p.add_argument("--trace", type=str, default="fcc-test")
    p.add_argument("--video", type=str, default="video1")
    p.add_argument("--trace-num", type=int, default=100, help="评测 trace 条数；-1 为目录内全部")
    p.add_argument("--seed", type=int, default=42, help="共享 mahimahi 起点表种子（与 test_on_env / run_rule_baselines_test 一致）")
    p.add_argument("--fixed-order", action="store_true", help="按固定 trace 序号生成起点表（trace_num=全部时自动开启）")
    p.add_argument(
        "--test-rounds",
        type=int,
        default=10,
        help="多轮评测；>1 时 mean_qoe 为各轮 mean_qoe 平均，与 run_rule_baselines_test / test_on_env 一致",
    )
    p.add_argument(
        "--policy-model",
        type=str,
        default=_DEFAULT_POLICY,
        help="Actor 权重路径（默认 merina/models/pretrain_policy_lin.model）",
    )
    p.add_argument(
        "--vae-model",
        type=str,
        default=_DEFAULT_VAE,
        help="BetaVAE 权重路径（默认 merina/models/pretrain_vae_lin.model）",
    )
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--log", action="store_true", help="使用对数 QoE（与 Merina --log 一致，需 log 训练权重）")
    p.add_argument("--stocha", action="store_true", help="随机策略（multinomial）；默认贪心 argmax")
    p.add_argument(
        "--cuda-id",
        type=int,
        default=None,
        help="若指定且 >=0：评测前设置 CUDA_VISIBLE_DEVICES；<0：强制仅用 CPU。省略则不改环境变量。",
    )
    args_ns = p.parse_args()

    from config import cfg

    if args_ns.trace not in cfg.trace_dirs:
        raise SystemExit(f"未知 trace: {args_ns.trace}")
    if args_ns.video not in cfg.video_size_dirs:
        raise SystemExit(f"未知 video: {args_ns.video}")

    out = run_merina_eval(args_ns)
    print(f"\nmean_qoe（与 run_rule_baselines_test 一致：每条 trace 去掉首尾块，仅中间块）: {out['mean_qoe']:.6f}")
    if out.get("num_test_rounds", 1) > 1:
        print(f"std_qoe_across_rounds: {out['std_qoe_across_rounds']:.6f}")
        print(f"mean_qoe_per_round: {out['mean_qoe_per_round']}")
    print(f"results_dir: {out['results_dir']}")
    print(f"total_chunk_num: {out['total_chunk_num']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
