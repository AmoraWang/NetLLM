#!/usr/bin/env python3
"""
在 NetLLM 带宽 trace 上运行 Merina（Actor + β-VAE），逐步记录：

- ``buffer_size``：每 chunk 下载后的缓冲（秒）
- ``action``：Actor 选出的码率档位 0–5
- ``belief_mu`` / ``belief_logvar`` / ``belief_latent``：Merina VAE 后验与送入 Actor 的 latent

每条网络 trace 写入一个 ``.pt`` 文件，结构::

    {
        "trajectory": TensorDict(..., batch_size=[T]),
        "meta": {trace_name, trace_index, video, latent_dim, ...},
    }

默认与 ``run_merina_baseline_test.py`` 使用相同 Merina 权重与 linear QoE。

示例::

  cd adaptive_bitrate_streaming
  python run/collect_merina_trajectory_tensordict.py \\
      --traces fcc-train --video video1 \\
      --output-dir data/traces/train/merina_trajectories/fcc-train_video1

  python run/collect_merina_trajectory_tensordict.py \\
      --trace-dir data/traces/train/fcc_hsdpa_cooked_traces \\
      --video video1 --max-traces 10 --cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN_POOL = os.path.join(_ABR_ROOT, "generate_exp_pool")
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)
if _GEN_POOL not in sys.path:
    sys.path.insert(0, _GEN_POOL)

import numpy as np

from generate_merina_exp_pool_v3 import (
    DEFAULT_MERINA_TRACE_DIR,
    MERINA_C_LEN,
    MERINA_DEFAULT_QUALITY,
    MERINA_REBUF_LIN,
    MERINA_REBUF_LOG,
    MERINA_S_INFO,
    MERINA_S_LEN,
    MERINA_SMOOTH_PENALTY,
    MERINA_VIDEO_BIT_RATE,
    M_IN_K,
    _count_video_chunks,
    _merina_video_size_prefix,
    load_traces_from_dir,
    merge_trace_corpora,
)
from run.merina_trajectory_tensordict_store import (
    build_merina_trajectory_tensordict,
    make_trace_pt_path,
    save_merina_trajectory_pt,
)

_MERINA_ROOT = os.path.join(_ABR_ROOT, "merina")
# _DEFAULT_POLICY = os.path.join(_MERINA_ROOT, "models", "pretrain_policy_lin.model")
# _DEFAULT_VAE = os.path.join(_MERINA_ROOT, "models", "pretrain_vae_lin.model")
_DEFAULT_POLICY = os.path.join(_MERINA_ROOT, "models", "policy_merina_ppo_2000.model")
_DEFAULT_VAE = os.path.join(_MERINA_ROOT, "models", "VAE_merina_ppo_2000.model")


def _resolve_traces(args_ns: argparse.Namespace) -> tuple[list, list, list, str]:
    if args_ns.traces:
        all_t, all_b, all_n = merge_trace_corpora(list(args_ns.traces))
        src = f"config 键 {list(args_ns.traces)}"
    else:
        all_t, all_b, all_n = load_traces_from_dir(args_ns.trace_dir)
        src = args_ns.trace_dir
    if not all_n:
        raise RuntimeError("未加载到任何 trace")
    return all_t, all_b, all_n, src


def _make_merina_env(
    *,
    cooked_time,
    cooked_bw,
    trace_name: str,
    video_prefix: str,
    args_ns: argparse.Namespace,
    rebuff_p: float,
    mahimahi_ptr: int | None,
):
    sys.path.insert(0, _MERINA_ROOT)
    import envs.fixed_env_log as merina_env_mod  # noqa: WPS433

    from baseline_special.utils.mahimahi_start import wrap_merina_env_with_shared_ptrs

    base_kw = dict(
        all_cooked_time=[cooked_time],
        all_cooked_bw=[cooked_bw],
        all_file_names=[trace_name],
        video_size_file=video_prefix,
        random_seed=int(args_ns.seed) % (2**32),
    )

    if args_ns.random_mahimahi_start:
        class RandomStartMerinaEnv(merina_env_mod.Environment):
            def __init__(self, *a, random_seed=42, **kw):
                super().__init__(*a, random_seed=random_seed, **kw)
                self._reroll_mahimahi_start()

            def _reroll_mahimahi_start(self) -> None:
                n = len(self.cooked_bw)
                if n > 1:
                    self.mahimahi_ptr = int(np.random.randint(1, n))
                else:
                    self.mahimahi_ptr = 1
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

            def get_video_chunk(self, quality):
                out = super().get_video_chunk(quality)
                if out[6]:
                    self._reroll_mahimahi_start()
                return out

        return RandomStartMerinaEnv(**base_kw)

    SharedPtrMerinaEnv = wrap_merina_env_with_shared_ptrs(merina_env_mod.Environment)
    ptr = 1 if mahimahi_ptr is None else int(mahimahi_ptr)
    return SharedPtrMerinaEnv(
        **base_kw,
        mahimahi_ptrs_by_trace_index=[ptr],
    )


def rollout_one_trace(
    *,
    cooked_time,
    cooked_bw,
    trace_name: str,
    trace_index: int,
    video_key: str,
    video_prefix: str,
    total_chunk_num: int,
    model_actor,
    model_vae,
    device_torch: str,
    args_ns: argparse.Namespace,
    rebuff_p: float,
    mahimahi_ptr: int | None,
) -> tuple[object, dict]:
    """跑完一条 trace 的一个 episode，返回 (TensorDict, meta)。"""
    import torch

    from baseline_special.utils.constants import BITRATE_LEVELS, BUFFER_NORM_FACTOR

    test_env = _make_merina_env(
        cooked_time=cooked_time,
        cooked_bw=cooked_bw,
        trace_name=trace_name,
        video_prefix=video_prefix,
        args_ns=args_ns,
        rebuff_p=rebuff_p,
        mahimahi_ptr=mahimahi_ptr,
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
    _, _, c_len, _, bitrate_versions, _, _ = test_env.get_env_info()
    a_dim = len(bitrate_versions)

    state_policy = np.zeros((MERINA_S_INFO, MERINA_S_LEN), dtype=np.float32)
    ob = np.zeros((2, c_len), dtype=np.float32)

    bit_rate = MERINA_DEFAULT_QUALITY
    last_bit_rate = MERINA_DEFAULT_QUALITY

    buf_list: list[float] = []
    act_list: list[int] = []
    prob_list: list[np.ndarray] = []
    mu_list: list[np.ndarray] = []
    logvar_list: list[np.ndarray] = []
    latent_list: list[np.ndarray] = []
    rew_list: list[float] = []
    rebuf_list: list[float] = []
    done_list: list[bool] = []

    while True:
        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain, _ = (
            test_env.get_video_chunk(int(bit_rate))
        )

        if args_ns.log:
            log_br = np.log(bitrate_versions[bit_rate] / float(bitrate_versions[0]))
            log_last = np.log(bitrate_versions[last_bit_rate] / float(bitrate_versions[0]))
            reward = float(
                log_br - rebuff_p * rebuf - MERINA_SMOOTH_PENALTY * abs(log_br - log_last)
            )
        else:
            reward = float(
                bitrate_versions[bit_rate] / M_IN_K
                - rebuff_p * rebuf
                - MERINA_SMOOTH_PENALTY
                * abs(bitrate_versions[bit_rate] - bitrate_versions[last_bit_rate])
                / M_IN_K
            )
        last_bit_rate = int(bit_rate)

        state_policy = np.roll(state_policy, -1, axis=1)
        ob = np.roll(ob, -1, axis=1)
        safe_d = max(float(delay), 1e-6)
        state_policy[0, -1] = float(video_chunk_size) / safe_d / M_IN_K
        state_policy[1, -1] = float(buffer_size) / BUFFER_NORM_FACTOR
        state_policy[2, -1] = bitrate_versions[bit_rate] / float(np.max(bitrate_versions))
        state_policy[3, -1] = np.minimum(video_chunk_remain, total_chunk_num) / float(total_chunk_num)
        state_policy[4 : 4 + a_dim, -1] = np.asarray(next_video_chunk_sizes, dtype=np.float64) / M_IN_K / M_IN_K
        state_policy[10, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        ob[0, -1] = float(video_chunk_size) / safe_d / M_IN_K
        ob[1, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR

        ob_ = np.asarray([ob], dtype=np.float32).transpose(0, 2, 1)
        ob_t = torch.from_numpy(ob_).to(device_torch)
        state_ = np.asarray([state_policy], dtype=np.float32)
        state_t = torch.from_numpy(state_).to(device_torch)

        with torch.no_grad():
            mu, log_var = model_vae.encode(ob_t)
            latent = model_vae.get_latent(ob_t).detach()
            prob = model_actor.forward(state_t, latent).detach()
            if torch.isnan(prob).any() or torch.isinf(prob).any():
                prob = torch.full_like(prob, 1.0 / float(prob.shape[-1]))
            prob = prob / prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if args_ns.stocha:
            act = prob.multinomial(num_samples=1).detach()
            next_bit_rate = int(act.squeeze().cpu().numpy())
        else:
            next_bit_rate = int(torch.argmax(prob, dim=-1).squeeze().cpu().numpy())
        next_bit_rate = max(0, min(BITRATE_LEVELS - 1, next_bit_rate))

        buf_list.append(float(buffer_size))
        act_list.append(int(next_bit_rate))
        prob_list.append(prob.squeeze(0).cpu().numpy().astype(np.float32))
        mu_list.append(mu.squeeze(0).cpu().numpy().astype(np.float32))
        logvar_list.append(log_var.squeeze(0).cpu().numpy().astype(np.float32))
        latent_list.append(latent.squeeze(0).cpu().numpy().astype(np.float32))
        rew_list.append(reward)
        rebuf_list.append(float(rebuf))
        done_list.append(bool(end_of_video))

        bit_rate = next_bit_rate

        if end_of_video:
            break

    trajectory = build_merina_trajectory_tensordict(
        buffer_size=torch.as_tensor(buf_list, dtype=torch.float32),
        action=torch.as_tensor(act_list, dtype=torch.int64),
        action_prob=torch.as_tensor(np.stack(prob_list, axis=0), dtype=torch.float32),
        belief_mu=torch.as_tensor(np.stack(mu_list, axis=0), dtype=torch.float32),
        belief_logvar=torch.as_tensor(np.stack(logvar_list, axis=0), dtype=torch.float32),
        belief_latent=torch.as_tensor(np.stack(latent_list, axis=0), dtype=torch.float32),
        reward=torch.as_tensor(rew_list, dtype=torch.float32),
        rebuffer=torch.as_tensor(rebuf_list, dtype=torch.float32),
        done=torch.as_tensor(done_list, dtype=torch.bool),
    )
    meta = {
        "trace_name": trace_name,
        "trace_index": int(trace_index),
        "video": video_key,
        "num_steps": int(trajectory.batch_size[0]),
        "latent_dim": int(mu_list[0].shape[-1]) if mu_list else int(args_ns.latent_dim),
        "policy": "stochastic" if args_ns.stocha else "greedy",
        "qoe": "log" if args_ns.log else "linear",
        "random_mahimahi_start": bool(args_ns.random_mahimahi_start),
        "mahimahi_ptr": None if args_ns.random_mahimahi_start else int(mahimahi_ptr or 1),
        "seed": int(args_ns.seed),
    }
    return trajectory, meta


def main() -> int:
    p = argparse.ArgumentParser(
        description="Merina 决策轨迹 → 每 trace 一个 TensorDict .pt（buffer / action / VAE belief）"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("data", "traces", "train", "merina_trajectories"),
        help="输出目录；每条 trace 一个 .pt",
    )
    p.add_argument(
        "--trace-dir",
        type=str,
        default=DEFAULT_MERINA_TRACE_DIR,
        help="默认 cooked 训练轨迹目录",
    )
    p.add_argument(
        "--traces",
        nargs="*",
        default=None,
        help="config.cfg.trace_dirs 键；指定后忽略 --trace-dir",
    )
    p.add_argument(
        "--video",
        type=str,
        default="video1",
        help="config.cfg.video_size_dirs 键",
    )
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--policy-model", type=str, default=_DEFAULT_POLICY)
    p.add_argument("--vae-model", type=str, default=_DEFAULT_VAE)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--log", action="store_true", help="对数 QoE（需 log 训练权重）")
    p.add_argument("--stocha", action="store_true", help="随机策略；默认贪心")
    p.add_argument(
        "--random-mahimahi-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="每条 trace 随机 mahimahi 起点（与经验池采集一致）；--no-random-mahimahi-start 则用固定 ptr",
    )
    p.add_argument(
        "--fixed-order",
        action="store_true",
        help="与 baseline 一致：按 seed 生成固定 mahimahi 指针（仅 --no-random-mahimahi-start 时生效）",
    )
    p.add_argument("--max-traces", type=int, default=-1, help="最多采集条数；-1 为全部")
    p.add_argument("--start-index", type=int, default=0, help="从第几条 trace 开始")
    p.add_argument(
        "--cuda-id",
        type=int,
        default=None,
        help=">=0 用 GPU；<0 强制 CPU",
    )
    p.add_argument("--cpu", action="store_true", help="等价 --cuda-id -1")
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="可选：写入 JSON manifest（trace_index → 文件名）",
    )
    args_ns = p.parse_args()

    cid = args_ns.cuda_id
    if args_ns.cpu:
        cid = -1
    if cid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if int(cid) < 0 else str(int(cid))

    sys.path.insert(0, _MERINA_ROOT)
    import torch
    from algos.AC_net_v6 import Actor
    from algos.beta_vae_v6 import BetaVAE

    from baseline_special.utils.mahimahi_start import build_mahimahi_ptrs_by_trace_index
    from config import cfg

    if args_ns.video not in cfg.video_size_dirs:
        raise SystemExit(f"未知 video: {args_ns.video}")
    video_size_dir = cfg.video_size_dirs[args_ns.video]
    video_prefix = _merina_video_size_prefix(video_size_dir)
    for br in range(6):
        fp = video_prefix + str(br)
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"缺少 Merina 所需文件: {fp}")
    total_chunk_num = _count_video_chunks(video_size_dir)

    all_t, all_b, all_n, trace_src = _resolve_traces(args_ns)
    n_total = len(all_n)
    start = max(0, int(args_ns.start_index))
    end = n_total if int(args_ns.max_traces) < 0 else min(n_total, start + int(args_ns.max_traces))
    if start >= end:
        raise SystemExit(f"空区间: start_index={start}, max_traces={args_ns.max_traces}, 共 {n_total} 条")

    policy_path = os.path.abspath(args_ns.policy_model)
    vae_path = os.path.abspath(args_ns.vae_model)
    if not os.path.isfile(policy_path):
        raise SystemExit(f"未找到 policy: {policy_path}")
    if not os.path.isfile(vae_path):
        raise SystemExit(f"未找到 VAE: {vae_path}")

    use_cuda = torch.cuda.is_available() and (cid is None or int(cid) >= 0)
    device_torch = "cuda:0" if use_cuda else "cpu"
    map_loc = device_torch

    s_info, s_len, c_len = MERINA_S_INFO, MERINA_S_LEN, MERINA_C_LEN
    rebuff_p = MERINA_REBUF_LOG if args_ns.log else MERINA_REBUF_LIN
    latent_dim = int(args_ns.latent_dim)

    model_actor = Actor(6, latent_dim, s_info, s_len)
    model_vae = BetaVAE(in_channels=2, hist_dim=c_len, latent_dim=latent_dim)
    model_actor.load_state_dict(torch.load(policy_path, map_location=map_loc))
    model_vae.load_state_dict(torch.load(vae_path, map_location=map_loc))
    model_actor.eval()
    model_vae.eval()
    model_actor = model_actor.to(device_torch)
    model_vae = model_vae.to(device_torch)

    ptrs: list[int] | None = None
    if not args_ns.random_mahimahi_start:
        ptrs = build_mahimahi_ptrs_by_trace_index(
            all_b,
            seed=int(args_ns.seed),
            fixed=bool(args_ns.fixed_order),
        )

    out_root = os.path.abspath(args_ns.output_dir)
    if args_ns.traces:
        sub = "_".join(args_ns.traces)
    else:
        sub = os.path.basename(os.path.normpath(args_ns.trace_dir))
    output_dir = os.path.join(out_root, f"{sub}_{args_ns.video}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"trace 来源: {trace_src}（共 {n_total} 条，采集 [{start}, {end})）")
    print(f"Merina 设备: {device_torch}")
    print(f"输出目录: {output_dir}")

    manifest: dict[str, object] = {
        "format": "merina_trajectory_tensordict_v1",
        "trace_source": trace_src,
        "video": args_ns.video,
        "files": {},
    }

    saved = 0
    for idx in range(start, end):
        ptr_i = None if ptrs is None else ptrs[idx]
        trajectory, meta = rollout_one_trace(
            cooked_time=all_t[idx],
            cooked_bw=all_b[idx],
            trace_name=all_n[idx],
            trace_index=idx,
            video_key=args_ns.video,
            video_prefix=video_prefix,
            total_chunk_num=total_chunk_num,
            model_actor=model_actor,
            model_vae=model_vae,
            device_torch=device_torch,
            args_ns=args_ns,
            rebuff_p=rebuff_p,
            mahimahi_ptr=ptr_i,
        )
        out_path = make_trace_pt_path(output_dir, idx, all_n[idx])
        save_merina_trajectory_pt(out_path, trajectory, meta)
        manifest["files"][str(idx)] = os.path.basename(out_path)
        saved += 1
        if saved <= 3 or saved == end - start:
            print(
                f"  [{idx}] {all_n[idx]} → {os.path.basename(out_path)} "
                f"(T={meta['num_steps']}, latent={meta['latent_dim']})"
            )
        elif saved == 4:
            print("  ...")

    if args_ns.manifest:
        manifest_path = os.path.abspath(args_ns.manifest)
    else:
        manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print(f"完成: {saved} 个 .pt → {output_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
