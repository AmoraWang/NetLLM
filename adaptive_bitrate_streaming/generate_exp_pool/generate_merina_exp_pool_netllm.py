#!/usr/bin/env python3
"""
使用 Merina 预训练策略（Actor + BetaVAE）在 NetLLM 带宽 trace 与 video_size_* 上采集轨迹；
**状态**与 ``generate_exp_pool.py`` 在 ``abr_llm_version=v2``（默认）时一致：
numpy ``float32``、shape **(6, 6)**（Pensieve / NetLLM 行序），供 ``run_abr.py`` 默认 **ABRLLM_v2** 使用。

默认带宽目录：``data/traces/train/fcc_hsdpa_cooked_traces``（可用 ``--trace-dir`` / ``--traces`` 覆盖，逻辑同 ``generate_merina_exp_pool_v3.py``）。

**奖励**与 ``generate_exp_pool.collect_experience`` 中 Pensieve 分支一致（``VIDEO_BIT_RATE``、
``REBUF_PENALTY``、``SMOOTH_PENALTY``），便于与 GENET/MPC 经验池混训或对齐尺度。

默认写入 ``teacher_logits``（Merina Actor 的 6 维 **Softmax 概率**），供 ``run_abr.py --loss-type ce_kl``
使用；训练时请加 ``--teacher-is-prob``。若加 ``--teacher-as-log-prob`` 则存 ``log(prob)`` 作伪 logits。

带宽合并、Merina 环境随机起点、CUDA 回退与 ``generate_merina_exp_pool_v3.py`` 对齐；
参数风格可共用 ``--traces`` / ``--videos`` / ``--cpu`` 等。

示例：
  cd adaptive_bitrate_streaming
  python generate_exp_pool/generate_merina_exp_pool_netllm.py \\
      --traces fcc-train --videos video1 \\
      --output artifacts/exp_pools/merina_fcc-train_video1_logits.pkl --cpu
  python run/run_abr.py --adapt --loss-type ce_kl --teacher-is-prob \\
      --exp-pool-path artifacts/exp_pools/merina_fcc-train_video1_logits.pkl ...
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)

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
    _MERINA_ROOT,
    _count_video_chunks,
    _merina_video_size_prefix,
    load_traces_from_dir,
    merge_trace_corpora,
)

_DEFAULT_POLICY = os.path.join(_MERINA_ROOT, "models", "pretrain_policy_lin.model")
_DEFAULT_VAE = os.path.join(_MERINA_ROOT, "models", "pretrain_vae_lin.model")


def _apply_state_pool_step_pensieve(
    state_pool: np.ndarray,
    bit_rate: int,
    delay: float,
    buffer_size: float,
    video_chunk_size: float,
    next_video_chunk_sizes,
    video_chunk_remain: float,
) -> None:
    """与 ``generate_exp_pool.collect_experience`` 中 v2 分支一致（6x6）。"""
    from baseline_special.utils.constants import (
        BITRATE_LEVELS,
        BUFFER_NORM_FACTOR,
        CHUNK_TIL_VIDEO_END_CAP,
        M_IN_K,
        MAX_VIDEO_BIT_RATE,
        VIDEO_BIT_RATE,
    )

    state_pool[:] = np.roll(state_pool, -1, axis=1)
    d = max(float(delay), 1e-6)
    state_pool[0, -1] = VIDEO_BIT_RATE[int(bit_rate)] / float(MAX_VIDEO_BIT_RATE)
    state_pool[1, -1] = buffer_size / BUFFER_NORM_FACTOR
    state_pool[2, -1] = float(video_chunk_size) / d / M_IN_K
    state_pool[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
    state_pool[4, :BITRATE_LEVELS] = np.asarray(next_video_chunk_sizes, dtype=np.float64) / M_IN_K / M_IN_K
    state_pool[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)


def _actor_prob_to_teacher(prob, *, as_log_prob: bool) -> np.ndarray:
    """Merina Actor 输出 Softmax 概率 → 写入经验池的 teacher 向量 (6,)。"""
    import torch

    p = prob.detach().float().reshape(-1)
    if p.numel() != 6:
        raise ValueError(f"期望 6 档码率概率，得到 shape {tuple(prob.shape)}")
    if torch.isnan(p).any() or torch.isinf(p).any():
        p = torch.full_like(p, 1.0 / 6.0)
    p = p / p.sum().clamp(min=1e-8)
    if as_log_prob:
        arr = torch.log(p.clamp(min=1e-6)).cpu().numpy().astype(np.float32)
    else:
        arr = p.cpu().numpy().astype(np.float32)
    return arr


def collect_one_video(
    all_cooked_time: list,
    all_cooked_bw: list,
    all_file_names: list,
    video_size_dir: str,
    model_actor,
    model_vae,
    device_torch: str,
    args_ns: argparse.Namespace,
    rebuff_p_merina: float,
) -> tuple[list, list, list, list, list]:
    """Merina 策略 + NetLLM (6,6) 状态 + NetLLM 式标量 reward + Merina teacher 分布。"""
    sys.path.insert(0, _MERINA_ROOT)
    import envs.fixed_env_log as merina_env_mod  # noqa: WPS433

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
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain, curr = super().get_video_chunk(quality)
            if end_of_video:
                self._reroll_mahimahi_start()
            return delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain, curr

    from baseline_special.utils.constants import (
        BITRATE_LEVELS,
        BUFFER_NORM_FACTOR,
        M_IN_K,
        REBUF_PENALTY,
        SMOOTH_PENALTY,
        VIDEO_BIT_RATE,
    )

    video_prefix = _merina_video_size_prefix(video_size_dir)
    for br in range(6):
        p = video_prefix + str(br)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少 Merina 所需文件: {p}")

    total_chunk_num = _count_video_chunks(video_size_dir)
    np.random.seed(int(args_ns.seed) % (2**32))

    test_env = RandomStartMerinaEnv(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        video_size_file=video_prefix,
        random_seed=int(args_ns.seed) % (2**32),
    )
    test_env.set_env_info(
        MERINA_S_INFO,
        MERINA_S_LEN,
        MERINA_C_LEN,
        total_chunk_num,
        MERINA_VIDEO_BIT_RATE,
        1,
        rebuff_p_merina,
        MERINA_SMOOTH_PENALTY,
        0,
    )
    _, _, c_len, _, bitrate_versions, _, _ = test_env.get_env_info()
    a_dim = len(bitrate_versions)

    from baseline_special.utils.constants import S_INFO, S_LEN

    state_pool = np.zeros((S_INFO, S_LEN), dtype=np.float32)
    state_policy = np.zeros((MERINA_S_INFO, MERINA_S_LEN), dtype=np.float32)
    ob = np.zeros((2, c_len), dtype=np.float32)

    bit_rate = MERINA_DEFAULT_QUALITY
    last_bit_rate = MERINA_DEFAULT_QUALITY

    total_s: list[np.ndarray] = []
    total_a: list[int] = []
    total_r: list[float] = []
    total_d: list[bool] = []
    total_teacher: list[np.ndarray] = []

    ep_s: list[np.ndarray] = []
    ep_a: list[int] = []
    ep_r: list[float] = []
    ep_d: list[bool] = []
    ep_teacher: list[np.ndarray] = []

    import torch

    pending_teacher: np.ndarray | None = None

    for _video_idx in range(len(all_file_names)):
        pending_teacher = None
        while True:
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain, _ = test_env.get_video_chunk(int(bit_rate))

            reward = float(
                VIDEO_BIT_RATE[int(bit_rate)] / M_IN_K
                - REBUF_PENALTY * rebuf
                - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[int(bit_rate)] - VIDEO_BIT_RATE[int(last_bit_rate)]) / M_IN_K
            )

            last_bit_rate = int(bit_rate)

            if pending_teacher is not None:
                ep_s.append(np.copy(state_pool))
                ep_a.append(int(bit_rate))
                ep_r.append(reward)
                ep_d.append(bool(end_of_video))
                ep_teacher.append(pending_teacher.copy())

            _apply_state_pool_step_pensieve(
                state_pool,
                int(bit_rate),
                float(delay),
                float(buffer_size),
                float(video_chunk_size),
                next_video_chunk_sizes,
                float(video_chunk_remain),
            )

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
                latent = model_vae.get_latent(ob_t).detach()
                prob = model_actor.forward(state_t, latent).detach()
                pending_teacher = _actor_prob_to_teacher(
                    prob.squeeze(0),
                    as_log_prob=bool(args_ns.teacher_as_log_prob),
                )

            prob_t = torch.from_numpy(pending_teacher).to(device_torch).float().reshape(1, -1)
            if args_ns.teacher_as_log_prob:
                prob_t = torch.softmax(prob_t, dim=-1)
            if args_ns.stocha:
                act = prob_t.multinomial(num_samples=1).detach()
                bit_rate = int(act.squeeze().cpu().numpy())
            else:
                bit_rate = int(torch.argmax(prob_t, dim=-1).squeeze().cpu().numpy())
            bit_rate = max(0, min(BITRATE_LEVELS - 1, bit_rate))

            if end_of_video:
                if len(ep_s) > 1:
                    total_s.extend(ep_s[1:])
                    total_a.extend(ep_a[1:])
                    total_r.extend(ep_r[1:])
                    total_d.extend(ep_d[1:])
                    total_teacher.extend(ep_teacher[1:])
                ep_s.clear()
                ep_a.clear()
                ep_r.clear()
                ep_d.clear()
                ep_teacher.clear()

                state_pool.fill(0.0)
                state_policy.fill(0.0)
                ob.fill(0.0)
                bit_rate = MERINA_DEFAULT_QUALITY
                last_bit_rate = MERINA_DEFAULT_QUALITY
                pending_teacher = None

                if _video_idx + 1 >= len(all_file_names):
                    return total_s, total_a, total_r, total_d, total_teacher
                break

    return total_s, total_a, total_r, total_d, total_teacher


def main() -> int:
    p = argparse.ArgumentParser(description="Merina 决策 + NetLLM Pensieve 状态 (6,6) 经验池")
    p.add_argument(
        "--output",
        type=str,
        default=os.path.join("artifacts", "exp_pools", "merina_netllm_6x6_merged.pkl"),
        help="输出 ExperiencePool 的 .pkl",
    )
    p.add_argument(
        "--trace-dir",
        type=str,
        default=DEFAULT_MERINA_TRACE_DIR,
        help="默认：data/traces/train/fcc_hsdpa_cooked_traces",
    )
    p.add_argument(
        "--traces",
        nargs="*",
        default=None,
        help="若提供则按 config.cfg.trace_dirs 多键合并并忽略 --trace-dir",
    )
    p.add_argument("--videos", nargs="+", default=["video1"], help="config.cfg.video_size_dirs 中的键，默认仅 video1")
    p.add_argument("--seed", type=int, default=100003)
    p.add_argument("--policy-model", type=str, default=_DEFAULT_POLICY)
    p.add_argument("--vae-model", type=str, default=_DEFAULT_VAE)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument(
        "--merina-log-reward",
        action="store_true",
        help="仅影响 Merina 环境内部 rebuff_p（与 log 权重一致）；池内 reward 仍为 NetLLM 线性标量",
    )
    p.add_argument("--stocha", action="store_true")
    p.add_argument(
        "--teacher-as-log-prob",
        action="store_true",
        help="teacher_logits 存 log(prob)（训练时不加 --teacher-is-prob）；默认存概率",
    )
    p.add_argument("--cuda-id", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
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

    from config import cfg
    from plm_special.data.exp_pool import ExperiencePool

    for vk in args_ns.videos:
        if vk not in cfg.video_size_dirs:
            raise SystemExit(f"未知 video: {vk}")

    if args_ns.traces:
        all_t, all_b, all_n = merge_trace_corpora(list(args_ns.traces))
        trace_src = f"config 键 {list(args_ns.traces)}"
    else:
        all_t, all_b, all_n = load_traces_from_dir(args_ns.trace_dir)
        trace_src = args_ns.trace_dir
    print(f"合并 trace 条数: {len(all_n)}（来自 {trace_src}）")

    policy_path = os.path.abspath(args_ns.policy_model)
    vae_path = os.path.abspath(args_ns.vae_model)
    if not os.path.isfile(policy_path):
        raise SystemExit(f"未找到 policy: {policy_path}")
    if not os.path.isfile(vae_path):
        raise SystemExit(f"未找到 VAE: {vae_path}")

    use_cuda = torch.cuda.is_available() and (cid is None or int(cid) >= 0)
    device_torch = "cuda:0" if use_cuda else "cpu"

    s_info, s_len, c_len = MERINA_S_INFO, MERINA_S_LEN, MERINA_C_LEN
    rebuff_p_merina = MERINA_REBUF_LOG if args_ns.merina_log_reward else MERINA_REBUF_LIN

    model_actor = Actor(6, args_ns.latent_dim, s_info, s_len)
    model_vae = BetaVAE(in_channels=2, hist_dim=c_len, latent_dim=args_ns.latent_dim)
    model_actor.load_state_dict(torch.load(policy_path, map_location=device_torch))
    model_vae.load_state_dict(torch.load(vae_path, map_location=device_torch))
    model_actor.eval()
    model_vae.eval()
    model_actor = model_actor.to(device_torch)
    model_vae = model_vae.to(device_torch)

    if use_cuda:
        try:
            ob_sm = torch.zeros(1, c_len, 2, device=device_torch, dtype=torch.float32)
            st_sm = torch.zeros(1, s_info, s_len, device=device_torch, dtype=torch.float32)
            with torch.no_grad():
                z = model_vae.get_latent(ob_sm)
                _ = model_actor(st_sm, z)
        except RuntimeError as e:
            print(f"[warn] CUDA 前向失败，改用 CPU: {e}")
            device_torch = "cpu"
            model_actor = model_actor.cpu()
            model_vae = model_vae.cpu()

    pool = ExperiencePool()
    for vk in args_ns.videos:
        video_size_dir = cfg.video_size_dirs[vk]
        print(f"--- 采集 video={vk} ---")
        ts, ta, tr, td, tt = collect_one_video(
            all_t,
            all_b,
            all_n,
            video_size_dir,
            model_actor,
            model_vae,
            device_torch,
            args_ns,
            rebuff_p_merina,
        )
        print(f"    transitions: {len(ts)}")
        for i in range(len(ts)):
            pool.add(
                state=ts[i],
                action=ta[i],
                reward=tr[i],
                done=td[i],
                teacher_logits=tt[i],
            )

    out_path = os.path.abspath(args_ns.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pool, f)

    if len(pool.states):
        s0 = np.asarray(pool.states[0])
        print(f"首条 state shape: {s0.shape} dtype={s0.dtype}")
    if getattr(pool, "has_teacher_logits", False):
        t0 = np.asarray(pool.teacher_logits[0], dtype=np.float32).reshape(-1)
        print(f"teacher_logits: {len(pool.teacher_logits)} 条, 首条 shape={t0.shape}, sum={t0.sum():.4f}")
        if args_ns.teacher_as_log_prob:
            print("  训练: run_abr.py --loss-type ce_kl（勿加 --teacher-is-prob）")
        else:
            print("  训练: run_abr.py --loss-type ce_kl --teacher-is-prob")
    print(f"总 transition 数: {len(pool)}")
    print(f"已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
