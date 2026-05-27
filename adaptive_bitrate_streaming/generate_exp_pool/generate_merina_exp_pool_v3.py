#!/usr/bin/env python3
"""
使用 Merina 预训练策略（Actor + BetaVAE）在 NetLLM 带宽 trace 与 video_size_* 上采集轨迹，
状态布局与 ``generate_exp_pool.py --abr-llm-version v3`` 一致（numpy float32, shape (11, 6)），
供 ``run_abr.py --abr-llm-version v3`` 训练 ABRLLM_v3。

默认带宽：``adaptive_bitrate_streaming/data/traces/train/fcc_hsdpa_cooked_traces`` 下全部轨迹
（``load_traces``）；可用 ``--trace-dir`` 改目录；若指定 ``--traces`` 则改为 ``config.cfg.trace_dirs`` 多键合并。

默认视频：仅 ``video1``（``--videos`` 默认为 ``video1``；可传多个键扩展）

每个视频：在「合并后的全部 trace」上顺序跑完一条 episode / trace，再将各 episode 的
第 0 步丢弃（与 ``generate_exp_pool.collect_experience`` 的 ``states[1:]`` 对齐），
最后把所有 transition 顺序写入**同一个** ``ExperiencePool`` 并 ``pickle.dump``。

示例：
  cd adaptive_bitrate_streaming
  python generate_exp_pool/generate_merina_exp_pool_v3.py --output artifacts/exp_pools/exp_pool.pkl --cpu
  # 或：--cuda-id -1  （当 PyTorch 不支持当前 GPU 架构、CUDA 前向异常或输出含 NaN 时推荐）
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
_MERINA_ROOT = os.path.join(_ABR_ROOT, "merina")
# 默认与仓库内 HSDPA 训练轨迹目录一致（相对 adaptive_bitrate_streaming 根）
DEFAULT_MERINA_TRACE_DIR = os.path.join(_ABR_ROOT, "data", "traces", "train", "fcc_hsdpa_cooked_traces")
_DEFAULT_POLICY = os.path.join(_MERINA_ROOT, "models", "pretrain_policy_lin.model")
_DEFAULT_VAE = os.path.join(_MERINA_ROOT, "models", "pretrain_vae_lin.model")

MERINA_S_INFO = 11
MERINA_S_LEN = 2
MERINA_C_LEN = 8
MERINA_VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
MERINA_REBUF_LIN = 4.3
MERINA_REBUF_LOG = 2.66
MERINA_SMOOTH_PENALTY = 1
MERINA_DEFAULT_QUALITY = 1

M_IN_K = 1000.0


def _count_video_chunks(video_size_dir: str) -> int:
    path0 = os.path.join(video_size_dir, "video_size_0")
    with open(path0, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _merina_video_size_prefix(video_size_dir: str) -> str:
    d = os.path.abspath(video_size_dir)
    if not d.endswith(os.sep):
        d = d + os.sep
    return os.path.join(d, "video_size_")


def merge_trace_corpora(trace_keys: list[str]) -> tuple[list, list, list]:
    """按顺序拼接多个 trace 目录下的 cooked 序列与文件名（带前缀避免重名）。"""
    from baseline_special.utils.utils import load_traces
    from config import cfg

    all_t: list = []
    all_b: list = []
    all_n: list = []
    for key in trace_keys:
        if key not in cfg.trace_dirs:
            raise KeyError(f"未知 trace 键: {key}，请在 config.cfg.trace_dirs 中配置")
        trace_dir = cfg.trace_dirs[key]
        if not os.path.isdir(trace_dir):
            raise FileNotFoundError(f"trace 目录不存在: {trace_dir}")
        t, b, n, _ = load_traces(trace_dir)
        for i in range(len(n)):
            all_t.append(t[i])
            all_b.append(b[i])
            all_n.append(f"{key}__{n[i]}")
    if not all_n:
        raise RuntimeError("合并后 trace 条数为 0，请检查目录与文件")
    return all_t, all_b, all_n


def load_traces_from_dir(trace_dir: str, name_prefix: str = "fcc_hsdpa") -> tuple[list, list, list]:
    """从单个目录加载全部 cooked 轨迹（与 ``load_traces`` 行为一致），文件名加前缀避免与其它来源合并时冲突。"""
    from baseline_special.utils.utils import load_traces

    trace_dir = os.path.abspath(trace_dir)
    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(f"trace 目录不存在: {trace_dir}")
    t, b, n, _ = load_traces(trace_dir)
    if not n:
        raise RuntimeError(f"目录内无有效 trace 文件: {trace_dir}")
    pref = [f"{name_prefix}__{x}" for x in n]
    return list(t), list(b), pref


def _apply_state_pool_step_v3(
    state_pool: np.ndarray,
    bit_rate: int,
    delay: float,
    buffer_size: float,
    video_chunk_size: float,
    next_video_chunk_sizes,
    video_chunk_remain: float,
) -> None:
    """与 ``generate_exp_pool.py`` 中 ``abr_llm_version=v3`` 的状态更新一致（11x6）。"""
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
    state_pool[0, -1] = float(video_chunk_size) / d / M_IN_K
    state_pool[1, -1] = buffer_size / BUFFER_NORM_FACTOR
    state_pool[2, -1] = VIDEO_BIT_RATE[int(bit_rate)] / float(MAX_VIDEO_BIT_RATE)
    state_pool[3, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
    sz = np.asarray(next_video_chunk_sizes, dtype=np.float64) / M_IN_K / M_IN_K
    for i in range(BITRATE_LEVELS):
        state_pool[4 + i, -1] = sz[i]
    state_pool[10, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR


def collect_one_video(
    all_cooked_time: list,
    all_cooked_bw: list,
    all_file_names: list,
    video_size_dir: str,
    model_actor,
    model_vae,
    device_torch: str,
    args_ns: argparse.Namespace,
    rebuff_p: float,
) -> tuple[list, list, list, list]:
    """在已合并的 trace 列表与指定视频目录上采集，返回展平的四元组列表。"""
    sys.path.insert(0, _MERINA_ROOT)
    import envs.fixed_env_log as merina_env_mod  # noqa: WPS433

    class RandomStartMerinaEnv(merina_env_mod.Environment):
        """每条 trace 随机 mahimahi 起点；勿修改 mahimahi_start_ptr（见 run_merina_baseline_test 说明）。"""

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
        rebuff_p,
        MERINA_SMOOTH_PENALTY,
        0,
    )
    _, _, c_len, _, bitrate_versions, _, _ = test_env.get_env_info()
    a_dim = len(bitrate_versions)

    from baseline_special.utils.constants import ABRLLM_V3_S_INFO, ABRLLM_V3_S_LEN, BUFFER_NORM_FACTOR, BITRATE_LEVELS

    state_pool = np.zeros((ABRLLM_V3_S_INFO, ABRLLM_V3_S_LEN), dtype=np.float32)
    state_policy = np.zeros((MERINA_S_INFO, MERINA_S_LEN), dtype=np.float32)
    vae_in_channels = 2
    ob = np.zeros((vae_in_channels, c_len), dtype=np.float32)

    bit_rate = MERINA_DEFAULT_QUALITY
    last_bit_rate = MERINA_DEFAULT_QUALITY

    total_s: list[np.ndarray] = []
    total_a: list[int] = []
    total_r: list[float] = []
    total_d: list[bool] = []

    ep_s: list[np.ndarray] = []
    ep_a: list[int] = []
    ep_r: list[float] = []
    ep_d: list[bool] = []

    import torch

    for _video_idx in range(len(all_file_names)):
        while True:
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain, _ = test_env.get_video_chunk(int(bit_rate))

            if args_ns.log:
                log_br = np.log(bitrate_versions[bit_rate] / float(bitrate_versions[0]))
                log_last = np.log(bitrate_versions[last_bit_rate] / float(bitrate_versions[0]))
                reward = float(
                    log_br - rebuff_p * rebuf - MERINA_SMOOTH_PENALTY * np.abs(log_br - log_last)
                )
            else:
                reward = float(
                    bitrate_versions[bit_rate] / M_IN_K
                    - rebuff_p * rebuf
                    - MERINA_SMOOTH_PENALTY
                    * np.abs(bitrate_versions[bit_rate] - bitrate_versions[last_bit_rate])
                    / M_IN_K
                )

            last_bit_rate = int(bit_rate)

            ep_s.append(np.copy(state_pool))
            ep_a.append(int(bit_rate))
            ep_r.append(reward)
            ep_d.append(bool(end_of_video))

            _apply_state_pool_step_v3(
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
                if torch.isnan(prob).any() or torch.isinf(prob).any():
                    prob = torch.full_like(prob, 1.0 / float(prob.shape[-1]))
                prob = prob / prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            if args_ns.stocha:
                act = prob.multinomial(num_samples=1).detach()
                bit_rate = int(act.squeeze().cpu().numpy())
            else:
                bit_rate = int(torch.argmax(prob, dim=-1).squeeze().cpu().numpy())

            bit_rate = max(0, min(BITRATE_LEVELS - 1, bit_rate))

            if end_of_video:
                if len(ep_s) > 1:
                    total_s.extend(ep_s[1:])
                    total_a.extend(ep_a[1:])
                    total_r.extend(ep_r[1:])
                    total_d.extend(ep_d[1:])
                ep_s.clear()
                ep_a.clear()
                ep_r.clear()
                ep_d.clear()

                state_pool.fill(0.0)
                state_policy.fill(0.0)
                ob.fill(0.0)
                bit_rate = MERINA_DEFAULT_QUALITY
                last_bit_rate = MERINA_DEFAULT_QUALITY

                if _video_idx + 1 >= len(all_file_names):
                    return total_s, total_a, total_r, total_d
                break

    return total_s, total_a, total_r, total_d


def main() -> int:
    p = argparse.ArgumentParser(description="Merina 策略采集 ABRLLM_v3 (11,6) 经验池并合并为多 trace × 多视频单 pkl")
    p.add_argument(
        "--output",
        type=str,
        default=os.path.join("artifacts", "exp_pools", "merina_abrllm_v3_merged.pkl"),
        help="输出 ExperiencePool 的 .pkl 路径",
    )
    p.add_argument(
        "--trace-dir",
        type=str,
        default=DEFAULT_MERINA_TRACE_DIR,
        help="默认：data/traces/train/fcc_hsdpa_cooked_traces；该目录下全部带宽文件参与采集",
    )
    p.add_argument(
        "--traces",
        nargs="*",
        default=None,
        help="若提供（一个或多个 config 键），则按 config.cfg.trace_dirs 合并多目录并忽略 --trace-dir",
    )
    p.add_argument(
        "--videos",
        nargs="+",
        default=["video1"],
        help="config.cfg.video_size_dirs 中的键，默认仅 video1",
    )
    p.add_argument("--seed", type=int, default=100003)
    p.add_argument("--policy-model", type=str, default=_DEFAULT_POLICY)
    p.add_argument("--vae-model", type=str, default=_DEFAULT_VAE)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--log", action="store_true", help="对数 QoE（需 log 训练权重）")
    p.add_argument("--stocha", action="store_true", help="随机策略 multinomial；默认贪心")
    p.add_argument(
        "--cuda-id",
        type=int,
        default=None,
        help=">=0：设置 CUDA_VISIBLE_DEVICES；<0：强制 CPU（旧 PyTorch 遇 sm_120 等新卡无算力时请用 -1）。默认：若 CUDA 可用则用 GPU",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="等价于 --cuda-id -1：Merina 前向强制在 CPU 上运行",
    )
    args_ns = p.parse_args()

    cid = args_ns.cuda_id
    if getattr(args_ns, "cpu", False):
        cid = -1
    if cid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if int(cid) < 0 else str(int(cid))

    sys.path.insert(0, _MERINA_ROOT)
    import torch
    from algos.beta_vae_v6 import BetaVAE
    from algos.AC_net_v6 import Actor

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
    rebuff_p = MERINA_REBUF_LOG if args_ns.log else MERINA_REBUF_LIN

    map_loc = device_torch
    model_actor = Actor(6, args_ns.latent_dim, s_info, s_len)
    model_vae = BetaVAE(in_channels=2, hist_dim=c_len, latent_dim=args_ns.latent_dim)
    model_actor.load_state_dict(torch.load(policy_path, map_location=map_loc))
    model_vae.load_state_dict(torch.load(vae_path, map_location=map_loc))
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
            print(f"[warn] CUDA 上前向失败（常见于 GPU 算力与当前 PyTorch 不匹配），改用 CPU: {e}")
            device_torch = "cpu"
            model_actor = model_actor.cpu()
            model_vae = model_vae.cpu()

    pool = ExperiencePool()

    for vk in args_ns.videos:
        video_size_dir = cfg.video_size_dirs[vk]
        print(f"--- 采集 video={vk} dir={video_size_dir} ---")
        ts, ta, tr, td = collect_one_video(
            all_t,
            all_b,
            all_n,
            video_size_dir,
            model_actor,
            model_vae,
            device_torch,
            args_ns,
            rebuff_p,
        )
        print(f"    transitions: {len(ts)}")
        for i in range(len(ts)):
            pool.add(state=ts[i], action=ta[i], reward=tr[i], done=td[i])

    out_path = os.path.abspath(args_ns.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pool, f)

    if len(pool.states):
        s0 = np.asarray(pool.states[0])
        print(f"首条 state shape: {s0.shape} dtype={s0.dtype}")
    print(f"总 transition 数: {len(pool)}")
    print(f"已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
