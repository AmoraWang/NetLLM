#!/usr/bin/env python3
"""
将 ``data/traces/train/eval_trajectories/*.pt``（TensorDict rollout）转为
``plm_special.data.exp_pool.ExperiencePool`` 并保存为 pickle。

字段映射（与合成轨迹格式一致）：
  - states  <- observation；默认输出 NetLLM/Pensieve (6, H)，与 evaluate.py / ABRLLM v1–v2 行序一致
  - 源 observation 常见为 Merina (11, H) 或紧凑 6 行布局（0 吞吐、1 缓冲、2 上一码率、3 剩余、4 下一分片、5 时延）
  - actions <- action 的 6 维 one-hot → argmax 得到码率档位 0–5
  - teacher_logits <- logits (B, 48, 6)，与 action 同 chunk 对齐（供 run_abr --loss-type ce_kl）
  - rewards <- episode_reward
  - dones   <- done

张量布局约定（batch_size = [32, 48]）：
  - 维 0（32）：并行 batch / 子轨迹条数
  - 维 1（48）：源数据中每个 episode 的 chunk 数；**仅保留前 47 块**（索引 0–46），舍弃第 48 块（索引 47）
  - 第 47 块（1-based，索引 46）的 ``done`` 置为 ``True``（与评测时丢弃首尾块、只用中间 47 块一致）

示例：
  cd adaptive_bitrate_streaming
  # 默认 pensieve (6,6)，供 ABRLLM v1/v2 / run_abr
  python run/convert_eval_trajectories_to_exp_pool.py \\
      --input-dir data/traces/train/eval_trajectories \\
      --output artifacts/exp_pools/eval_trajectories_pensieve_6x6.pkl

  # Merina (11,6) 直出（legacy；v3 训练请用 pensieve 6×6）
  python run/convert_eval_trajectories_to_exp_pool.py \\
      --state-format merina \\
      --output artifacts/exp_pools/eval_trajectories_merina_11x6.pkl \\
      --batch-index 0

  # 每个 .pt 只取 batch 维第 0 条（文件名含 episode 且仅一条轨迹时）
  python run/convert_eval_trajectories_to_exp_pool.py \\
      --input-dir data/traces/train/eval_trajectories \\
      --output artifacts/exp_pools/eval_trajectories_ep0_only.pkl \\
      --batch-index 0
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)

import numpy as np
import torch

from plm_special.data.exp_pool import ExperiencePool

BITRATE_LEVELS = 6
MERINA_STATE_ROWS = 11
PENSIEVE_STATE_ROWS = 6
DEFAULT_STATE_COLS = 6
SOURCE_HISTORY_LEN = 8  # 合成 .pt 中 observation 的历史维长度
SOURCE_CHUNKS_PER_EPISODE = 48  # .pt 中每条 episode 的 chunk 维长度
CHUNKS_TO_KEEP = 47  # 仅保留前 47 块，舍弃最后一块（索引 47）
LAST_KEPT_CHUNK_INDEX = CHUNKS_TO_KEEP - 1  # 0-based：46，即第 47 块
STATE_FORMAT_MERINA = "merina"
STATE_FORMAT_PENSIEVE = "pensieve"

# NetLLM / Pensieve (6, H) 行序（与 plm_special/evaluate.py、ABRLLM_v2 StateEncoder 一致）
# 0 上一码率 | 1 缓冲 | 2 吞吐 | 3 下载时延 | 4 六档下一分片大小 | 5 剩余 chunk
PENSIEVE_ROW_LAST_BITRATE = 0
PENSIEVE_ROW_BUFFER = 1
PENSIEVE_ROW_THROUGHPUT = 2
PENSIEVE_ROW_DOWNLOAD_TIME = 3
PENSIEVE_ROW_NEXT_CHUNK = 4
PENSIEVE_ROW_REMAIN = 5

# 合成 .pt 中常见的 6 行紧凑布局（与 Merina 11 行前 6 类语义相同，但行序不同）
COMPACT6_ROW_THROUGHPUT = 0
COMPACT6_ROW_BUFFER = 1
COMPACT6_ROW_LAST_BITRATE = 2
COMPACT6_ROW_REMAIN = 3
COMPACT6_ROW_NEXT_CHUNK = 4
COMPACT6_ROW_DOWNLOAD_TIME = 5

# 紧凑/Merina 源布局 → NetLLM Pensieve 行下标（与 evaluate.py 一致）
# 源: 0 吞吐, 1 缓冲, 2 上一码率, 3 剩余, 4 下一分片, 5 时延
PENSIEVE_ROW_ORDER_FROM_COMPACT6 = (
    COMPACT6_ROW_LAST_BITRATE,
    COMPACT6_ROW_BUFFER,
    COMPACT6_ROW_THROUGHPUT,
    COMPACT6_ROW_DOWNLOAD_TIME,
    COMPACT6_ROW_NEXT_CHUNK,
    COMPACT6_ROW_REMAIN,
)


def _one_hot_to_bitrate_index(action_vec: torch.Tensor) -> int:
    """(6,) one-hot → 码率索引 0..5。"""
    a = action_vec.detach().cpu().float().flatten()
    if a.numel() != BITRATE_LEVELS:
        raise ValueError(f"action 最后一维应为 {BITRATE_LEVELS}，得到 shape={tuple(action_vec.shape)}")
    idx = int(a.argmax().item())
    if idx < 0 or idx >= BITRATE_LEVELS:
        raise ValueError(f"action argmax 越界: {idx}")
    return idx


def _extract_logits(logits_vec: torch.Tensor) -> np.ndarray:
    """(6,) 或 (1,6) 教师 logits → float32 (6,)。"""
    v = logits_vec.detach().cpu().float().flatten()
    if v.numel() != BITRATE_LEVELS:
        raise ValueError(
            f"logits 最后一维应为 {BITRATE_LEVELS}，得到 shape={tuple(logits_vec.shape)}"
        )
    return v.numpy().astype(np.float32)


def _slice_history(
    state: torch.Tensor,
    *,
    state_cols: int,
    history_slice: str,
) -> torch.Tensor:
    """从 2D observation 矩阵截取历史列。"""
    if state.ndim != 2:
        raise ValueError(
            f"observation 压扁后应为 2D，得到 shape={tuple(state.shape)}"
        )
    cols = int(state.shape[1])
    if cols == state_cols:
        return state
    if cols > state_cols:
        if history_slice == "last":
            return state[:, -state_cols:]
        if history_slice == "first":
            return state[:, :state_cols]
        raise ValueError(f"未知 history_slice={history_slice!r}，应为 last 或 first")
    raise ValueError(
        f"历史长度 {cols} 小于目标 {state_cols}；源 .pt 通常为 {SOURCE_HISTORY_LEN}"
    )


def _ensure_feature_major(state: torch.Tensor) -> torch.Tensor:
    """
    保证矩阵为 (特征行, 历史列) = (6|11, 6|8)。

    合成 .pt 里偶见 (历史, 特征) 如 (8, 11) / (8, 6)，需转置后再做行重排。
    """
    if state.ndim != 2:
        raise ValueError(f"observation 压扁后应为 2D，得到 shape={tuple(state.shape)}")
    r, c = int(state.shape[0]), int(state.shape[1])
    feature_rows = {PENSIEVE_STATE_ROWS, MERINA_STATE_ROWS}
    history_cols = {DEFAULT_STATE_COLS, SOURCE_HISTORY_LEN}
    if r in history_cols and c in feature_rows:
        return state.T
    if r in feature_rows and c in history_cols:
        return state
    if r in feature_rows and c in feature_rows:
        return state
    return state


def _obs_to_matrix(
    obs: torch.Tensor,
    *,
    state_cols: int,
    history_slice: str = "last",
) -> np.ndarray:
    """observation → (rows, state_cols) float32，保留源行数（6 或 11）。"""
    state = _ensure_feature_major(obs.detach().cpu().float().squeeze())
    state = _slice_history(
        state,
        state_cols=state_cols,
        history_slice=history_slice,
    )
    return state.numpy().astype(np.float32)


def _merina11_to_pensieve6(merina: np.ndarray) -> np.ndarray:
    """
    Merina (11, H) → NetLLM Pensieve (6, H)。

    Merina 行：0 吞吐、1 缓冲、2 上一码率、3 剩余块、4–9 下一分片六档、10 下载时间。
    """
    if merina.shape[0] != MERINA_STATE_ROWS:
        raise ValueError(f"Merina 状态行数应为 {MERINA_STATE_ROWS}，得到 {merina.shape[0]}")
    h = merina.shape[1]
    pensieve = np.zeros((PENSIEVE_STATE_ROWS, h), dtype=np.float32)
    pensieve[PENSIEVE_ROW_LAST_BITRATE] = merina[2]
    pensieve[PENSIEVE_ROW_BUFFER] = merina[1]
    pensieve[PENSIEVE_ROW_THROUGHPUT] = merina[0]
    pensieve[PENSIEVE_ROW_DOWNLOAD_TIME] = merina[10]
    pensieve[PENSIEVE_ROW_REMAIN] = merina[3]
    # 与 evaluate.py 一致：第 4 行 6 列存放六档下一分片大小（取 Merina 4–9 行当前列）
    pensieve[PENSIEVE_ROW_NEXT_CHUNK, :BITRATE_LEVELS] = merina[
        4 : 4 + BITRATE_LEVELS, -1
    ]
    return pensieve


def _compact6_to_pensieve6(compact: np.ndarray) -> np.ndarray:
    """
    合成轨迹 / Merina test_v5 的 6 行紧凑布局 → NetLLM Pensieve (6, H)。

    源行序：0 吞吐、1 缓冲、2 上一码率、3 剩余、4 下一分片（一行六列）、5 下载时延。
    目标行序：0 上一码率、1 缓冲、2 吞吐、3 下载时延、4 下一分片、5 剩余。
    """
    if compact.shape[0] != PENSIEVE_STATE_ROWS:
        raise ValueError(f"紧凑状态行数应为 {PENSIEVE_STATE_ROWS}，得到 {compact.shape[0]}")
    return compact[np.array(PENSIEVE_ROW_ORDER_FROM_COMPACT6, dtype=np.int64)].copy()


def reorder_state_to_netllm_pensieve(state: np.ndarray) -> np.ndarray:
    """将 (6, H) 紧凑布局或已存错序的经验池状态转为 NetLLM Pensieve 行序。"""
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[0] == MERINA_STATE_ROWS:
        return _merina11_to_pensieve6(arr)
    if arr.shape[0] == PENSIEVE_STATE_ROWS:
        return _compact6_to_pensieve6(arr)
    raise ValueError(f"状态行数应为 6 或 11，得到 {arr.shape[0]}")


_BITRATE_NORMS = np.array([300, 750, 1200, 1850, 2850, 4300], dtype=np.float32) / 4300.0


def _bitrate_norm_distance(x: float) -> float:
    """与 6 档归一化码率的最近距离；越小越像「上一码率」行。"""
    if abs(x) < 1e-9:
        return float("inf")
    return float(np.min(np.abs(_BITRATE_NORMS - x)))


def _looks_like_compact6_layout(state: np.ndarray) -> bool:
    """
    启发式：紧凑布局下「上一码率」在第 2 行；Pensieve 则在第 0 行。
    用最后一列谁更接近离散码率档位判断。
    """
    if state.shape[0] != PENSIEVE_STATE_ROWS:
        return False
    v0 = float(state[0, -1])
    v2 = float(state[2, -1])
    if abs(v0) < 1e-9 and abs(v2) < 1e-9:
        return False
    d0 = _bitrate_norm_distance(v0)
    d2 = _bitrate_norm_distance(v2)
    # 紧凑：码率在第 2 行 → d2 < d0
    return d2 < d0


def _matrix_to_pensieve6(matrix: np.ndarray) -> np.ndarray:
    """按源行数将观测矩阵转为 Pensieve (6, H)。"""
    rows = matrix.shape[0]
    if rows == MERINA_STATE_ROWS:
        return _merina11_to_pensieve6(matrix)
    if rows == PENSIEVE_STATE_ROWS:
        return _compact6_to_pensieve6(matrix)
    raise ValueError(
        f"observation 行数应为 {MERINA_STATE_ROWS}（Merina）或 {PENSIEVE_STATE_ROWS}（紧凑 6 行），"
        f"得到 {rows}"
    )


def _extract_state(
    obs: torch.Tensor,
    *,
    state_format: str,
    state_cols: int,
    history_slice: str = "last",
) -> np.ndarray:
    """按目标布局从 observation 提取状态。"""
    matrix = _obs_to_matrix(obs, state_cols=state_cols, history_slice=history_slice)
    if state_format == STATE_FORMAT_MERINA:
        if matrix.shape[0] != MERINA_STATE_ROWS:
            raise ValueError(
                f"--state-format merina 需要 {MERINA_STATE_ROWS} 行观测，得到 {matrix.shape[0]}；"
                f"若源为 6 行紧凑布局请使用默认 pensieve。"
            )
        return matrix
    if state_format == STATE_FORMAT_PENSIEVE:
        out = _matrix_to_pensieve6(matrix)
        if _looks_like_compact6_layout(out):
            out = _compact6_to_pensieve6(out)
        return out
    raise ValueError(
        f"未知 state_format={state_format!r}，应为 {STATE_FORMAT_MERINA!r} 或 {STATE_FORMAT_PENSIEVE!r}"
    )


def _expected_state_shape(state_format: str, state_cols: int) -> tuple[int, int]:
    rows = PENSIEVE_STATE_ROWS if state_format == STATE_FORMAT_PENSIEVE else MERINA_STATE_ROWS
    return (rows, state_cols)


def _scalar_bool(x: torch.Tensor) -> bool:
    return bool(x.detach().cpu().flatten()[0].item())


def _scalar_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().flatten()[0].item())


def _load_trajectory(path: str) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def trajectory_to_exp_pool(
    traj: object,
    *,
    batch_indices: list[int] | None = None,
    state_format: str = STATE_FORMAT_PENSIEVE,
    state_cols: int = DEFAULT_STATE_COLS,
    mark_chunk_end_done: bool = True,
    history_slice: str = "last",
    include_teacher_logits: bool = True,
    require_teacher_logits: bool = False,
) -> ExperiencePool:
    """
    将单个 TensorDict 轨迹转为 ExperiencePool。
    按 (batch_idx, chunk_idx) 顺序展开；每条 episode 仅保留前 ``CHUNKS_TO_KEEP``（47）个块。

    若存在 ``logits`` 字段（典型 shape ``(B, 48, 6)``），写入 ``teacher_logits`` 供 CE+KL 训练。
    """
    required = ("observation", "action", "episode_reward", "done")
    for key in required:
        if key not in traj:
            raise KeyError(f"轨迹缺少字段 {key!r}，现有 keys={list(traj.keys())}")

    obs = traj["observation"]
    act = traj["action"]
    rew = traj["episode_reward"]
    done = traj["done"]

    if obs.ndim < 4:
        raise ValueError(f"observation 维数过少: shape={tuple(obs.shape)}")
    batch_size, num_chunks_src = int(obs.shape[0]), int(obs.shape[1])
    if num_chunks_src < CHUNKS_TO_KEEP:
        raise ValueError(
            f"源 chunk 数 {num_chunks_src} < {CHUNKS_TO_KEEP}，无法舍弃最后一块后仍保留 47 块"
        )

    has_logits_key = "logits" in traj
    if require_teacher_logits and not has_logits_key:
        raise KeyError(
            f"轨迹缺少字段 'logits'，现有 keys={list(traj.keys())}"
        )
    use_logits = include_teacher_logits and has_logits_key
    logits = None
    if use_logits:
        logits = traj["logits"]
        if logits.ndim < 2:
            raise ValueError(f"logits 维数过少: shape={tuple(logits.shape)}")
        if int(logits.shape[0]) != batch_size or int(logits.shape[1]) != num_chunks_src:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} 与 observation batch/chunk "
                f"({batch_size}, {num_chunks_src}) 不一致"
            )
        if int(logits.shape[-1]) != BITRATE_LEVELS:
            raise ValueError(
                f"logits 最后一维应为 {BITRATE_LEVELS}，得到 shape={tuple(logits.shape)}"
            )

    if batch_indices is None:
        batch_indices = list(range(batch_size))
    else:
        for b in batch_indices:
            if b < 0 or b >= batch_size:
                raise IndexError(f"batch_index {b} 超出范围 [0, {batch_size})")

    expected_state_shape = _expected_state_shape(state_format, state_cols)
    pool = ExperiencePool()
    for b in batch_indices:
        for c in range(CHUNKS_TO_KEEP):
            state = _extract_state(
                obs[b, c],
                state_format=state_format,
                state_cols=state_cols,
                history_slice=history_slice,
            )
            if state.shape != expected_state_shape:
                raise ValueError(
                    f"batch={b} chunk={c} 状态形状 {state.shape} != 期望 {expected_state_shape}"
                )
            action = _one_hot_to_bitrate_index(act[b, c])
            reward = _scalar_float(rew[b, c])
            d = _scalar_bool(done[b, c])
            # 第 47 块（索引 46）为 episode 结束；源张量 done 常为全 False
            if mark_chunk_end_done and c == LAST_KEPT_CHUNK_INDEX:
                d = True
            teacher_logits = _extract_logits(logits[b, c]) if use_logits else None
            pool.add(state, action, reward, d, teacher_logits=teacher_logits)

    return pool


def merge_exp_pools(pools: list[ExperiencePool]) -> ExperiencePool:
    merged = ExperiencePool()
    for pool in pools:
        n = len(pool)
        tl_list = getattr(pool, "teacher_logits", [])
        has_tl = len(tl_list) == n
        for i in range(n):
            tl = tl_list[i] if has_tl else None
            merged.add(
                pool.states[i],
                pool.actions[i],
                pool.rewards[i],
                pool.dones[i],
                teacher_logits=tl,
            )
    return merged


def _collect_pt_paths(input_dir: str, pattern: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"在 {input_dir!r} 下未找到匹配 {pattern!r} 的 .pt 文件")
    return paths


def _print_row_order_sample(pool: ExperiencePool, index: int = 0) -> None:
    """打印一条状态最后一列，便于确认行序（0=上一码率, 2=吞吐, 3=时延, 5=剩余）。"""
    s = np.asarray(pool.states[index], dtype=np.float32)
    if s.shape[0] != PENSIEVE_STATE_ROWS:
        return
    print(
        "  行序抽检 (最后一列): "
        f"row0(上一码率)={s[0, -1]:.6f}, row1(缓冲)={s[1, -1]:.6f}, "
        f"row2(吞吐)={s[2, -1]:.6f}, row3(时延)={s[3, -1]:.6f}, "
        f"row5(剩余)={s[5, -1]:.6f}"
    )


def fix_exp_pool_states_inplace(pool: ExperiencePool) -> int:
    """就地修正经验池中 (6,H) 紧凑行序 → NetLLM Pensieve。返回修正条数。"""
    n_fixed = 0
    for i in range(len(pool)):
        s = np.asarray(pool.states[i], dtype=np.float32)
        if s.shape[0] != PENSIEVE_STATE_ROWS:
            continue
        if not _looks_like_compact6_layout(s):
            continue
        pool.states[i] = _compact6_to_pensieve6(s)
        n_fixed += 1
    return n_fixed


def _print_pool_stats(pool: ExperiencePool, label: str) -> None:
    n = len(pool)
    n_done = sum(1 for d in pool.dones if d)
    print(f"[{label}] transitions={n}, done=True 条数={n_done}")
    if n:
        shapes = {np.asarray(s).shape for s in pool.states[: min(8, n)]}
        print(f"  状态形状样例: {shapes}")
        print(
            f"  action min/max: {min(pool.actions)}/{max(pool.actions)}, "
            f"reward min/max: {min(pool.rewards):.4f}/{max(pool.rewards):.4f}"
        )
        if getattr(pool, "has_teacher_logits", False):
            print(f"  teacher_logits: {len(pool.teacher_logits)} 条 (shape (6,) per step)")
        else:
            print("  teacher_logits: 无")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="将 eval_trajectories TensorDict .pt 转为 ExperiencePool pickle"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/traces/train/eval_trajectories",
        help="包含 .pt 轨迹的目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/exp_pools/eval_trajectories_pensieve_6x6.pkl",
        help="输出 ExperiencePool pickle 路径",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="输入文件名 glob",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help="仅使用该 batch 维下标（默认：使用全部 32 条）",
    )
    parser.add_argument(
        "--state-format",
        type=str,
        choices=(STATE_FORMAT_MERINA, STATE_FORMAT_PENSIEVE),
        default=STATE_FORMAT_PENSIEVE,
        help="pensieve=(6,6) NetLLM 行序（默认，v2/v3 训练）；merina=(11,6) legacy 直出",
    )
    parser.add_argument(
        "--state-cols",
        type=int,
        default=DEFAULT_STATE_COLS,
        help="输出状态历史长度（默认 6；与 S_LEN 一致）",
    )
    parser.add_argument(
        "--history-slice",
        type=str,
        choices=("last", "first"),
        default="last",
        help="源 observation 历史维>state-cols 时：last=取最近列，first=取最早列",
    )
    parser.add_argument(
        "--mark-chunk-end-done",
        action="store_true",
        default=True,
        help="每个 batch 第 47 块（chunk 索引 46，舍弃第 48 块后）置 done=True（默认开启）",
    )
    parser.add_argument(
        "--no-mark-chunk-end-done",
        action="store_false",
        dest="mark_chunk_end_done",
        help="仅使用张量中的 done，不在 chunk 末尾补 True",
    )
    parser.add_argument(
        "--fix-existing-pkl",
        type=str,
        default=None,
        metavar="PATH",
        help="不读 .pt，仅将已有 pickle 中紧凑行序 (6,H) 就地改为 NetLLM Pensieve 并写回 --output",
    )
    parser.add_argument(
        "--include-teacher-logits",
        action="store_true",
        default=True,
        help="若 .pt 含 logits (B,48,6)，写入经验池 teacher_logits（默认开启）",
    )
    parser.add_argument(
        "--no-teacher-logits",
        action="store_false",
        dest="include_teacher_logits",
        help="不读取 logits，仅导出 states/actions/rewards/dones",
    )
    parser.add_argument(
        "--require-teacher-logits",
        action="store_true",
        help="每个 .pt 必须包含 logits 字段，否则报错",
    )
    args = parser.parse_args(argv)

    if args.fix_existing_pkl:
        in_path = os.path.abspath(args.fix_existing_pkl)
        out_path = os.path.abspath(args.output)
        with open(in_path, "rb") as f:
            pool = pickle.load(f)
        n_fixed = fix_exp_pool_states_inplace(pool)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(pool, f)
        print(f"已从 {in_path} 修正 {n_fixed}/{len(pool)} 条状态，保存到 {out_path}")
        _print_pool_stats(pool, "fixed")
        _print_row_order_sample(pool, index=47)
        return 0

    input_dir = os.path.abspath(args.input_dir)
    output_path = os.path.abspath(args.output)
    expected_shape = _expected_state_shape(args.state_format, args.state_cols)

    batch_indices = None if args.batch_index is None else [args.batch_index]

    pt_paths = _collect_pt_paths(input_dir, args.pattern)
    print(f"找到 {len(pt_paths)} 个 .pt 文件，目录: {input_dir}")

    pools: list[ExperiencePool] = []
    for path in pt_paths:
        traj = _load_trajectory(path)
        pool = trajectory_to_exp_pool(
            traj,
            batch_indices=batch_indices,
            state_format=args.state_format,
            state_cols=args.state_cols,
            mark_chunk_end_done=args.mark_chunk_end_done,
            history_slice=args.history_slice,
            include_teacher_logits=args.include_teacher_logits,
            require_teacher_logits=args.require_teacher_logits,
        )
        print(f"  {os.path.basename(path)} → {len(pool)} 条 transition")
        pools.append(pool)

    merged = merge_exp_pools(pools)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)

    _print_pool_stats(merged, "merged")
    if args.state_format == STATE_FORMAT_PENSIEVE and len(merged):
        _print_row_order_sample(merged, index=min(47, len(merged) - 1))
    print(f"已保存: {output_path}")
    abr_hint = (
        "run_abr.py --abr-llm-version v2（默认）"
        if args.state_format == STATE_FORMAT_PENSIEVE
        else "run_abr.py --abr-llm-version v3"
    )
    ce_kl_hint = ""
    if getattr(merged, "has_teacher_logits", False):
        ce_kl_hint = " 经验池含 teacher_logits，可用 run_abr.py --loss-type ce_kl。"
    elif args.include_teacher_logits:
        ce_kl_hint = " 未写入 teacher_logits（.pt 无 logits 或使用了 --no-teacher-logits）。"
    print(
        f"\n提示: 输出状态形状 {expected_shape}（{args.state_format}），可用于 {abr_hint}。"
        f"源 observation 行数为 11（Merina）或 6（紧凑布局），历史维按 --history-slice={args.history_slice} 截取为 {args.state_cols}；"
        f"每条 episode 保留前 {CHUNKS_TO_KEEP} 块（舍弃源数据中索引 {LAST_KEPT_CHUNK_INDEX + 1} 的最后一块）。"
        f"{ce_kl_hint}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
