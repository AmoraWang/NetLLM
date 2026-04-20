"""
测试时完整记录 (state, action, reward, done)，结构与 ExperiencePool 一致，并保存为 .pkl。

与 generate_exp_pool / DP 生成的经验池相同：每条 transition 为
  - state: numpy.ndarray, shape (S_INFO, S_LEN) == (6, 6), float32
  - action: int, 码率档位 0..BITRATE_LEVELS-1
  - reward: float, 与环境中定义的 QoE reward 一致（未经过 process_reward_fn）
  - done: bool, 该步之后是否 episode 结束

记录时机：每一步在调用 env.get_video_chunk(bit_rate) 之前，保存当前 state、
即将用于下载的 bit_rate；在环境返回后写入与 generate_exp_pool 相同的 raw reward 与 end_of_video。

默认 **不写入每个 episode 的第一个 transition**（``timestep==0`` 的那一步，与 ``generate_exp_pool`` / Pensieve 跳过首步 reward 一致），
因此每个完整 episode 在 pkl 中通常为 **47 条** transition（总 chunk 数 48 时）。

用法（在已有测试流程中替换 `test_on_env`）::

    from plm_special.test_record_exp_pool import test_on_env_with_exp_pool

    out_pkl = os.path.join(results_dir, "test_exp_pool.pkl")
    results = test_on_env_with_exp_pool(
        args, model, results_dir, env_settings, target_return,
        exp_pool_path=out_pkl,
        max_ep_num=args.trace_num,
        process_reward_fn=test_process_reward_fn,
        seed=args.seed,
    )
"""

import copy
import os
import pickle
import time

import numpy as np
import torch

from baseline_special.env import Environment
from baseline_special.utils.constants import (
    BITRATE_LEVELS,
    BUFFER_NORM_FACTOR,
    CHUNK_TIL_VIDEO_END_CAP,
    DEFAULT_QUALITY,
    MAX_VIDEO_BIT_RATE,
    M_IN_K,
    REBUF_PENALTY,
    S_INFO,
    S_LEN,
    SMOOTH_PENALTY,
    VIDEO_BIT_RATE,
)
from plm_special.data.exp_pool import ExperiencePool
from plm_special.utils.utils import calc_mean_reward, clear_dir, mean_reward_from_results_log, set_random_seed


def test_on_env_with_exp_pool(
    args,
    model,
    results_dir,
    env_settings,
    target_return,
    exp_pool_path,
    max_ep_num=100,
    process_reward_fn=None,
    seed=0,
    skip_first_transition_per_episode=True,
    write_sim_logs=True,
):
    """
    与 plm_special.test.test_on_env 相同的仿真与指标，额外将 transition 写入 ExperiencePool 并保存为 pkl。

    Args:
        exp_pool_path: 输出 .pkl 路径（含文件名），父目录不存在时会创建。
        skip_first_transition_per_episode: 为 True 时每个 episode 不保存第一条 transition（仅 pkl，仿真与 result 日志仍完整）。
        write_sim_logs: 为 False 时不写入 result_sim_abr_* 文本，mean_qoe 仍从内存 results_log 计算；results_dir 可忽略。

    Returns:
        test_log: 与 test_on_env 相同字段，并增加 exp_pool_path、exp_pool_size。
    """
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    exp_pool = ExperiencePool()
    test_log = {}
    test_start = time.time()
    results_log = {}

    with torch.no_grad():
        if hasattr(model, "clear_dq"):
            model.clear_dq()

        env = Environment(**env_settings)

        time_stamp = 0
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        state = torch.zeros((1, 1, S_INFO, S_LEN), dtype=torch.float32, device=args.device)
        timestep = 0
        target_return_clone = copy.deepcopy(target_return)
        ep_count = 0
        episodes_return, episodes_len = 0, 0
        decision_times = []

        trace_idx = env.trace_idx
        results_log[trace_idx] = []

        set_random_seed(args.seed)

        while True:
            # 与经验池一致：记录执行 get_video_chunk 前的观测与即将采取的动作
            state_np = state.detach().cpu().numpy().reshape(S_INFO, S_LEN).astype(np.float32)
            action_int = int(bit_rate)

            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = env.get_video_chunk(bit_rate)

            time_stamp += delay
            time_stamp += sleep_time

            reward_raw = VIDEO_BIT_RATE[bit_rate] / M_IN_K - REBUF_PENALTY * rebuf - SMOOTH_PENALTY * abs(
                VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]
            ) / M_IN_K

            smoothness = abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            last_bit_rate = bit_rate

            results_log[trace_idx].append(
                [time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate], buffer_size, rebuf, video_chunk_size, delay, smoothness, reward_raw]
            )

            # 与 generate_exp_pool 一致：每个 episode 跳过首条 transition，pkl 中每集 47 条（48 chunk 时）
            if not skip_first_transition_per_episode or timestep > 0:
                exp_pool.add(
                    state=state_np,
                    action=action_int,
                    reward=float(reward_raw),
                    done=bool(end_of_video),
                )

            state = torch.roll(state, -1, dims=-1)
            state[..., 0, -1] = VIDEO_BIT_RATE[bit_rate] / MAX_VIDEO_BIT_RATE
            state[..., 1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[..., 2, -1] = video_chunk_size / delay / M_IN_K
            state[..., 3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR
            state[..., 4, :BITRATE_LEVELS] = torch.as_tensor(next_video_chunk_sizes, device=args.device, dtype=torch.float32) / M_IN_K / M_IN_K
            state[..., 5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

            if timestep > 0:
                r_proc = process_reward_fn(reward_raw)
                target_return = target_return - r_proc
                episodes_return += r_proc
                episodes_len += 1

            decision_start = time.time()
            bit_rate = model.sample(state, target_return, timestep)
            decision_end = time.time()
            decision_times.append(decision_end - decision_start)
            timestep += 1

            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                torch.zero_(state)
                timestep = 0
                target_return = copy.deepcopy(target_return_clone)

                ep_count += 1
                if ep_count >= max_ep_num:
                    break

                trace_idx = env.trace_idx
                results_log[trace_idx] = []

    test_log["time"] = time.time() - test_start

    if write_sim_logs:
        clear_dir(results_dir)
        all_file_names = env_settings["all_file_names"]
        for tidx, values in results_log.items():
            result_path = os.path.join(results_dir, "result_sim_abr_{}".format(all_file_names[tidx]))
            with open(result_path, "w", encoding="utf-8") as result_file:
                for items in values:
                    ts, br, buf, rb, chsz, dl, sm, rw = items
                    result_file.write(
                        str(ts)
                        + "\t"
                        + str(br)
                        + "\t"
                        + str(buf)
                        + "\t"
                        + str(rb)
                        + "\t"
                        + str(chsz)
                        + "\t"
                        + str(dl)
                        + "\t"
                        + str(sm)
                        + "\t"
                        + str(rw)
                        + "\n"
                    )

        mean_qoe = calc_mean_reward(result_files=os.listdir(results_dir), test_dir=results_dir, str="", skip_first_reward=True)
    else:
        mean_qoe = mean_reward_from_results_log(results_log, skip_first_reward=True)
    total_qoe = episodes_return
    mean_qoe_per_chunk = episodes_return / episodes_len if episodes_len > 0 else 0.0
    avg_decision_time = sum(decision_times) / len(decision_times) if decision_times else 0.0
    total_decision_time = sum(decision_times)

    test_log.update(
        {
            "mean_reward": mean_qoe,
            "mean_qoe": mean_qoe,
            "total_qoe": total_qoe,
            "mean_qoe_per_chunk": mean_qoe_per_chunk,
            "episodes_count": ep_count,
            "total_chunks": episodes_len,
            "avg_decision_time": avg_decision_time,
            "total_decision_time": total_decision_time,
            "decision_count": len(decision_times),
            "exp_pool_path": os.path.abspath(exp_pool_path),
            "exp_pool_size": len(exp_pool),
        }
    )

    _parent = os.path.dirname(os.path.abspath(exp_pool_path))
    if _parent:
        os.makedirs(_parent, exist_ok=True)
    with open(exp_pool_path, "wb") as f:
        pickle.dump(exp_pool, f)
    print(f"ExperiencePool saved: {os.path.abspath(exp_pool_path)}  (transitions: {len(exp_pool)})")

    print("\nDecision Time Statistics:")
    print(f"  Total decisions: {len(decision_times)}")
    print(f"  Total decision time: {total_decision_time:.4f} seconds")
    print(f"  Average decision time: {avg_decision_time:.6f} seconds ({avg_decision_time * 1000:.3f} ms)")
    if decision_times:
        print(f"  Min decision time: {min(decision_times) * 1000:.3f} ms")
        print(f"  Max decision time: {max(decision_times) * 1000:.3f} ms")

    return test_log
