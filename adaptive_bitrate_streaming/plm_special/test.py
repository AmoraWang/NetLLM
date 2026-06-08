import copy
import os
import shutil
import time

import numpy as np
import torch

from baseline_special.env import Environment
from baseline_special.utils.constants import (
    REBUF_PENALTY, SMOOTH_PENALTY, DEFAULT_QUALITY, S_INFO, S_LEN, BITRATE_LEVELS, BUFFER_NORM_FACTOR,
    M_IN_K, SMOOTH_PENALTY, VIDEO_BIT_RATE, CHUNK_TIL_VIDEO_END_CAP, MAX_VIDEO_BIT_RATE, DEFAULT_QUALITY,
)
from baseline_special.utils.mahimahi_start import apply_test_round_mahimahi_ptrs
from plm_special.utils.utils import calc_mean_reward, clear_dir, set_random_seed


def _write_results_round(results_dir, results_log, all_file_names):
    clear_dir(results_dir)
    for trace_idx, values in results_log.items():
        result_path = os.path.join(results_dir, 'result_sim_abr_{}'.format(all_file_names[trace_idx]))
        with open(result_path, 'w') as result_file:
            for items in values:
                time_stamp, bit_rate, buffer_size, rebuf, video_chunk_size, download_time, smoothness, reward = items
                result_file.write(str(time_stamp) + '\t' +
                                  str(bit_rate) + '\t' +
                                  str(buffer_size) + '\t' +
                                  str(rebuf) + '\t' +
                                  str(video_chunk_size) + '\t' +
                                  str(download_time) + '\t' +
                                  str(smoothness) + '\t' +
                                  str(reward) + '\n')


def _run_single_test_round(args, model, env_kw, numpy_random_seed, max_ep_num, target_return, process_reward_fn):
    """执行一整次测试（走完 max_ep_num 个 episode），返回该轮的指标与明细。"""
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    kw = dict(env_kw)
    if numpy_random_seed is not None:
        kw['numpy_random_seed'] = int(numpy_random_seed) % (2 ** 32)

    episodes_return = 0.0
    episodes_len = 0
    decision_times = []
    ep_count = 0
    results_log = {}

    set_random_seed(args.seed)

    with torch.no_grad():
        env = Environment(**kw)
        time_stamp = 0
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        state = torch.zeros((1, 1, S_INFO, S_LEN), dtype=torch.float32, device=args.device)
        timestep = 0
        target_return_clone = copy.deepcopy(target_return)
        trace_idx = env.trace_idx
        results_log[trace_idx] = []

        while True:
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = env.get_video_chunk(bit_rate)

            time_stamp += delay
            time_stamp += sleep_time

            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            smoothness = abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            last_bit_rate = bit_rate

            results_log[trace_idx].append([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate], buffer_size,
                                           rebuf, video_chunk_size, delay, smoothness, reward])

            state = torch.roll(state, -1, dims=-1)
            state[..., 0, -1] = VIDEO_BIT_RATE[bit_rate] / MAX_VIDEO_BIT_RATE
            state[..., 1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[..., 2, -1] = video_chunk_size / delay / M_IN_K
            state[..., 3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR
            state[..., 4, :BITRATE_LEVELS] = torch.as_tensor(
                next_video_chunk_sizes, device=args.device, dtype=torch.float32
            ) / M_IN_K / M_IN_K
            state[..., 5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

            if timestep > 0:
                reward = process_reward_fn(reward)
                target_return = target_return - reward
                episodes_return += reward
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

    mean_qoe_per_chunk = episodes_return / episodes_len if episodes_len > 0 else 0.0
    avg_decision_time = sum(decision_times) / len(decision_times) if len(decision_times) > 0 else 0.0

    round_log = {
        'episodes_count': ep_count,
        'total_chunks': episodes_len,
        'total_qoe': episodes_return,
        'mean_qoe_per_chunk': mean_qoe_per_chunk,
        'avg_decision_time': avg_decision_time,
        'total_decision_time': sum(decision_times),
        'decision_count': len(decision_times),
    }
    return results_log, round_log, decision_times


def _reset_results_dir(results_dir):
    """清空 results_dir 下上一轮留下的结果文件与 round_* 子目录。"""
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        return
    for name in os.listdir(results_dir):
        path = os.path.join(results_dir, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path) and name.startswith('round_'):
                shutil.rmtree(path)
        except OSError:
            pass


def test_on_env(args, model, results_dir, env_settings, target_return, max_ep_num=100, process_reward_fn=None, seed=0,
                num_test_rounds=1):
    """
    num_test_rounds>1：每轮在创建 Environment 前用 time.time_ns() 派生 numpy 种子，使每条带宽 trace 独立随机起点；
    最终 mean_reward / mean_qoe 为多轮各轮 mean_qoe（与 calc_mean_reward 一致：每 trace 丢弃首尾块）的算术平均。
    每轮明细写入 results_dir/round_XXX/，根目录写 summary.txt。
    num_test_rounds==1：用 ``args.seed`` 生成共享 ``all_mahimahi_ptrs``，各算法同一条 trace 同一起点；结果写在 results_dir 根目录。
    """
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    assert num_test_rounds >= 1
    multi = num_test_rounds > 1
    _reset_results_dir(results_dir)

    env_kw = {k: v for k, v in env_settings.items() if k != 'numpy_random_seed'}
    all_file_names = env_settings['all_file_names']
    test_start = time.time()

    round_mean_qoes = []
    all_decision_times = []
    last_round_log = None
    aggregated_episodes_return = 0.0
    total_chunks_all = 0
    episodes_all_rounds = 0

    for r in range(num_test_rounds):
        if multi:
            round_out = os.path.join(results_dir, f'round_{r:03d}')
        else:
            round_out = results_dir

        round_kw = dict(env_kw)
        env_seed = apply_test_round_mahimahi_ptrs(
            round_kw,
            base_seed=int(args.seed),
            round_idx=r,
            num_test_rounds=num_test_rounds,
        )

        results_log, round_log, decision_times = _run_single_test_round(
            args, model, round_kw, env_seed, max_ep_num, copy.deepcopy(target_return), process_reward_fn
        )

        _write_results_round(round_out, results_log, all_file_names)
        mean_qoe_round = calc_mean_reward(
            result_files=os.listdir(round_out),
            test_dir=round_out,
            str='',
            skip_first_reward=True,
            skip_last_reward=True,
        )
        round_mean_qoes.append(mean_qoe_round)
        aggregated_episodes_return += round_log['total_qoe']
        total_chunks_all += round_log['total_chunks']
        episodes_all_rounds += round_log['episodes_count']
        all_decision_times.extend(decision_times)
        last_round_log = round_log

    elapsed = time.time() - test_start

    mean_qoe = float(np.mean(round_mean_qoes)) if round_mean_qoes else 0.0
    std_qoe_across_rounds = float(np.std(round_mean_qoes)) if len(round_mean_qoes) > 1 else 0.0

    avg_decision_time = sum(all_decision_times) / len(all_decision_times) if all_decision_times else 0.0
    total_decision_time = sum(all_decision_times)
    mean_qoe_per_chunk = (
        aggregated_episodes_return / total_chunks_all if total_chunks_all > 0 else 0.0
    )

    test_log = {
        'time': elapsed,
        'num_test_rounds': num_test_rounds,
        'mean_reward': mean_qoe,
        'mean_qoe': mean_qoe,
        'mean_qoe_std_across_rounds': std_qoe_across_rounds,
        'mean_qoe_per_round': round_mean_qoes,
        'total_qoe': aggregated_episodes_return,
        'mean_qoe_per_chunk': mean_qoe_per_chunk,
        'episodes_count': episodes_all_rounds,
        'total_chunks': total_chunks_all,
        'avg_decision_time': avg_decision_time,
        'total_decision_time': total_decision_time,
        'decision_count': len(all_decision_times),
    }

    if multi:
        summary_path = os.path.join(results_dir, 'summary.txt')
        with open(summary_path, 'w') as sf:
            sf.write(f'num_test_rounds\t{num_test_rounds}\n')
            sf.write(f'mean_qoe\t{mean_qoe:.6f}\n')
            sf.write(f'std_qoe_across_rounds\t{std_qoe_across_rounds:.6f}\n')
            sf.write(f'mean_qoe_per_round\t{round_mean_qoes}\n')

    # print(f'\nDecision Time Statistics:')
    # print(f'  Total decisions: {len(all_decision_times)}')
    # print(f'  Total decision time: {total_decision_time:.4f} seconds')
    # print(f'  Average decision time: {avg_decision_time:.6f} seconds ({avg_decision_time * 1000:.3f} ms)')
    # if len(all_decision_times) > 0:
    #     print(f'  Min decision time: {min(all_decision_times) * 1000:.3f} ms')
    #     print(f'  Max decision time: {max(all_decision_times) * 1000:.3f} ms')
    # if multi:
    #     print(f'\nMulti-round test: mean_qoe (avg over rounds) = {mean_qoe:.6f}, std = {std_qoe_across_rounds:.6f}')

    return test_log
