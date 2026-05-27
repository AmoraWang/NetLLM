import os
import random
import numpy as np
from baseline_special.utils.constants import BITRATE_LEVELS
try:  # baselines use a different conda environment without torch, so we need to skip ModuleNotFoundError when runing baselines
    import torch
except ModuleNotFoundError:
    pass


def _teacher_logits_to_bct(teacher_logits_batch, device):
    """
    将 DataLoader 拼好的 teacher logits 转为 (B, C, T)，C=BITRATE_LEVELS。

    ``ExperienceDataset`` 每条样本为 (T, 6)；batch_size=1 时 collate 常为 (1, T, 6)。
    """
    tl = teacher_logits_batch
    if isinstance(tl, (list, tuple)) and len(tl) == 1:
        tl = tl[0]

    if isinstance(tl, torch.Tensor):
        t = tl.detach().float().to(device)
    elif isinstance(tl, np.ndarray):
        arr = np.asarray(tl, dtype=np.float32)
        if arr.dtype == object:
            arr = np.stack(
                [np.asarray(x, dtype=np.float32).reshape(BITRATE_LEVELS) for x in arr],
                axis=0,
            )
        t = torch.from_numpy(arr).float().to(device)
    elif isinstance(tl, (list, tuple)):
        t = torch.stack(
            [
                torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(BITRATE_LEVELS), device=device)
                for x in tl
            ],
            dim=0,
        )
    else:
        raise TypeError(f"Unsupported teacher_logits batch type: {type(tl)!r}")

    if t.dim() == 1:
        return t.view(1, BITRATE_LEVELS, 1)
    if t.dim() == 2:
        # (T, C)
        if t.shape[-1] != BITRATE_LEVELS:
            raise ValueError(f"teacher_logits 最后一维应为 {BITRATE_LEVELS}，得到 {tuple(t.shape)}")
        return t.unsqueeze(0).permute(0, 2, 1)
    if t.dim() == 3:
        if t.shape[-1] == BITRATE_LEVELS:
            return t.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        if t.shape[1] == BITRATE_LEVELS:
            return t  # already (B, C, T)
        raise ValueError(f"无法解析 teacher_logits 形状 {tuple(t.shape)}")
    raise ValueError(f"teacher_logits 维数异常: {t.dim()}")


def process_batch(batch, device='cpu', expect_teacher_logits=False):
    """
    Process batch of data.

    With teacher logits, batch has 5 fields; ``teacher_logits`` tensor shape (1, C, T).
    """
    if len(batch) == 5:
        states, actions, returns, timesteps, teacher_logits_batch = batch
    elif len(batch) == 4:
        states, actions, returns, timesteps = batch
    else:
        raise ValueError(f"Expected batch of length 4 or 5, got {len(batch)}")

    states = torch.cat(states, dim=0).unsqueeze(0).float().to(device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(1, -1)
    labels = actions.long()
    actions = ((actions + 1) / BITRATE_LEVELS).unsqueeze(2)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device).reshape(1, -1, 1)
    timesteps = torch.as_tensor(timesteps, dtype=torch.int32, device=device).unsqueeze(0)

    teacher_logits_out = None
    if len(batch) == 5:
        teacher_logits_out = _teacher_logits_to_bct(teacher_logits_batch, device)
    elif expect_teacher_logits:
        raise ValueError("loss_type='ce_kl' requires teacher_logits in the experience pool.")

    return states, actions, returns, timesteps, labels, teacher_logits_out


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def action2bitrate(action, last_bit_rate):
    """
    Genet special.
    Genet uses this strategy for converting actions to bitrates during testing.
    """
    # selected_action is 0-2
    # naive step implementation
    if action == 1:
        bit_rate = last_bit_rate
    elif action == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    if bit_rate < 0:
        bit_rate = 0
    if bit_rate > 5:
        bit_rate = 5
    return bit_rate


def calc_mean_reward(result_files, test_dir, str, skip_first_reward=True, skip_last_reward=True):
    """
    从 plm_special/test 写出的结果文件（每行至少 8 列，第 8 列 parse[7] 为 reward）聚合平均 QoE。

    基线评测约定：每个 trace 文件内先取全部有效 reward 行，再按块边界丢弃首尾——即只用
    ``vals[1:-1]``（中间块），与 ``skip_first_reward`` / ``skip_last_reward`` 为 True 时一致；
    例如共 49 个块对应 49 行 reward 时，中间 47 个块参与平均。
    """
    matching = [s for s in result_files if str in s]
    reward = []
    count = 0
    for log_file in matching:
        count += 1
        vals = []
        with open(os.path.join(test_dir, log_file), 'r') as f:
            for line in f:
                parse = line.split()
                if len(parse) < 8:
                    continue
                vals.append(float(parse[7]))
        if skip_first_reward and vals:
            vals = vals[1:]
        if skip_last_reward and vals:
            vals = vals[:-1]
        reward.extend(vals)
    print(count)
    if not reward:
        return 0.0
    return np.mean(reward)


def mean_reward_from_results_log(results_log, skip_first_reward=True, skip_last_reward=True):
    """
    与 calc_mean_reward(..., skip_first_reward, skip_last_reward) 一致，但从内存中的 results_log 聚合。
    results_log: trace_idx -> list of rows，每行至少 8 列，第 8 列为 reward（与 test 中 append 一致）。
    """
    reward = []
    for values in results_log.values():
        vals = []
        for items in values:
            if len(items) < 8:
                continue
            vals.append(float(items[7]))
        if skip_first_reward and vals:
            vals = vals[1:]
        if skip_last_reward and vals:
            vals = vals[:-1]
        reward.extend(vals)
    return float(np.mean(reward)) if reward else 0.0


def clear_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        return
    file_list = os.listdir(directory)
    for file in file_list:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)