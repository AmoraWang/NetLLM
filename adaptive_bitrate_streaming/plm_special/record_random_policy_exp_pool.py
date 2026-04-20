"""
使用随机码率策略在仿真环境中 rollout，生成与 ExperiencePool 相同格式的 pkl。
"""
from types import SimpleNamespace

from plm_special.random_bitrate_policy import RandomUniformBitratePolicy
from plm_special.test_record_exp_pool import test_on_env_with_exp_pool


def record_random_policy_exp_pool(
    device,
    seed,
    env_settings,
    results_dir,
    exp_pool_path,
    max_ep_num,
    process_reward_fn=None,
    target_return=0.0,
    skip_first_transition_per_episode=True,
    policy_seed=None,
    write_sim_logs=True,
):
    """
    Args:
        device: 如 'cuda:0' 或 'cpu'
        seed: 传入 test_on_env_with_exp_pool，用于 set_random_seed（环境与日志可复现）
        policy_seed: 若不为 None，仅用于 RandomUniformBitratePolicy 的独立 RNG（否则与全局 seed 解耦时用）
        target_return: 仅用于与 test流程一致的 return 累减统计；随机策略不读该值
        write_sim_logs: False 时不写 result_sim_abr_*，mean_qoe 从内存计算（适合 wm_traces 仅保留 pkl）
    """
    args = SimpleNamespace(device=device, seed=seed)
    policy = RandomUniformBitratePolicy(seed=policy_seed if policy_seed is not None else seed)
    return test_on_env_with_exp_pool(
        args,
        policy,
        results_dir,
        env_settings,
        target_return,
        exp_pool_path,
        max_ep_num=max_ep_num,
        process_reward_fn=process_reward_fn,
        seed=seed,
        skip_first_transition_per_episode=skip_first_transition_per_episode,
        write_sim_logs=write_sim_logs,
    )
