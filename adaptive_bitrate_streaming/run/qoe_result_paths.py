"""各 baseline 测试结果日志目录路径（与 run_*_baseline_test / DP.py 约定一致）。"""
from __future__ import annotations

import os

from config import cfg

# 用户常用别名 → config.trace_dirs 键
TRACE_ALIASES: dict[str, str] = {
    "fcc-test": "fcc-test",
    "norway3g-test": "Norway3G-test",
    "norway3G-test": "Norway3G-test",
    "Norway3G-test": "Norway3G-test",
    "hsr-test": "hsr-test",
    "oboe-test": "Oboe-test",
    "Oboe-test": "Oboe-test",
}

# 输出文件名用的 slug（小写 oboe / norway3g 便于脚本批处理）
TRACE_OUTPUT_SLUG: dict[str, str] = {
    "fcc-test": "fcc-test",
    "Norway3G-test": "norway3G-test",
    "hsr-test": "hsr-test",
    "Oboe-test": "oboe-test",
}

DEFAULT_CDF_ALGOS: tuple[str, ...] = (
    "mpc",
    "bba",
    "genet",
    "comyco",
    "merina",
    "DP",
)

OURS_PT_ALGO = "ours"

# 自动收集默认：baseline + Ours（.pt 文件名中的 algo 键）
DEFAULT_COLLECT_ALGOS: tuple[str, ...] = DEFAULT_CDF_ALGOS + (OURS_PT_ALGO,)

# run_abr.py --test 默认与 bash/run_plot_*.sh 中 OURS_DIR 一致
DEFAULT_ABRLLM_COLLECT: dict[str, object] = {
    "plm_type": "llama",
    "plm_size": "base",
    "abr_llm_version": "v2",
    "rank": 128,
    "w": 20,
    "gamma": 1.0,
    "target_return_scale": 1.0,
}

# collect .pt 文件名中的 algo → plot_qoe_* 图例键名
PT_ALGO_TO_PLOT_KEY: dict[str, str] = {
    "bba": "BOLA",
    "mpc": "RobustMPC",
    "genet": "Pensieve",
    "merina": "MERINA",
    "comyco": "Comyco",
    "DP": "Oracle",
    OURS_PT_ALGO: "ours",
}

PLOT_KEY_TO_PT_ALGO: dict[str, str] = {
    plot_key: pt_algo for pt_algo, plot_key in PT_ALGO_TO_PLOT_KEY.items()
}


def plot_key_to_pt_algo(plot_key: str) -> str | None:
    """图例键名 → .pt 文件名中的 algo（如 BOLA→bba, ours→ours）。"""
    if plot_key.lower() == OURS_PT_ALGO:
        return OURS_PT_ALGO
    return PLOT_KEY_TO_PT_ALGO.get(plot_key)

DEFAULT_CDF_TRACES: tuple[str, ...] = (
    "fcc-test",
    "Norway3G-test",
    "hsr-test",
    "Oboe-test",
)


def normalize_trace_key(trace: str) -> str:
    key = TRACE_ALIASES.get(trace, trace)
    if key not in cfg.trace_dirs:
        raise KeyError(
            f"未知 trace: {trace!r}（可用: {', '.join(sorted(set(TRACE_ALIASES.values())))}）"
        )
    return key


def trace_output_slug(trace_cfg_key: str) -> str:
    return TRACE_OUTPUT_SLUG.get(trace_cfg_key, trace_cfg_key)


def count_trace_files(trace_cfg_key: str) -> int:
    trace_dir = cfg.trace_dirs[trace_cfg_key]
    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(f"trace 目录不存在: {trace_dir}")
    return sum(
        1
        for name in os.listdir(trace_dir)
        if os.path.isfile(os.path.join(trace_dir, name)) and not name.startswith(".")
    )


def resolve_trace_num(trace_cfg_key: str, trace_num: int | None) -> int:
    """trace_num<=0 或 None 表示使用该 trace 目录下全部文件数。"""
    if trace_num is not None and trace_num > 0:
        return trace_num
    return count_trace_files(trace_cfg_key)


def build_baseline_log_dir(
    algo: str,
    *,
    trace: str,
    video: str = "video1",
    trace_num: int,
    fixed_order: bool = True,
    seed: int = 666,
    test_rounds: int = 1,
) -> str:
    """
    返回某算法在某测试集上的日志根目录（递归扫描 result_sim_abr_* / log_test_*）。

    algo: mpc | bba | genet | DP | merina | comyco
    """
    algo = algo.lower()
    trace_key = normalize_trace_key(trace)
    base = os.path.join(cfg.results_dir, f"{trace_key}_{video}")
    fixed = str(fixed_order)
    rounds_tag = f"seed_{seed}_rounds_{test_rounds}"

    if algo in ("mpc", "bba"):
        return os.path.join(
            base,
            f"trace_num_{trace_num}_fixed_{fixed}",
            "rule_baseline",
            algo,
            rounds_tag,
        )
    if algo == "genet":
        return os.path.join(
            base,
            f"trace_num_{trace_num}_fixed_{fixed}",
            "rl_baseline",
            "genet",
            rounds_tag,
        )
    if algo == "dp":
        return os.path.join(
            base,
            f"trace_num_{trace_num}_fixed_{fixed}",
            "DP",
            rounds_tag,
        )
    if algo == "merina":
        return os.path.join(base, f"trace_num_{trace_num}_merina", rounds_tag)
    if algo == "comyco":
        return os.path.join(base, f"trace_num_{trace_num}_comyco_merina", rounds_tag)

    raise ValueError(f"未知算法: {algo!r}，支持: {DEFAULT_CDF_ALGOS}")


def build_ours_log_dir(
    *,
    trace: str,
    video: str = "video1",
    trace_num: int,
    fixed_order: bool = True,
    seed: int = 666,
    plm_type: str = "llama",
    plm_size: str = "base",
    abr_llm_version: str = "v2",
    rank: int = 128,
    w: int = 20,
    gamma: float = 1.0,
    target_return_scale: float = 1.0,
) -> str:
    """
    ABRLLM 测试结果目录（与 ``run/run_abr.py`` 中 ``results_dir`` 一致）。

    示例::

        artifacts/results/fcc-test_video1/trace_num_100_fixed_True/llama_base/
            abrllm_v2_rank_128_w_20_gamma_1.0_tgt_scale_1.0_seed_666/
    """
    trace_key = normalize_trace_key(trace)
    base = os.path.join(cfg.results_dir, f"{trace_key}_{video}")
    fixed = str(fixed_order)
    return os.path.join(
        base,
        f"trace_num_{trace_num}_fixed_{fixed}",
        f"{plm_type}_{plm_size}",
        (
            f"abrllm_{abr_llm_version}_rank_{rank}_w_{w}_gamma_{gamma}"
            f"_tgt_scale_{target_return_scale}_seed_{seed}"
        ),
    )


def build_algo_log_dir(
    algo: str,
    *,
    trace: str,
    video: str = "video1",
    trace_num: int,
    fixed_order: bool = True,
    seed: int = 666,
    test_rounds: int = 1,
    plm_type: str = "llama",
    plm_size: str = "base",
    abr_llm_version: str = "v2",
    rank: int = 128,
    w: int = 20,
    gamma: float = 1.0,
    target_return_scale: float = 1.0,
) -> str:
    """baseline 或 ``ours``（ABRLLM）测试结果日志根目录。"""
    if algo.lower() == OURS_PT_ALGO:
        return build_ours_log_dir(
            trace=trace,
            video=video,
            trace_num=trace_num,
            fixed_order=fixed_order,
            seed=seed,
            plm_type=plm_type,
            plm_size=plm_size,
            abr_llm_version=abr_llm_version,
            rank=rank,
            w=w,
            gamma=gamma,
            target_return_scale=target_return_scale,
        )
    return build_baseline_log_dir(
        algo,
        trace=trace,
        video=video,
        trace_num=trace_num,
        fixed_order=fixed_order,
        seed=seed,
        test_rounds=test_rounds,
    )


def ours_run_command_hint(
    trace_cfg_key: str,
    trace_num: int,
    *,
    video: str = "video1",
    seed: int = 666,
    plm_type: str = "llama",
    plm_size: str = "base",
    abr_llm_version: str = "v2",
    rank: int = 128,
    w: int = 20,
    gamma: float = 1.0,
    target_return_scale: float = 1.0,
) -> str:
    tn = trace_num if trace_num > 0 else -1
    return (
        f"python run/run_abr.py --test --abr-llm-version {abr_llm_version} "
        f"--trace {trace_cfg_key} --video {video} --trace-num {tn} "
        f"--seed {seed} --fixed-order --test-rounds 1 "
        f"--plm-type {plm_type} --plm-size {plm_size} --rank {rank} --w {w} "
        f"--gamma {gamma} --target-return-scale {target_return_scale}"
    )


def algo_run_command_hint(
    algo: str,
    trace_cfg_key: str,
    trace_num: int,
    *,
    video: str = "video1",
    seed: int = 666,
    test_rounds: int = 1,
    plm_type: str = "llama",
    plm_size: str = "base",
    abr_llm_version: str = "v2",
    rank: int = 128,
    w: int = 20,
    gamma: float = 1.0,
    target_return_scale: float = 1.0,
) -> str:
    if algo.lower() == OURS_PT_ALGO:
        return ours_run_command_hint(
            trace_cfg_key,
            trace_num,
            video=video,
            seed=seed,
            plm_type=plm_type,
            plm_size=plm_size,
            abr_llm_version=abr_llm_version,
            rank=rank,
            w=w,
            gamma=gamma,
            target_return_scale=target_return_scale,
        )
    return baseline_run_command_hint(algo, trace_cfg_key, trace_num)


def baseline_run_command_hint(algo: str, trace_cfg_key: str, trace_num: int) -> str:
    """生成补跑实验的建议命令（单行）。"""
    slug = trace_output_slug(trace_cfg_key)
    tn = trace_num if trace_num > 0 else -1
    if algo in ("mpc", "bba", "genet"):
        extra = "  # conda activate tensorflowv1" if algo == "genet" else ""
        return (
            f"python run/run_rule_baselines_test.py --algorithm {algo} "
            f"--trace {trace_cfg_key} --video video1 --trace-num {tn} "
            f"--seed 666 --test-rounds 1 --fixed-order{extra}"
        )
    if algo == "merina":
        return (
            f"python run/run_merina_baseline_test.py --trace {trace_cfg_key} "
            f"--video video1 --trace-num {tn} --seed 666 --test-rounds 1 --fixed-order"
        )
    if algo == "comyco":
        return (
            f"python run/run_comyco_merina_baseline_test.py --trace {trace_cfg_key} "
            f"--video video1 --trace-num {tn} --seed 666 --test-rounds 1 --fixed-order "
            f"--cuda-id 0  # conda activate tensorflowv1"
        )
    if algo == "DP":
        return (
            f"python generate_exp_pool/DP.py --mode test --trace {trace_cfg_key} "
            f"--video video1 --trace-num {tn} --seed 666 --test-rounds 1"
        )
    return f"# unknown algo {algo} trace {slug}"
