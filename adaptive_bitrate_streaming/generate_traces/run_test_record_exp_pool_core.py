"""
单次测试与批量测试共用的逻辑（供 run_test_record_exp_pool.py / run_test_record_exp_pool_batch.py 调用）。

本文件位于 generate_traces/ 下，依赖 _bootstrap 将 cwd 置于 adaptive_bitrate_streaming 根目录。
"""
import os
import pickle
from collections import Counter

import _bootstrap  # noqa: F401  — chdir + sys.path

import torch
from munch import Munch

from config import cfg
from baseline_special.utils.utils import load_traces
from plm_special.data.dataset import ExperienceDataset
from plm_special.test_record_exp_pool import test_on_env_with_exp_pool
from plm_special.utils.utils import set_random_seed
from run_abr import load_model

PROJECT_ROOT = _bootstrap.PROJECT_ROOT
# 供旧代码/外部 import：与 PROJECT_ROOT 相同
_PROJECT_ROOT = PROJECT_ROOT
DEFAULT_RECORD_EXP_POOL_DIR = os.path.join(PROJECT_ROOT, "wm_traces", "model")
DEFAULT_RECORD_EXP_POOL_NAME = "test_record_exp_pool.pkl"


def build_parser():
    """与 run_test_record_exp_pool.py 相同的命令行参数。"""
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Test ABRLLM and save recorded ExperiencePool to pickle (same format as training exp_pool).",
    )

    parser.add_argument(
        "--exp-pool-path",
        help="经验池 pkl，用于 ExperienceDataset 的 reward 边界与 target_return（与训练时一致）",
        default="artifacts/exp_pools/exp_pool.pkl",
    )
    parser.add_argument("--sample-step", type=int, help="the steps for sampling experiences", default=None)

    parser.add_argument("--trace", type=str, default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", type=str, default="video1")
    parser.add_argument("--fixed-order", action="store_true")

    parser.add_argument("--plm-type", type=str, default="llama")
    parser.add_argument("--plm-size", type=str, default="base")
    parser.add_argument("--rank", type=int, default=-1)

    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--state-embedding-dim", type=int, default=None)

    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--state-use-self-attention", action="store_true")
    parser.add_argument("--state-attn-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--fusion-method",
        type=str,
        choices=["weighted_sum", "mean", "concat", "mamba"],
        default="weighted_sum",
    )

    parser.add_argument("--w", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--eval-per-epoch", type=int, default=1)
    parser.add_argument("--save-checkpoint-per-epoch", type=int, default=10)
    parser.add_argument("--target-return-scale", type=float, default=1.0)
    parser.add_argument("--which-layer", type=int, default=-1)

    parser.add_argument("--grad-accum-steps", dest="grad_accum_steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="权重目录（含 LoRA 或 model.bin）；默认与 run_abr 一致，指向根据 exp-pool 与超参推断的 best_model",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-out", dest="device_out", default=None)
    parser.add_argument("--device-mid", dest="device_mid", default=None)

    parser.add_argument(
        "--exp-pool-output",
        type=str,
        default=None,
        help=f"输出的测试经验池 .pkl 路径；默认 {DEFAULT_RECORD_EXP_POOL_DIR}/{DEFAULT_RECORD_EXP_POOL_NAME}",
    )

    parser.set_defaults(frozen=True)
    parser.set_defaults(state_use_self_attention=True)

    return parser


def normalize_args(args):
    if args.state_embedding_dim is None:
        args.state_embedding_dim = args.state_feature_dim
    if args.state_attn_hidden_dim is None:
        args.state_attn_hidden_dim = args.state_embedding_dim
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.device_out is None:
        args.device_out = args.device
    args.max_length = args.w
    return args


def discover_trace_datasets(script_dir=None):
    """
    扫描 data/traces/{train,valid,test} 下每个子目录为一个 trace 数据集。
    若不同 split 下出现同名文件夹，文件名前缀使用 {split}_{name} 区分。

    Returns:
        list of (trace_key, trace_dir_abspath)
    """
    if script_dir is None:
        script_dir = PROJECT_ROOT
    root = os.path.join(script_dir, "data", "traces")
    entries = []
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            sub = os.path.join(split_dir, name)
            if os.path.isdir(sub):
                entries.append((split, name, os.path.abspath(sub)))

    if not entries:
        return []

    name_counts = Counter(n for _, n, _ in entries)
    out = []
    for split, name, path in entries:
        trace_key = name if name_counts[name] == 1 else f"{split}_{name}"
        out.append((trace_key, path))
    return out


def load_exp_dataset_info(args):
    path = args.exp_pool_path
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    with open(path, "rb") as f:
        exp_pool = pickle.load(f)
    exp_dataset = ExperienceDataset(
        exp_pool,
        gamma=args.gamma,
        scale=args.scale,
        max_length=args.w,
        sample_step=args.sample_step,
    )
    return Munch(exp_dataset.exp_dataset_info)


def resolve_model_dir(args):
    train_exp_pool_info = args.exp_pool_path.split("/")[-4:-1]
    train_exp_pool_info = "_".join(train_exp_pool_info)

    models_dir = os.path.join(
        cfg.plm_ft_dir,
        f"{args.plm_type}_{args.plm_size}",
        train_exp_pool_info + f"_ss_{args.sample_step}",
        f"abrllm_rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_sfd_{args.state_feature_dim}"
        f"_sattn_{args.state_use_self_attention}_sahd_{args.state_attn_hidden_dim}_fusion_{args.fusion_method}"
        f"_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_epochs_{args.num_epochs}_seed_{args.seed}",
    )
    best_model_dir = os.path.join(models_dir, "best_model")
    return args.model_dir if args.model_dir is not None else best_model_dir


def run_single_record_test(
    args,
    exp_dataset_info,
    abrllm_model,
    trace_key,
    trace_dir,
    video_key,
    video_size_dir,
    exp_pool_output_path,
):
    """
    执行一次测试并写入 pkl。不调用 load_model（由调用方在首次测试前加载权重）。

    trace_key / video_key：用于 results 子目录命名；trace_dir / video_size_dir：实际磁盘路径。
    """
    set_random_seed(args.seed)

    all_cooked_time, all_cooked_bw, all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)
    trace_num = min(args.trace_num, len(all_file_names))
    if args.trace_num == -1:
        trace_num = len(all_file_names)
    if trace_num == len(all_file_names):
        fixed_order = True
    else:
        fixed_order = args.fixed_order

    env_settings = {
        "all_cooked_time": all_cooked_time,
        "all_cooked_bw": all_cooked_bw,
        "all_file_names": all_file_names,
        "all_mahimahi_ptrs": all_mahimahi_ptrs,
        "video_size_dir": video_size_dir,
        "fixed": fixed_order,
        "trace_num": trace_num,
    }

    results_dir = os.path.join(
        cfg.results_dir,
        f"{trace_key}_{video_key}",
        f"trace_num_{trace_num}_fixed_{fixed_order}",
        f"{args.plm_type}_{args.plm_size}",
        f"abrllm_rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_tgt_scale_{args.target_return_scale}_seed_{args.seed}",
    )

    os.makedirs(DEFAULT_RECORD_EXP_POOL_DIR, exist_ok=True)
    parent = os.path.dirname(os.path.abspath(exp_pool_output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    def process_reward(
        reward,
        max_reward=exp_dataset_info.max_reward,
        min_reward=exp_dataset_info.min_reward,
        scale=args.scale,
    ):
        reward = min(max_reward, max(min_reward, reward))
        return (reward - min_reward) / (max_reward - min_reward) / scale

    target_return = exp_dataset_info.max_return * args.target_return_scale

    return test_on_env_with_exp_pool(
        args,
        abrllm_model,
        results_dir,
        env_settings,
        target_return,
        exp_pool_path=exp_pool_output_path,
        max_ep_num=trace_num,
        process_reward_fn=process_reward,
        seed=args.seed,
    )


def build_model_and_load(args):
    """创建 ABRLLM 并从 model_dir 加载权重。返回 (model, model_dir)。"""
    model_path = os.path.join(cfg.plm_dir, args.plm_type, args.plm_size)
    args.model_path = model_path
    args.llm_dim = cfg.plm_embed_sizes[args.plm_type][args.plm_size]

    from plm_special.models.low_rank import peft_model
    from ABRLLM_v2 import ABRLLM

    abrllm_model = ABRLLM(args)
    abrllm_model.device = torch.device(args.device)
    abrllm_model = abrllm_model.to(args.device)
    if args.rank != -1:
        abrllm_model.plm = peft_model(abrllm_model.plm, args.plm_type, rank=args.rank)

    model_dir = resolve_model_dir(args)
    assert os.path.exists(model_dir), f"Model weight dir does not exist: {model_dir}"
    abrllm_model = load_model(args, abrllm_model, model_dir)
    return abrllm_model, model_dir
