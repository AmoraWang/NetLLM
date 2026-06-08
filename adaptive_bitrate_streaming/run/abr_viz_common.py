"""ABRLLM v2/v3 对齐可视化共用的模型构建与权重加载。"""
from __future__ import annotations

import os

import torch

from config import cfg
from plm_special.models.low_rank import peft_model
from run.run_abr import load_model


def patch_viz_args(args) -> None:
    """补全 visualize / alignment_suite 与 run_abr 一致的字段。"""
    if getattr(args, "sample_step", None) is None:
        args.sample_step = args.w
    if getattr(args, "device", None) is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if getattr(args, "state_embedding_dim", None) is None:
        args.state_embedding_dim = args.state_feature_dim
    if getattr(args, "state_attn_hidden_dim", None) is None:
        args.state_attn_hidden_dim = args.state_feature_dim
    args.model_path = os.path.join(cfg.plm_dir, args.plm_type, args.plm_size)
    args.llm_dim = cfg.plm_embed_sizes[args.plm_type][args.plm_size]
    args.max_length = args.w
    # v3 对比头：与训练时 --contrast-dim 一致；仅 v3 且 >0 时构建 MLP
    if getattr(args, "abr_llm_version", "v2") == "v3":
        if not hasattr(args, "align_lambda"):
            args.align_lambda = 0.1
        if not hasattr(args, "align_temperature"):
            args.align_temperature = 0.07
        if not hasattr(args, "contrast_dim"):
            args.contrast_dim = 256
    else:
        args.contrast_dim = 0


def import_abrllm_class(abr_llm_version: str):
    if abr_llm_version == "v3":
        from ABRLLM_v3 import ABRLLM

        return ABRLLM
    from ABRLLM_v2 import ABRLLM

    return ABRLLM


def build_abrllm(args, *, load_weights: bool = True, model_dir: str | None = None):
    """
    构建 ABRLLM 并可选加载 checkpoint（LoRA + modules_except_plm）。
    ``model_dir`` 默认 ``args.model_dir``。
    """
    patch_viz_args(args)
    ABRLLM = import_abrllm_class(getattr(args, "abr_llm_version", "v2"))
    model = ABRLLM(args)
    model.device = torch.device(args.device)
    model = model.to(args.device)
    if args.rank > 0:
        model.plm = peft_model(model.plm, args.plm_type, rank=args.rank)
    if load_weights:
        ckpt = model_dir or args.model_dir
        model = load_model(args, model, ckpt)
    model.eval()
    return model
