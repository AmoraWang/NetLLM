"""adaptive_bitrate_streaming 包根路径（run/、bash/、generate_exp_pool/ 的上级目录）。"""
from __future__ import annotations

import os
import sys

# 本文件所在目录 = adaptive_bitrate_streaming/
ABR_ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(ABR_ROOT, "run")
BASH_DIR = os.path.join(ABR_ROOT, "bash")
GENERATE_EXP_POOL_DIR = os.path.join(ABR_ROOT, "generate_exp_pool")
MERINA_DIR = os.path.join(ABR_ROOT, "merina")
REPO_ROOT = os.path.dirname(ABR_ROOT)
PLM_DIR = os.path.join(REPO_ROOT, "downloaded_plms")


def setup_sys_path(*, include_generate_exp_pool: bool = False) -> str:
    """将包根目录（及可选 generate_exp_pool/）加入 sys.path，便于子目录脚本 import。"""
    if ABR_ROOT not in sys.path:
        sys.path.insert(0, ABR_ROOT)
    if include_generate_exp_pool and GENERATE_EXP_POOL_DIR not in sys.path:
        sys.path.insert(0, GENERATE_EXP_POOL_DIR)
    return ABR_ROOT


def join_abr(*parts: str) -> str:
    return os.path.join(ABR_ROOT, *parts)
