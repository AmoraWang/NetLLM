"""
本目录内脚本须最先 import本模块：将工作目录设为 adaptive_bitrate_streaming 根目录，
并把该根目录加入 sys.path，以便 config / data / run_abr 等与仓库布局一致。
"""
import os
import sys

_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_GEN_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)
