#!/usr/bin/env bash
# 在 adaptive_bitrate_streaming 根目录下执行
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python generate_traces/run_record_random_exp_pool_seeds.py \
  --seed-start 2 \
  --seed-end 60 \
  --trace-num -1 \
  --device cpu
