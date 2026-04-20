#!/usr/bin/env bash
# DP Oracle：所有 trace 数据集 × video1–3 → wm_traces/DP/*.pkl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python generate_traces/run_dp_oracle_exp_pool_batch.py \
  --seed 100003 \
  --trace-limit -1
