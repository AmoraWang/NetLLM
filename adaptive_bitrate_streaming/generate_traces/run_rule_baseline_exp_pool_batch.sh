#!/usr/bin/env bash
#规则/基线（MPC、BBA、genet 等）批量经验池 → wm_traces/{算法名}/*.pkl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python generate_traces/run_rule_baseline_exp_pool_batch.py \
  --algorithms mpc bba \
  --trace-num -1 \
  --seed 100003 \
  --cuda-id 0 \
  --fixed-order
