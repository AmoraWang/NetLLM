#!/usr/bin/env bash
# 在 adaptive_bitrate_streaming根目录下执行 Python（generate_traces 内脚本依赖 cwd=项目根）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python generate_traces/run_test_record_exp_pool.py \
  --frozen \
  --state-use-self-attention \
  --grad-accum-steps 32 \
  --seed 666 \
  --plm-type llama \
  --plm-size base \
  --rank 128 \
  --device cuda:0 \
  --state-feature-dim 512 \
  --w 20 \
  --gamma 1. \
  --lr 5e-5 \
  --warmup-steps 2000 \
  --num-epochs 70 \
  --eval-per-epoch 2 \
  --target-return-scale 1 \
  --save-checkpoint-per-epoch 40 \
  --state-attn-hidden-dim 2048 \
  --exp-pool-path artifacts/exp_pools/exp_pool.pkl
