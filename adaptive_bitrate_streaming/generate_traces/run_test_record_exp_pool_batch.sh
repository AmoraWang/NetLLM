#!/usr/bin/env bash
# 在 adaptive_bitrate_streaming 根目录下执行
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python generate_traces/run_test_record_exp_pool_batch.py \
  --frozen \
  --state-use-self-attention \
  --grad-accum-steps 32 \
  --seed 666 \
  --plm-type llama \
  --plm-size large \
  --rank 128 \
  --device cuda:0 \
  --state-feature-dim 256 \
  --w 20 \
  --gamma 1. \
  --lr 1e-4 \
  --warmup-steps 2000 \
  --num-epochs 78 \
  --eval-per-epoch 2 \
  --target-return-scale 1 \
  --save-checkpoint-per-epoch 78 \
  --state-attn-hidden-dim 3072 \
  --exp-pool-path artifacts/exp_pools/exp_pool.pkl
