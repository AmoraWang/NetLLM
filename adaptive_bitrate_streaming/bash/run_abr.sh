#!/usr/bin/env bash
# 在 adaptive_bitrate_streaming 包根目录执行；也可从任意路径：bash bash/run_abr.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# python run/run_abr.py \
#    \
#   --test \
#   --loss-type ce_kl \
#   --frozen \
#   --state-use-self-attention \
#   --grad-accum-steps 32 \
#   --seed 666 \
#   --plm-type llama \
#   --plm-size base \
#   --rank 128 \
#   --device cuda:0 \
#   --state-feature-dim 512 \
#   --w 20 \
#   --gamma 1. \
#   --lr 5e-5 \
#   --warmup-steps 300 \
#   --num-epochs 60 \
#   --eval-per-epoch 2 \
#   --target-return-scale 1 \
#   --save-checkpoint-per-epoch 50 \
#   --state-attn-hidden-dim 2048 \
#   --test-rounds 1 \
#   --video video1 \
#   --trace fcc-train \
#   --sample-step 15 \

  python run/run_abr.py \
  --adapt \
  --abr-llm-version v2 \
  --test \
  --loss-type ce_kl \
  --kd-alpha 0.5 \
  --kd-temperature 2.0 \
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
  --warmup-steps 300 \
  --num-epochs 60 \
  --eval-per-epoch 2 \
  --target-return-scale 1 \
  --save-checkpoint-per-epoch 50 \
  --state-attn-hidden-dim 2048 \
  --test-rounds 1 \
  --video video1 \
  --trace fcc-test \
  --sample-step 15 \
