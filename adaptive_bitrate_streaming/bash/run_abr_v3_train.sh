#!/usr/bin/env bash
# 训练 ABRLLM_v3（状态 Merina 行序 11x6，对应 ABRLLM_v3.py）。
# 请先使用 v3 经验池，例如：
#   python generate_exp_pool/generate_exp_pool.py --models genet --trace fcc-train --video video1 \
#       --trace-num -1 --seed 100003 --abr-llm-version v3 --cuda-id 0
# 然后将下方 --exp-pool-path 改为实际生成的 .../abr_v3/exp_pool.pkl（目录名随 models/trace/seed 变化）。
#
# 在 adaptive_bitrate_streaming 包根执行：bash bash/run_abr_v3_train.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python run/run_abr.py \
  --test \
  --abr-llm-version v3 \
  --exp-pool-path artifacts/exp_pools/exp_pool.pkl \
  --trace fcc-test \
  --video video1 \
  --trace-num 100 \
  --frozen \
  --state-use-self-attention \
  --grad-accum-steps 32 \
  --seed 666 \
  --plm-type llama \
  --plm-size base \
  --rank 128 \
  --device cuda:0 \
  --state-feature-dim 256 \
  --w 20 \
  --gamma 1. \
  --lr 0.0001 \
  --warmup-steps 2000 \
  --num-epochs 50 \
  --eval-per-epoch 2 \
  --target-return-scale 1 \
  --save-checkpoint-per-epoch 40 \
  --state-attn-hidden-dim 2048
