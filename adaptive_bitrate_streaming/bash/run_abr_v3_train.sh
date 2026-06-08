#!/usr/bin/env bash
# 训练 ABRLLM_v3（与 v2 相同的 NetLLM 6×6 状态；v3 使用语义模板，模型骨干同 v2）。
# 经验池可与 v2 共用，或单独生成到 abr_v3 目录便于区分：
#   python generate_exp_pool/generate_exp_pool.py --models genet --trace fcc-train --video video1 \
#       --trace-num -1 --seed 100003 --abr-llm-version v3 --cuda-id 0
# 将下方 --exp-pool-path 指向 .../abr_v3/exp_pool.pkl 或 abr_v2/exp_pool.pkl（形状均为 6×6）。
#
# 在 adaptive_bitrate_streaming 包根执行：bash bash/run_abr_v3_train.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python run/run_abr.py \
  --test \
  --loss-type ce_kl \
  --kd-alpha 0.9 \
  --kd-temperature 4.0 \
  --abr-llm-version v3 \
  --exp-pool-path artifacts/exp_pools/exp_pool.pkl \
  --trace fcc-test \
  --video video2 \
  --trace-num 100 \
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
  --lr 0.0001 \
  --warmup-steps 300 \
  --num-epochs 60 \
  --eval-per-epoch 2 \
  --target-return-scale 1 \
  --save-checkpoint-per-epoch 40 \
  --state-attn-hidden-dim 2048 \
  --align-lambda 0.1 \
  --align-temperature 0.07 \
  --contrast-dim 256 \
  --sample-step 15 \

# TODO: 测试kd-alpha=0.9; 测试large模型