#!/usr/bin/env bash
# 训练 / 测试 ABRLLM_v4（v2 + 带宽 β-VAE；需 --state-use-self-attention）。
# 经验池可与 v2 共用（6×6 状态）：artifacts/exp_pools/exp_pool.pkl
#
# 可选环境变量（覆盖下方默认）：
#   BW_VAE_FUSION=concat|residual   BW_VAE_LATENT_DIM=16
#   TRACE=fcc-train  VIDEO=video1  DEVICE=cuda:0  SEED=666
#
# 在 adaptive_bitrate_streaming 包根执行：bash bash/run_abr_v4_train.sh
# -residual 和 -concat

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BW_VAE_FUSION="${BW_VAE_FUSION:-residual}"
BW_VAE_LATENT_DIM="${BW_VAE_LATENT_DIM:-16}"
TRACE="${TRACE:-fcc-test}"
VIDEO="${VIDEO:-video1}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-666}"

python run/run_abr.py \
  --adapt \
  --test \
  --abr-llm-version v4 \
  --loss-type ce_kl \
  --kd-alpha 0.5 \
  --kd-temperature 2.0 \
  --exp-pool-path artifacts/exp_pools/merina_merged_logits.pkl \
  --trace "${TRACE}" \
  --video "${VIDEO}" \
  --trace-num 100 \
  --frozen \
  --state-use-self-attention \
  --state-attn-hidden-dim 2048 \
  --bw-vae-latent-dim "${BW_VAE_LATENT_DIM}" \
  --bw-vae-beta 0.4 \
  --bw-vae-kl-weight 0.01 \
  --bw-vae-fusion "${BW_VAE_FUSION}" \
  --grad-accum-steps 32 \
  --seed "${SEED}" \
  --plm-type llama \
  --plm-size base \
  --rank 128 \
  --device "${DEVICE}" \
  --state-feature-dim 512 \
  --w 20 \
  --gamma 1. \
  --lr 5e-5 \
  --warmup-steps 300 \
  --num-epochs 70 \
  --eval-per-epoch 2 \
  --target-return-scale 1 \
  --save-checkpoint-per-epoch 50 \
  --test-rounds 1 \
  --sample-step 15 \
