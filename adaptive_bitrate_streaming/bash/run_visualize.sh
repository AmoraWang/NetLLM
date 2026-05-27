#!/usr/bin/env bash
# 在 adaptive_bitrate_streaming 包根执行：bash bash/run_visualize.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python run/visualize_alignment.py \
  --model-dir data/ft_plms/llama_base/artifacts_exp_pools_ss_15/abrllm_v2_rank_128_w_20_gamma_1.0_sfd_512_sattn_True_sahd_2048_fusion_weighted_sum_loss_ce_kl_kdalpha_0.5_kdtemp_2.0_lr_5e-05_wd_0.0001_warm_300_epochs_60_seed_666/best_model \
  --exp-pool-path artifacts/exp_pools/exp_pool.pkl \
  --sample-index 0 \
  --w 20 \
  --rank 128 \
  --state-feature-dim 512 \
  --state-attn-hidden-dim 2048 \
  --output-dir artifacts/alignment_viz/sample0
