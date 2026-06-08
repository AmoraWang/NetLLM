#!/usr/bin/env bash
# 四项对齐深化分析：多样本 / 遮挡 / 随机vs训练 / 词云
# v3 checkpoint 必须传 --abr-llm-version v3 及与训练一致的结构超参。
# 在 adaptive_bitrate_streaming 包根执行：bash bash/run_alignment_suite.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-data/ft_plms/llama_base/artifacts_exp_pools_ss_15/abrllm_v3_rank_128_w_20_gamma_1.0_sfd_256_sattn_True_sahd_2048_fusion_weighted_sum_loss_ce_kl_kdalpha_0.9_kdtemp_4.0_align_0.1_temp_0.07_cdim_256_lr_0.0001_wd_0.0001_warm_300_epochs_60_seed_666/best_model}"
OUT_DIR="${OUT_DIR:-artifacts/alignment_viz/v3_trained}"

"${PYTHON}" run/alignment_analysis_suite.py \
  --abr-llm-version v3 \
  --contrast-dim 256 \
  --align-lambda 0.1 \
  --state-feature-dim 256 \
  --state-attn-hidden-dim 2048 \
  --sample-step 15 \
  --model-dir "${MODEL_DIR}" \
  --exp-pool-path artifacts/exp_pools/exp_pool.pkl \
  --sample-index 0 \
  --output-dir "${OUT_DIR}"
