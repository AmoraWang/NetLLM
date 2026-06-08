#!/usr/bin/env bash
# 绘制测试集 QoE CDF（默认从 collect 输出的 .pt 读取，含 Ours）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/amora/anaconda3/envs/netllm/bin/python}"
TRACE="${TRACE:-fcc-test}"
VIDEO="${VIDEO:-video1}"
TRACE_NUM="${TRACE_NUM:-100}"
SEED="${SEED:-666}"
ROUNDS="${ROUNDS:-1}"
QOE_PT_DIR="${QOE_PT_DIR:-artifacts/qoe_cdf_columns/${VIDEO}}"
OURS_DIR="${OURS_DIR:-artifacts/results/${TRACE}_${VIDEO}/trace_num_${TRACE_NUM}_fixed_True/llama_base/abrllm_v2_rank_128_w_20_gamma_1.0_tgt_scale_1.0_seed_${SEED}}"

"${PYTHON}" run/plot_qoe_cdf.py \
  --qoe-pt-dir "${QOE_PT_DIR}" \
  --trace "${TRACE}" \
  --video "${VIDEO}" \
  --trace-num "${TRACE_NUM}" \
  --seed "${SEED}" \
  --test-rounds "${ROUNDS}" \
  --ours-dir "${OURS_DIR}" \
  --output "artifacts/figures/qoe_cdf_${TRACE}_${VIDEO}.png" \
  --stats-json "artifacts/figures/qoe_cdf_${TRACE}_${VIDEO}.json" \
  "$@"
