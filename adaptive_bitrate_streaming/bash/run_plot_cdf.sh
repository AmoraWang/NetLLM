#!/usr/bin/env bash
# 绘制测试集 QoE CDF（需各算法已跑完测试并写出 result_sim_abr_* / log_test_* 日志）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/amora/anaconda3/envs/netllm/bin/python}"
TRACE="${TRACE:-Norway3G-test}"
VIDEO="${VIDEO:-video2}"
TRACE_NUM="${TRACE_NUM:-41}"
SEED="${SEED:-666}"
ROUNDS="${ROUNDS:-1}"
OURS_DIR="${OURS_DIR:-artifacts/results/${TRACE}_${VIDEO}/trace_num_${TRACE_NUM}_fixed_True/llama_base/abrllm_v2_rank_128_w_20_gamma_1.0_tgt_scale_1.0_seed_${SEED}}"

"${PYTHON}" run/plot_qoe_cdf.py \
  --preset netllm_baselines \
  --trace "${TRACE}" \
  --video "${VIDEO}" \
  --trace-num "${TRACE_NUM}" \
  --seed "${SEED}" \
  --test-rounds "${ROUNDS}" \
  --ours-dir "${OURS_DIR}" \
  --output "artifacts/figures/qoe_cdf_${TRACE}_${VIDEO}.png" \
  --stats-json "artifacts/figures/qoe_cdf_${TRACE}_${VIDEO}.json" \
  "$@"
