#!/usr/bin/env bash
# 绘制 QoE 及分量分组柱状图（需各算法已跑完测试）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/amora/anaconda3/envs/netllm/bin/python}"
TRACE="${TRACE:-fcc-test}"
VIDEO="${VIDEO:-video2}"
TRACE_NUM="${TRACE_NUM:-100}"
SEED="${SEED:-666}"
ROUNDS="${ROUNDS:-1}"
BAR_ALGOS="${BAR_ALGOS:-ours,BOLA,RobustMPC,Pensieve,MERINA,Comyco,Oracle}"
OURS_DIR="${OURS_DIR:-artifacts/results/${TRACE}_${VIDEO}/trace_num_${TRACE_NUM}_fixed_True/llama_base/abrllm_v2_rank_128_w_20_gamma_1.0_tgt_scale_1.0_seed_${SEED}}"

"${PYTHON}" run/plot_qoe_components_bar.py \
  --preset netllm_baselines \
  --trace "${TRACE}" \
  --video "${VIDEO}" \
  --trace-num "${TRACE_NUM}" \
  --seed "${SEED}" \
  --test-rounds "${ROUNDS}" \
  --ours-dir "${OURS_DIR}" \
  --bar-algos "${BAR_ALGOS}" \
  --output "artifacts/figures/qoe_components_bar_${TRACE}_${VIDEO}.png" \
  --stats-json "artifacts/figures/qoe_components_bar_${TRACE}_${VIDEO}.json" \
  "$@"
