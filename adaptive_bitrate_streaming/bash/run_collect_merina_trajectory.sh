#!/usr/bin/env bash
# Merina 决策轨迹 → 每 trace 一个 TensorDict .pt（buffer / action / VAE belief）
#
# 环境变量（可选）：
#   TRACES=fcc-train  VIDEO=video1  SEED=666  MAX_TRACES=10  DEVICE=cpu
#   OUT=data/traces/train/merina_trajectories
#
# 在 adaptive_bitrate_streaming 根目录：
#   bash bash/run_collect_merina_trajectory.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TRACES="${TRACES:-fcc-test}"
VIDEO="${VIDEO:-video1}"
SEED="${SEED:-666}"
OUT="${OUT:-artifacts/traces/merina_trajectories}"
MAX_TRACES="${MAX_TRACES:--1}"
START_INDEX="${START_INDEX:-0}"

EXTRA=()
if [[ -n "${DEVICE:-}" ]]; then
  if [[ "${DEVICE}" == "cpu" ]]; then
    EXTRA+=(--cpu)
  else
    EXTRA+=(--cuda-id 0)
  fi
fi
if [[ "${RANDOM_MAHIMAHI:-1}" == "0" ]]; then
  EXTRA+=(--no-random-mahimahi-start --fixed-order)
fi

python run/collect_merina_trajectory_tensordict.py \
  --traces "${TRACES}" \
  --video "${VIDEO}" \
  --output-dir "${OUT}" \
  --seed "${SEED}" \
  --max-traces "${MAX_TRACES}" \
  --start-index "${START_INDEX}" \
  "${EXTRA[@]}" \
  "$@"
