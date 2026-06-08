#!/usr/bin/env bash
# 收集 CDF / 柱状图用 per-trace mean QoE → 各 (测试集, 算法) 一个 .pt TensorDict
# 含 baseline (mpc/bba/...) 与 Ours (ours → fcc-test_ours.pt)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/amora/anaconda3/envs/netllm/bin/python}"
VIDEO="${VIDEO:-video1}"
OUT="${OUT:-artifacts/qoe_cdf_columns/${VIDEO}}"
SEED="${SEED:-666}"

# 指定测试集（带宽轨迹集合，非单条 trace 文件）：
#   TRACE=fcc-test bash bash/run_collect_qoe_cdf_columns.sh
#   TRACES=fcc-test,norway3G-test bash bash/run_collect_qoe_cdf_columns.sh
# 未设置 TRACE/TRACES 时，收集全部默认测试集（fcc / norway3G / hsr / oboe）
TRACE_ARGS=()
if [[ -n "${TRACES:-}" ]]; then
  TRACE_ARGS=(--traces "${TRACES}")
elif [[ -n "${TRACE:-}" ]]; then
  TRACE_ARGS=(--traces "${TRACE}")
fi

# 仅检查缺失实验：
#   CHECK=1 bash bash/run_collect_qoe_cdf_columns.sh
# 不收集 Ours：
#   NO_OURS=1 bash bash/run_collect_qoe_cdf_columns.sh
if [[ "${CHECK:-0}" == "1" ]]; then
  EXTRA=(--check-only)
else
  EXTRA=()
fi
if [[ "${NO_OURS:-0}" == "1" ]]; then
  EXTRA+=(--no-ours)
fi

"${PYTHON}" run/collect_qoe_cdf_columns.py \
  --video "${VIDEO}" \
  "${TRACE_ARGS[@]}" \
  --trace-num -1 \
  --seed "${SEED}" \
  --test-rounds 1 \
  --fixed-order \
  --abr-llm-version "${ABR_LLM_VERSION:-v2}" \
  --rank "${RANK:-128}" \
  --w "${W:-20}" \
  --gamma "${GAMMA:-1.0}" \
  --target-return-scale "${TARGET_RETURN_SCALE:-1.0}" \
  --output-dir "${OUT}" \
  "${EXTRA[@]}" \
  "$@"

echo "结果目录: ${OUT}"
echo "Ours 示例: ${OUT}/fcc-test_ours.pt"
echo "读取示例:"
echo "  from run.qoe_tensordict_store import load_qoe_result_pt"
echo "  td = load_qoe_result_pt('${OUT}/fcc-test_mpc.pt')"
