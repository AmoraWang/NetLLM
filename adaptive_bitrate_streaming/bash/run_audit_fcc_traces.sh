#!/usr/bin/env bash
# 审计 fcc-test / fcc-train / fcc-valid 轨迹在 data/traces 其余目录中的来源
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python3}"
OUT="${OUT:-artifacts/trace_provenance_fcc_audit.tsv}"

"${PYTHON}" run/audit_fcc_trace_provenance.py \
  --output "${OUT}" \
  "$@"

echo "明细: ${OUT}"
echo "汇总: ${OUT%.tsv}_summary.txt"
