#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# conda activate netllm

# python run/run_rule_baselines_test.py \
#   --algorithm bba \
#   --trace SolisWiFi-test \
#   --video video1 \
#   --trace-num -1 \
#   --seed 666 \
#   --test-rounds 10

#   python run/run_rule_baselines_test.py \
#   --algorithm mpc \
#   --trace SolisWiFi-test \
#   --video video1 \
#   --trace-num -1 \
#   --seed 666 \
#   --test-rounds 10

# python run/run_merina_baseline_test.py \
#   --trace Lumos5G-test \
#   --video video1 \
#   --trace-num -1 \
#   --seed 666 \
#   --test-rounds 10 \
#   --cuda-id 0

# conda activate tensorflowv1

python run/run_rule_baselines_test.py \
  --algorithm genet \
  --trace SolisWiFi-test \
  --video video1 \
  --trace-num -1 \
  --seed 666 \
  --test-rounds 10 \
  --cuda-id 0

python run/run_comyco_merina_baseline_test.py \
  --trace SolisWiFi-test \
  --video video1 \
  --trace-num -1 \
  --seed 666 \
  --test-rounds 10 \
  --cuda-id 0

# python generate_exp_pool/DP.py \
#   --mode test \
#   --trace Lumos5G-test \
#   --video video1 \
#   --trace-num -1 \
#   --seed 666 \
#   --test-rounds 1