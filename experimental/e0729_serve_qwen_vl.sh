#!/usr/bin/env bash
# Serve Qwen2.5-VL-3B-Instruct with vLLM for the agentflow integration tests.
#
# The example configs under examples/*/configs/ point at http://0.0.0.0:8010/v1.
# On pre-Ampere GPUs (e.g. Quadro RTX 6000, sm_75) vLLM's default FlashInfer
# backend fails to JIT-build its prefill kernels, so we pin TRITON_ATTN and
# force fp16 (no bfloat16 on Turing).
#
# Usage:
#   bash experimental/e0729_serve_qwen_vl.sh            # GPU 2, port 8010
#   GPU=1 PORT=8020 bash experimental/e0729_serve_qwen_vl.sh

set -euo pipefail

GPU="${GPU:-2}"
PORT="${PORT:-8010}"
LOG="${LOG:-/tmp/vllm_qwenvl_${PORT}.log}"
VLLM_BIN="${VLLM_BIN:-/home/khoi/miniconda3/envs/vllm/bin/vllm}"

CUDA_VISIBLE_DEVICES="$GPU" nohup "$VLLM_BIN" serve Qwen/Qwen2.5-VL-3B-Instruct \
  --host 0.0.0.0 --port "$PORT" \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --attention-backend TRITON_ATTN \
  > "$LOG" 2>&1 &

echo "vllm pid=$! gpu=$GPU port=$PORT log=$LOG"
