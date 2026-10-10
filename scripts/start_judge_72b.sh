#!/usr/bin/env bash
# Run in a separate vLLM environment on dedicated judge GPUs.
set -Eeuo pipefail
: "${JUDGE_MODEL_DIR:?set the local Qwen2.5-72B-Instruct checkpoint directory}"
[[ -s "$JUDGE_MODEL_DIR/config.json" && -s "$JUDGE_MODEL_DIR/model.safetensors.index.json" ]]
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_USE_V1=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL_DIR" \
    --served-model-name Qwen2.5-72B-Instruct \
    --host "${JUDGE_HOST:-0.0.0.0}" --port "${JUDGE_PORT:-18901}" \
    --tensor-parallel-size 4 --dtype bfloat16 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 32 \
    --disable-log-requests
