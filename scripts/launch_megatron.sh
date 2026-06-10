#!/usr/bin/env bash
# Llama-3 70B trace-capture run via Megatron-LM.
# 8 nodes x 8 H100 = 64 GPUs. TP=8 PP=2 DP=4. Profiles iteration 5 only.
#
# This script is a thin wrapper around third_party/megatron/pretrain_gpt.py.
# The src/megatron_trace_hook module is imported up-front (via PYTHONPATH +
# `python -m`) so Megatron's training_step is wrapped in capture_trace()
# for the profile window.
#
# Required env on the launching node:
#   NNODES         total node count (default: 8)
#   NODE_RANK      this node's rank in [0, NNODES) (default: 0)
#   MASTER_ADDR    rank-0 node's reachable IP (default: 127.0.0.1, single-node only)
#   MASTER_PORT    rendezvous port (default: 29500)
#   TOKENIZER_PATH path to Llama-3 tokenizer.model
#   DATA_PATH      path to pre-tokenized dataset prefix (Megatron's mmap format)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${TOKENIZER_PATH:?TOKENIZER_PATH must be set to the Llama-3 tokenizer.model path}"
: "${DATA_PATH:?DATA_PATH must be set to the pre-tokenized dataset prefix}"

mkdir -p traces
export PYTHONPATH="$ROOT:$ROOT/third_party/megatron:${PYTHONPATH:-}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Import the trace hook before Megatron's main loop starts.
# PYTHONSTARTUP runs per-interpreter (i.e. per-rank under torchrun), which is
# exactly what we want: each rank decides locally whether it's in the profile
# subset and wraps its own training_step.
export PYTHONSTARTUP="$ROOT/src/megatron_trace_hook.py"

exec .venv/bin/torchrun \
  --nnodes="${NNODES:-8}" \
  --nproc-per-node=8 \
  --node-rank="${NODE_RANK:-0}" \
  --rdzv-backend=static \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  third_party/megatron/pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 2 \
  --num-layers 80 \
  --hidden-size 8192 \
  --num-attention-heads 64 \
  --group-query-attention \
  --num-query-groups 8 \
  --ffn-hidden-size 28672 \
  --seq-length 8192 \
  --max-position-embeddings 8192 \
  --micro-batch-size 1 \
  --global-batch-size 64 \
  --train-iters 10 \
  --lr 1.0e-5 \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model "$TOKENIZER_PATH" \
  --data-path "$DATA_PATH" \
  --bf16 \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --profile \
  --profile-step-start 5 \
  --profile-step-end 6 \
  --save-interval 100000 \
  --eval-interval 100000 \
  --eval-iters 0
