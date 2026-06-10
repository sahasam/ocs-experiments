#!/usr/bin/env bash
# Toy training run: 4 ranks via torchrun, gloo backend (CPU only).
# macOS note: --standalone tries to bind to ::1 and clients can't reach it,
# so we use explicit IPv4 rdzv endpoint instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p traces
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo0}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec .venv/bin/torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC:-4}" \
  --rdzv-backend=static \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  --node-rank=0 \
  src/train.py \
  --config configs/toy_model.yaml \
  --output-dir traces/
