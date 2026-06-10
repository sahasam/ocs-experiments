#!/usr/bin/env bash
# Run AstraSim on the Llama-3 70B chakra ETs produced by Megatron + chakra_converter.
# Topology: 8 nodes x 8 GPUs, two-tier (intra-node NVLink ring + inter-node IB switch).
# Expects traces/chakra_workload.{0..63}.et to exist; produce them with:
#   bash scripts/launch_megatron.sh        # on the GPU cluster
#   NRANKS=64 bash scripts/convert_to_chakra.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASTRA_SIM_DIR="${ASTRA_SIM_DIR:-/Users/sahas/workplace/astra-sim}"
ASTRA_IMG="${ASTRA_IMG:-astra-sim:latest}"

if ! ls "$ROOT/traces/chakra_workload."*.et >/dev/null 2>&1; then
  echo "ERROR: no chakra_workload.*.et found. Run scripts/launch_megatron.sh then scripts/convert_to_chakra.sh first." >&2
  exit 1
fi

docker run --rm \
  -v "$ASTRA_SIM_DIR":/app/astra-sim \
  -v "$ROOT":/app/llm-parallelism \
  "$ASTRA_IMG" \
  /app/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware \
    --workload-configuration=/app/llm-parallelism/traces/chakra_workload \
    --system-configuration=/app/llm-parallelism/astrasim/llama70b/system.json \
    --network-configuration=/app/llm-parallelism/astrasim/llama70b/network.yml \
    --remote-memory-configuration=/app/llm-parallelism/astrasim/llama70b/no_memory_expansion.json
