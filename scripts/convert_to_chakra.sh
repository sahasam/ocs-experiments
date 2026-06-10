#!/usr/bin/env bash
# Convert per-rank ExecutionTrace JSONs into Chakra protobuf ET files.
#
# Why no chakra_trace_link: trace_link is for merging the host execution
# trace with a GPU device trace (Kineto with CUDA events). On CPU there
# is no separate device trace, and chakra_converter accepts the raw
# ExecutionTraceObserver JSON directly. The Kineto JSONs we generate
# alongside are only used by tools/inspect_kineto.py for human inspection.
#
# Output naming matches AstraSim's discovery pattern: chakra_workload.<rank>.et
# AstraSim is invoked as: AstraSim --workload-configuration=traces/chakra_workload
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NRANKS="${NRANKS:-4}"
PREFIX="${PREFIX:-traces/chakra_workload}"

for ((rank = 0; rank < NRANKS; rank++)); do
  src="traces/et_rank${rank}.json"
  dst="${PREFIX}.${rank}.et"
  if [ ! -f "$src" ]; then
    echo "ERROR: missing $src — run scripts/launch_train.sh first" >&2
    exit 1
  fi
  echo "[rank ${rank}] $src -> $dst"
  .venv/bin/chakra_converter \
    --log-filename "/tmp/chakra_converter_rank${rank}.log" \
    PyTorch \
    --input "$src" \
    --output "$dst"
done

echo
echo "Done. Produced:"
ls -la "${PREFIX}."*.et
