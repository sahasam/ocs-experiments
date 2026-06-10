#!/usr/bin/env bash
# Install chakra (mlcommons) into the project venv.
# Uses --no-deps to skip HolisticTraceAnalysis, which pulls in jupyterlab
# and dozens of MB of notebook-only dependencies we don't need for the
# converter. For Path C (CPU-only) we don't run chakra_trace_link, only
# chakra_converter, which works on the raw ExecutionTrace JSON directly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d third_party/chakra ]; then
  mkdir -p third_party
  git clone --depth 1 https://github.com/mlcommons/chakra.git third_party/chakra
fi

.venv/bin/pip install --no-deps ./third_party/chakra
.venv/bin/pip install protobuf

echo
echo "Verifying chakra_converter is on PATH..."
.venv/bin/chakra_converter --help > /dev/null && echo "  chakra_converter: OK"
