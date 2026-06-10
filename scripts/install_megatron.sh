#!/usr/bin/env bash
# Install Megatron-LM into the project venv.
# Mirrors install_chakra.sh: vendor the repo under third_party/, install --no-deps.
# Megatron's full install pulls in apex, transformer-engine, tensorboard, etc.;
# we only need megatron.core importable from inside pretrain_gpt.py.
# On real GPU nodes you will need transformer_engine separately for the H100
# fused kernels; that lives in the H100 base image, not in this scaffold.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d third_party/megatron ]; then
  mkdir -p third_party
  git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git third_party/megatron
fi

.venv/bin/pip install --no-deps -e ./third_party/megatron

echo
echo "Verifying megatron.core is importable..."
.venv/bin/python -c "import megatron.core; print('  megatron.core:', megatron.core.__file__)"
