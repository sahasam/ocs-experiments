#!/usr/bin/env bash
# Generate AstraSim-compatible Chakra ETs for a distributed LLM workload using
# STAGE (Symbolic Tensor Graph, astra-sim/symbolic_tensor_graph). Pure-CPU,
# no GPU cluster needed. Output lands in the same layout the existing
# run_astrasim.sh expects:  results/<name>/<name>.<rank>.et  (+ comm_group.json).
#
# Unlike generate_et.sh (which writes our own hand-rolled Text workload with the
# MHA/no-GQA + no-embedding approximations), STAGE emits the real op DAG with
# GQA, embeddings, TP/SP all-gather + reduce-scatter, DP all-reduce, and PP
# send/recv -- validated to 0.23% comm-volume error vs a real 128-GPU H100 run.
#
# Usage (all overridable via env):
#   bash generate_stage_et.sh                       # 8B, TP=8 PP=2 DP=4 (64 ranks)
#   DP=8 TP=1 PP=1 bash generate_stage_et.sh        # pure DP=8
#   MODEL=70b TP=8 PP=2 DP=4 bash generate_stage_et.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAGE_DIR="${STAGE_DIR:-$REPO_ROOT/third_party/STAGE}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

# Parallelism degrees (num NPUs = DP*TP*PP*SP).
DP=${DP:-4}
TP=${TP:-8}
PP=${PP:-2}
SP=${SP:-1}

# Model preset: 8b or 70b (Llama-3 shapes). Override individual dims below.
MODEL=${MODEL:-8b}
if [[ "$MODEL" == "8b" ]]; then
  DMODEL=${DMODEL:-4096};  DFF=${DFF:-14336}; HEAD=${HEAD:-32}
  KVHEAD=${KVHEAD:-8};     STACKS=${STACKS:-32}
elif [[ "$MODEL" == "70b" ]]; then
  DMODEL=${DMODEL:-8192};  DFF=${DFF:-28672}; HEAD=${HEAD:-64}
  KVHEAD=${KVHEAD:-8};     STACKS=${STACKS:-80}
else
  echo "unknown MODEL=$MODEL (use 8b or 70b, or set DMODEL/DFF/HEAD/KVHEAD/STACKS)" >&2
  exit 2
fi
SEQ=${SEQ:-8192}
BATCH=${BATCH:-64}
DVOCAL=${DVOCAL:-128256}     # Llama-3 vocab (GQA-aware embedding sizing)
WEIGHT_SHARDED=${WEIGHT_SHARDED:-false}

NPUS=$(( DP * TP * PP * SP ))
WORKLOAD_NAME=${WORKLOAD_NAME:-llama3_${MODEL}_tp${TP}_pp${PP}_dp${DP}}
# Absolute path: we `cd` into STAGE_DIR before invoking main.py, so a relative
# --output_dir would resolve under STAGE's tree instead of results/.
OUT_DIR="$SCRIPT_DIR/results/${WORKLOAD_NAME}"
mkdir -p "$OUT_DIR"

echo "STAGE generation -> $OUT_DIR/"
echo "  model=$MODEL  dmodel=$DMODEL dff=$DFF head=$HEAD kvhead=$KVHEAD stacks=$STACKS seq=$SEQ batch=$BATCH"
echo "  parallelism: DP=$DP TP=$TP PP=$PP SP=$SP  => $NPUS ranks"

# STAGE hardcodes /dev/shm for scratch (Linux tmpfs); on macOS fall back to the
# system temp dir. graph.py honors STAGE_TMP_DIR.
export STAGE_TMP_DIR="${STAGE_TMP_DIR:-$(dirname "$(mktemp -u)")}"

( cd "$STAGE_DIR" && "$PYTHON_BIN" main.py \
    --output_dir "$OUT_DIR" \
    --output_name "${WORKLOAD_NAME}.%d.et" \
    --model_type llama \
    --dp "$DP" --tp "$TP" --pp "$PP" --sp "$SP" \
    --dmodel "$DMODEL" --dff "$DFF" --head "$HEAD" --kvhead "$KVHEAD" \
    --num_stacks "$STACKS" --seq "$SEQ" --batch "$BATCH" --dvocal "$DVOCAL" \
    --weight_sharded "$WEIGHT_SHARDED" \
    --chakra_schema_version v0.0.4 )

# STAGE writes the comm-group file as <name>.json (derived from output_name);
# normalize to comm_group.json so run_astrasim_stage.sh finds it predictably.
if [[ -f "$OUT_DIR/${WORKLOAD_NAME}.json" ]]; then
  mv "$OUT_DIR/${WORKLOAD_NAME}.json" "$OUT_DIR/comm_group.json"
fi

echo "Generated $(ls "$OUT_DIR"/*.et | wc -l | tr -d ' ') ET files + comm_group.json"
ls -lh "$OUT_DIR/" | head
