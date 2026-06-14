#!/usr/bin/env bash
# Run a STAGE-generated workload through AstraSim's ns-3 backend (packet-level
# simulation with HPCC congestion control on a 2-tier fat-tree).
#
# This is the PS-with-congestion baseline for the OCS-vs-PS comparison.
# The OCS side uses run_astrasim_stage.sh (FullyConnected, congestion-unaware).
#
# Usage:
#   bash run_astrasim_ns3.sh <workload_name> <npus>
#   bash run_astrasim_ns3.sh llama3_8b_tp8_pp2_dp8 128
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WORKLOAD_NAME=${1:?'workload name required (subdir under results/)'}
NPUS=${2:?'num npus required'}

RESULTS_DIR="results/${WORKLOAD_NAME}"
WORKLOAD_PREFIX="/app/llm-parallelism/astrasim/synthetic/${RESULTS_DIR}/${WORKLOAD_NAME}"
COMM_GROUP="/app/llm-parallelism/astrasim/synthetic/${RESULTS_DIR}/comm_group.json"

SYSTEM_CFG=${SYSTEM_CFG:-/app/llm-parallelism/astrasim/synthetic/stage_configs/system.json}
MEMORY_CFG=${MEMORY_CFG:-/app/llm-parallelism/astrasim/synthetic/stage_configs/memory.json}

# Logical topology: flat 1D of NPUS ranks.
LOGICAL_TOPO_CFG=${LOGICAL_TOPO_CFG:-/app/llm-parallelism/astrasim/synthetic/stage_configs/ns3_logical_${NPUS}.json}

# Physical topology: 2-tier fat-tree at 400 Gbps matching the OCS link bandwidth.
NS3_TOPO_FILE=${NS3_TOPO_FILE:-/app/llm-parallelism/astrasim/synthetic/stage_configs/ns3_topo_${NPUS}_fat_400g.txt}

ASTRASIM_NS3_BIN=${ASTRASIM_NS3_BIN:-/app/astra-sim/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-optimized}
ASTRASIM_IMAGE=${ASTRASIM_IMAGE:-astra-sim-bigmem:latest}
DOCKER_SHM=${DOCKER_SHM:-4g}

LOG_DIR="${RESULTS_DIR}/logs"
# NS3_OUT_SUBDIR isolates outputs of different topologies for the same workload
# (e.g. ns3_output = fat-tree, ns3_output_clos = thin Clos) so they don't clobber.
NS3_OUT_DIR="${RESULTS_DIR}/${NS3_OUT_SUBDIR:-ns3_output}"
mkdir -p "$LOG_DIR" "$NS3_OUT_DIR"
LOG="${LOG_DIR}/${LOG_NAME:-astrasim_ns3}.log"

# ns-3 input files: flow.txt (background flows, empty = none) and trace.txt
# (node IDs to monitor at packet level, 0 = none). Use empty flow from ns-3
# source tree; generate a zero-node trace file per run.
NS3_OUT_CONTAINER="/app/llm-parallelism/astrasim/synthetic/${NS3_OUT_DIR}"
NS3_FLOW_FILE="/app/astra-sim/extern/network_backend/ns-3/scratch/output/flow.txt"
NS3_TRACE_FILE="${NS3_OUT_CONTAINER}/trace_nodes.txt"
echo "0" > "${NS3_OUT_DIR}/trace_nodes.txt"

# Generate a per-run ns-3 config.txt with absolute container paths.
NS3_CFG_LOCAL="${NS3_OUT_DIR}/ns3_config.txt"
cat > "$NS3_CFG_LOCAL" <<EOF
ENABLE_QCN 1
USE_DYNAMIC_PFC_THRESHOLD 1

PACKET_PAYLOAD_SIZE 1000

TOPOLOGY_FILE ${NS3_TOPO_FILE}
FLOW_FILE ${NS3_FLOW_FILE}
TRACE_FILE ${NS3_TRACE_FILE}
TRACE_OUTPUT_FILE ${NS3_OUT_CONTAINER}/mix.tr
FCT_OUTPUT_FILE ${NS3_OUT_CONTAINER}/fct.txt
PFC_OUTPUT_FILE ${NS3_OUT_CONTAINER}/pfc.txt
QLEN_MON_FILE ${NS3_OUT_CONTAINER}/qlen.txt
QLEN_MON_START 0
QLEN_MON_END 20000000000000

SIMULATOR_STOP_TIME 40000000000000

CC_MODE 3
ALPHA_RESUME_INTERVAL 1
RATE_DECREASE_INTERVAL 4
CLAMP_TARGET_RATE 0
RP_TIMER 900
EWMA_GAIN 0.00390625
FAST_RECOVERY_TIMES 1
RATE_AI 50Mb/s
RATE_HAI 100Mb/s
MIN_RATE 100Mb/s
DCTCP_RATE_AI 1000Mb/s

ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 1
L2_BACK_TO_ZERO 0

HAS_WIN 1
GLOBAL_T 1
VAR_WIN 1
FAST_REACT 1
U_TARGET 0.95
MI_THRESH 0
INT_MULTI 1
MULTI_RATE 0
SAMPLE_FEEDBACK 0
PINT_LOG_BASE 1.05
PINT_PROB 1.0
NIC_TOTAL_PAUSE_TIME 0

RATE_BOUND 1
ACK_HIGH_PRIO 0
LINK_DOWN 0 0 0
ENABLE_TRACE 1

KMAX_MAP 6 25000000000 400 40000000000 800 100000000000 1600 200000000000 2400 400000000000 3200 2400000000000 3200
KMIN_MAP 6 25000000000 100 40000000000 200 100000000000 400 200000000000 600 400000000000 800 2400000000000 800
PMAX_MAP 6 25000000000 0.2 40000000000 0.2 100000000000 0.2 200000000000 0.2 400000000000 0.2 2400000000000 0.2

BUFFER_SIZE 32
EOF
NS3_CFG_CONTAINER="/app/llm-parallelism/astrasim/synthetic/${NS3_CFG_LOCAL}"

echo "Running AstraSim+ns-3 on $WORKLOAD_NAME ($NPUS ranks) [image=$ASTRASIM_IMAGE] ..."
echo "  topology=$NS3_TOPO_FILE  out=$NS3_OUT_DIR  log=$LOG"

# DETACH=1 launches the container with `docker run -d` so it OUTLIVES the
# launching shell (and any Claude session teardown). In foreground mode Docker
# ties the container's lifetime to the client process; when that dies it SIGTERMs
# the container and you lose the still-running ranks (the Exp 4 failure). Monitor
# a detached run with: docker logs -f <container>  (or tail the fct/qlen files).
CONTAINER_NAME=${CONTAINER_NAME:-astrasim-ns3-${WORKLOAD_NAME}-${NS3_OUT_SUBDIR:-ns3_output}}
if [[ "${DETACH:-0}" == "1" ]]; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  CID=$(docker run -d --name "$CONTAINER_NAME" --shm-size="$DOCKER_SHM" \
    -v /Users/sahas/workplace/astra-sim:/app/astra-sim \
    -v /Users/sahas/workplace/llm-parallelism:/app/llm-parallelism \
    "$ASTRASIM_IMAGE" \
    "$ASTRASIM_NS3_BIN" \
      --workload-configuration="$WORKLOAD_PREFIX" \
      --comm-group-configuration="$COMM_GROUP" \
      --system-configuration="$SYSTEM_CFG" \
      --network-configuration="$NS3_CFG_CONTAINER" \
      --remote-memory-configuration="$MEMORY_CFG" \
      --logical-topology-configuration="$LOGICAL_TOPO_CFG")
  echo "Launched detached container: $CONTAINER_NAME ($CID)"
  echo "Monitor:  docker logs -f $CONTAINER_NAME"
  echo "          tail -1 ${NS3_OUT_DIR}/qlen.txt   # simulated time (ns)"
  exit 0
fi

docker run --rm --shm-size="$DOCKER_SHM" \
  -v /Users/sahas/workplace/astra-sim:/app/astra-sim \
  -v /Users/sahas/workplace/llm-parallelism:/app/llm-parallelism \
  "$ASTRASIM_IMAGE" \
  "$ASTRASIM_NS3_BIN" \
    --workload-configuration="$WORKLOAD_PREFIX" \
    --comm-group-configuration="$COMM_GROUP" \
    --system-configuration="$SYSTEM_CFG" \
    --network-configuration="$NS3_CFG_CONTAINER" \
    --remote-memory-configuration="$MEMORY_CFG" \
    --logical-topology-configuration="$LOGICAL_TOPO_CFG" \
  > "$LOG" 2>&1

echo "--- AstraSim+ns-3 per-rank summary (sys[0]) ---"
grep -E "sys\[0\], (Wall time|GPU time|Comm time)|sys\[0\] finished" "$LOG" | tail -4
echo "exit signal: $(tail -1 "$LOG")"
echo "Full log: $LOG"
