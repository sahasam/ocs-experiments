# ocs-experiments

Simulation study of **optical circuit switching (OCS) for LLM training interconnects**, using [STAGE](https://github.com/astra-sim/symbolic_tensor_graph) to generate Chakra execution traces and [AstraSim](https://github.com/astra-sim/astra-sim) for coupled distributed training simulation.

The central question: how much does circuit-switched (optical) inter-node fabric hurt a training step under different parallelism layouts, and what determines the impact?

## Pipeline

```
STAGE (CPU, symbolic)          AstraSim (Docker)            OCS trace-replay (Python)
──────────────────────   →    ──────────────────────   →   ───────────────────────────
Chakra ETs for any             Coupled per-rank timing       Re-time OCS tier under
DP/TP/PP/SP layout             (PP stalls, compute           capacity-C contention,
No GPU cluster needed          overlap, roofline)            propagate delays
```

**No GPU cluster required.** STAGE generates symbolically-exact Chakra ETs validated at 0.23% comm-volume error vs a real 128-GPU H100 run. AstraSim runs in Docker.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
bash scripts/install_chakra.sh          # clones mlcommons/chakra, installs --no-deps

# Docker image for AstraSim (jemalloc variant required for 64-rank runs)
docker build -f astrasim/synthetic/stage_configs/Dockerfile.bigmem \
             -t astra-sim-bigmem:latest \
             astrasim/synthetic/stage_configs/
```

## Running an experiment

```bash
cd astrasim/synthetic
make           # generate ETs → simulate → OCS C-sweep (defaults: 8B TP8 PP2 DP4)
make help      # show all targets and current variable values
```

Override parallelism or model:

```bash
MODEL=70b DP=8 make
```

Individual steps:

```bash
make et        # generate Chakra ETs via STAGE (CPU, ~seconds)
make sim       # run AstraSim with trace logging (~3 s, Docker)
make sweep     # run OCS circuit-capacity C-sweep
```

See [`astrasim/synthetic/README.md`](astrasim/synthetic/README.md) for full details.

## Experiments

### Exp 1 — Guard band × circuit bandwidth sweep (sirius-like OCS)

**Config:** Llama-3 8B, DP=16, 2 nodes, `sirius_like_circuit` preset (100 ns slot, rotor mode).
Clock-skew σ = 1 ns. Compute-ideal step = 435.7 ms.

Cells marked `✗` are infeasible: circuit collision probability exceeds 1e-6 at that guard/σ combination.

**Step time (ms) — rows: guard band, columns: circuit bandwidth**

| guard \ bw | 100 GB/s | 50 GB/s | 30 GB/s | 10 GB/s | 3 GB/s |
|---|---|---|---|---|---|
| ≤7 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 10 ns | **436.7** | 437.0 | 437.9 | 444.6 | 876.7 |
| 15 ns | 436.7 | 437.1 | 438.1 | 445.4 | 919.1 |
| 30 ns | 436.7 | 437.4 | 438.6 | 448.5 | 1082.7 |
| 50 ns | 436.9 | 438.1 | 439.7 | 545.2 | 1453.4 |
| 70 ns | 437.7 | 439.7 | 444.6 | 804.6 | 2318.3 |
| 90 ns | 443.2 | 545.2 | 804.6 | 2102.1 | 6643.1 |

**Finding:** At σ=1 ns, **guard=10 ns is the sweet spot** — first feasible cell (P_collision≈3e-12), effective bandwidth 90% of nominal, step time within 0.2% of compute floor. The feasibility cliff is sharp (guard=7 ns fails, guard=10 ns is fine); the bandwidth tax above it is linear. For compute-bound LLM training at ≥30 GB/s, over-guarding up to 50 ns costs <1%; for comm-dominated workloads (≤10 GB/s), the same over-guard costs 25%.

---

### Exp 2 — OCS capacity sweep, TP8/PP2/DP4 hybrid parallelism

**Config:** Llama-3 8B, 64 ranks (TP=8, PP=2, DP=4). STAGE-generated traces, AstraSim roofline (H100: peak 300 TFLOPS, mem-bw 900 GB/s). AstraSim baseline: wall=1549 ms, GPU=596 ms, exposed comm=953 ms.

OCS tier = DP all-reduces (group size 4) + PP point-to-point sends. TP collectives stay on NVLink (intra-node, not OCS-visible). `C` = number of simultaneous circuits the fabric can support.

**Exposed step-time impact vs C=∞ baseline (1549 ms)**

| Circuits (C) | Max wall (ms) | Step overhead |
|---|---|---|
| ∞ (packet ref) | 1549 | 0% |
| 32 | 1549 | 0% |
| 16 | 1555 | +0.4% |
| 8 | 1566 | +1.1% |
| 4 | 1589 | **+2.6%** |
| 2 | 1636 | +5.6% |
| 1 | 1728 | +11.6% |

**Finding:** For 8B TP8/PP2/DP4, OCS contention has small exposed impact unless severely under-provisioned. At C=4 (4 simultaneous circuits for 64 ranks), step time grows by only 2.6% despite raw burst-blocking of +691 ms — **~74% of contention hides behind backward compute**. The dominant step cost is the PP bubble (rank 0 pipeline recv-wait ≈ 1056 ms), which OCS barely touches. C≥32 is identical to a packet-switched fabric.

**Validation:** the replay reproduces AstraSim's coupled timing exactly at C=∞ (rank0=1549.08 ms, rank32=1057.74 ms) — PP asymmetry and pipeline stalls are preserved by construction.

---

---

### Exp 3 — ns-3 fat-tree PS baseline: topology mismatch finding

**Config:** Same workload as Exp 2 (8B TP8/PP2/DP8, 128 ranks). Physical network: 2-tier fat-tree, 8 leaf × 16 GPUs, 8 spine, 400 Gbps / 0.5 µs links, 2:1 oversubscription. Congestion control: HPCC.

**Outcome:** Run was terminated after 24 hours without completing. After 24 h, ns-3 simulated time reached 2,051 ms (vs 778 ms analytical baseline — 2.6× slowdown) with 226K flows completed and no rank finished. FCTs grew from ~0.3 ms to 1–5 ms over the run.

**Root cause:** `direct` all-reduce (designed for OCS/fully-connected) on a 2:1 oversubscribed fat-tree. DP=8 with direct algorithm generates **896 simultaneous 8 MB cross-switch flows** competing for 64 leaf-spine links → ~14:1 link overload. HPCC oscillations cascaded through the entire step.

**Key finding:** This is not a valid PS baseline. A real PS cluster on a fat-tree would use **ring all-reduce** (~128 concurrent flows vs 896). The correct comparison is OCS+direct vs PS+ring — framing OCS's advantage as *enabling the more efficient collective algorithm*, not just avoiding congestion.

**Next step:** Re-run ns-3 with ring algorithm in system config, or use analytical congestion-aware + Switch as a fast conservative baseline.

See [`experiments/exp3-ns3-fat-tree-baseline.md`](experiments/exp3-ns3-fat-tree-baseline.md) for full details.

---

### Open experiments

- **PS baseline (fair)** — re-run ns-3 with `ring` all-reduce, or run analytical congestion-aware + Switch topology as a fast lower bound.
- **Rotor latency-floor sweep** — replay models bandwidth-sharing only; `T_cycle/2` circuit-wait floor for PP sends not yet applied. Negligible for ns rotors; potentially significant for ms MEMS.
- **PP-heavy TP8/PP4/DP2** — adversarial OCS config (smaller DP group, heavier pipeline pressure).
- **GPU capture validation** — real Megatron-LM trace (Phase 0: 8×A100, pure DP) to validate STAGE byte-counts and compute times.

## Repo layout

```
astrasim/synthetic/
├── Makefile                    # entry point: make / make et / make sim / make sweep
├── run_ocs_sweep.py            # OCS C-sweep CLI (wraps ocs_replay)
├── generate_stage_et.sh        # STAGE wrapper: DP/TP/PP/MODEL → Chakra ETs
├── run_astrasim_stage.sh       # AstraSim Docker runner (bigmem image, trace logging)
├── stage_configs/              # system.json, memory.json, Dockerfile.bigmem
├── hybrid_net/
│   ├── trace_loader.py         # parse AstraSim trace log → per-node timeline
│   ├── ocs_penalty.py          # isolate OCS-tier flows, concurrency profile
│   ├── ocs_replay.py           # delay-propagation C-sweep (the deliverable)
│   ├── dag_sim.py              # roofline + collective math (used by replay)
│   ├── et_loader.py            # Chakra ET reader
│   ├── collectives.py          # direct-algorithm collective cost model
│   ├── tdm_model.py            # TDM/rotor OCS bandwidth derating
│   ├── scheduler.py            # circuit scheduling
│   └── presets.py              # sirius_like, rotornet_like, etc.
├── workloads/                  # legacy per-rank workload descriptors
└── results/                    # gitignored (ETs, logs, traces)
src/
├── model.py                    # nanoGPT (CPU smoke test)
├── trace_capture.py            # ExecutionTraceObserver wrapper
├── megatron_trace_hook.py      # Megatron-LM step capture hook
configs/
├── llama3_70b.yaml             # TP=8, PP=2, DP=4 capture config
scripts/
├── install_megatron.sh
├── launch_megatron.sh
└── convert_to_chakra.sh
```

## Key references

- [STAGE](https://arxiv.org/abs/2511.10480) (arXiv 2511.10480) — symbolic tensor graph ET generation
- [AstraSim](https://github.com/astra-sim/astra-sim) — distributed ML system simulator
- [Chakra](https://github.com/mlcommons/chakra) — execution trace schema
