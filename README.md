# ocs-experiments

Simulation study of **optical circuit switching (OCS) for LLM training interconnects**, using [STAGE](https://github.com/astra-sim/symbolic_tensor_graph) to generate Chakra execution traces and [AstraSim](https://github.com/astra-sim/astra-sim) for coupled distributed training simulation.

The central question: how much does an optical-circuit-switched (OCS) inter-node fabric help or hurt a training step versus a packet-switched (PS) fabric, under different parallelism layouts, and what determines the impact?

## Headline findings

- **The OCS advantage is two-dimensional, and the two dimensions are very different sizes.** For an 8B model on 400 Gbps-class links it decomposes into *(1)* the better collective algorithm OCS enables (direct all-reduce on dedicated circuits vs ring): **~0.3%** at 50 GB/s, and *(2)* freedom from fabric congestion vs a thin shared PS spine: **~3.8%**. Congestion is ~10× the algorithm effect, but both are small here because the workload is bubble/compute-bound — the network is a few-percent effect whichever way you cut it.
- **OCS sensitivity is dominated by DP degree, not model size.** 8B/70B at DP=8 need circuit count C ≥ 128 to stay within ~1% of a packet fabric; 405B at DP=2 is essentially OCS-immune at any realistic C. Scaling ranks by doubling DP (not PP) is the dangerous path for fabric design.
- **Circuit setup latency is irrelevant** at commercial MEMS timescales (≤27 ms): 1F1B pipeline scheduling leaves 100 ms–5 s of slack that absorbs it. Circuit *capacity* (C), not latency, is the binding constraint.
- **The contention OCS erases is on the shared spine, and PP traffic pays it.** On a thin 8:1 Clos, PP cross-stage sends run 5.6× slower than ideal while intra-pod DP all-reduce is unaffected — OCS's dedicated circuits target exactly that traffic.

## Pipeline

```
STAGE (CPU, symbolic)        AstraSim / coupled sim        OCS engines (Python)         PS ground truth
─────────────────────   →   ──────────────────────   →   ──────────────────────   vs  ──────────────────
Chakra ETs for any           Coupled per-rank timing       C-sweep contention            Packet-level ns-3
DP/TP/PP/SP layout           (PP stalls, compute           + coupled forward DAG         fat-tree / Clos,
No GPU cluster needed         overlap, roofline)            scheduler (direct algo)       HPCC, ring algo
```

**No GPU cluster required.** STAGE generates symbolically-exact Chakra ETs validated at 0.23% comm-volume error vs a real 128-GPU H100 run. AstraSim and ns-3 run in Docker. Both fabrics are anchored to a common ideal floor (infinite bandwidth) so each is reported as overhead *above* the unavoidable compute+latency cost.

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

See [`lab-notes/experiments/2026-06-11-exp3-ns3-fat-tree-baseline.md`](lab-notes/experiments/2026-06-11-exp3-ns3-fat-tree-baseline.md) for full details.

---

### Exp 4 — 16-GPU OCS vs PS, ring-on-both-sides (congestion-only baseline)

**Config:** Llama-3 8B, 16 ranks (TP=1, PP=2, DP=8). OCS side analytical (FullyConnected, congestion-free); PS side packet-level ns-3 on a 4:1 fat-tree, HPCC. **Ring all-reduce on both sides** to isolate the topology effect from the algorithm confound found in Exp 3.

**Finding:** A clean, deliberately *conservative* lower bound. PP stage-1 step is +2.7% on the PS fat-tree vs the OCS floor. The per-flow breakdown (from `fct.txt`) is the mechanism: PP activation/gradient sends run **4.1× slower** than ideal because 268 MB flows compete with DP ring traffic on the oversubscribed spine, while DP ring flows see 2–3×. This validates the core hypothesis — *DP traffic on a shared fabric degrades the PP critical path* — but ring-vs-ring throws away the headline OCS advantage (the direct algorithm), so a redesign was needed.

See [`lab-notes/experiments/2026-06-13-exp4-16gpu-ocs-ps-ns3.md`](lab-notes/experiments/2026-06-13-exp4-16gpu-ocs-ps-ns3.md).

---

### Exp 5 — Coupled forward DAG scheduler: OCS direct vs PS thin-Clos

**Config:** Same 8B TP1/PP2/DP8 workload. New OCS engine `coupled_sim.py` — a global event-driven scheduler over every rank's ET that combines a bandwidth-optimal *direct* collective cost model with the cross-rank PP send→recv / collective-barrier coupling (the PP bubble emerges from the graph, not a hack). The collective algorithm is switchable (`direct` = OCS, `ring` = PS) so one engine produces both sides. PS ground truth: a thin 8:1 Clos (2 leaf + 1 spine, DP intra-pod) in ns-3.

**Validation (two gates, both pass):** at infinite bandwidth the scheduler reproduces the AstraSim ideal floor (stage-1 exact to the ns, stage-0 +0.16%); at ring @ 50 GB/s it reproduces Exp 4's numbers to ~0.3%, with direct ≤ ring always.

**Finding — the two-dimensional decomposition (headline):**

| dimension | mechanism | step impact (8B, 50 GB/s) |
|---|---|---|
| Algorithm | direct all-reduce vs ring | **+0.3%** |
| Congestion | OCS floor vs thin-Clos PS (8:1) | **+3.8%** (stage 0), +4.0% (stage 1) |

On the thin spine, PP cross-stage sends pay **5.6× congestion** (worst flow 8.7×) while intra-pod DP all-reduce is unaffected — exactly the traffic OCS isolates. But a 5.6× PP penalty is only ~183 ms on the ~4.8 s step because most comm hides behind compute, so the step-level win is ~+3.8%. The algorithm dimension grows from +0.3% → +4.1% as the fabric narrows from 50 → 5 GB/s, which points to where OCS wins bigger.

See [`lab-notes/experiments/2026-06-13-exp5-coupled-ocs-direct-vs-ring.md`](lab-notes/experiments/2026-06-13-exp5-coupled-ocs-direct-vs-ring.md) and the [summary note](lab-notes/2026-06-14-note-to-ahmad.pdf).

---

### Open experiments

- **Push into the OCS-favorable regime** (highest value) — the few-percent win is gated by comm-hiding, not congestion size. Lower link bandwidth (100–200 Gbps), larger DP degree, smaller compute/step, and deeper pipelines all expose more comm on the critical path and should grow the gap.
- **Scale the controlled deep-dive to 70B / 405B** — the big models currently have only synthetic-sweep numbers; the coupled-vs-ns-3 comparison is 8B only.
- **Controlled 4:1-vs-8:1 spine A/B** — Exp 5's thin-Clos differs from Exp 4's fat-tree on two variables; a single-variable oversubscription sweep would cleanly isolate the spine effect.
- **QoS-aware PS baseline** — the current PS is best-effort (no priority queuing), a conservative comparison for OCS; quantify what QoS would neutralize.
- **GPU capture validation** — real Megatron-LM trace (Phase 0: 8×A100, pure DP) to validate STAGE byte-counts and compute times.

## Repo layout

```
astrasim/synthetic/
├── Makefile                    # entry point: make / make et / make sim / make sweep
├── run_ocs_sweep.py            # OCS C-sweep CLI (wraps ocs_replay)
├── run_coupled.py              # coupled forward DAG scheduler driver (ideal/direct/ring)
├── generate_stage_et.sh        # STAGE wrapper: DP/TP/PP/MODEL → Chakra ETs
├── run_astrasim_stage.sh       # AstraSim Docker runner (bigmem image, trace logging)
├── run_astrasim_ns3.sh         # ns-3 packet-level PS runner (DETACH, topo file)
├── analyze_fct.py              # per-flow-size congestion slowdown from ns-3 fct.txt
├── stage_configs/              # system.json, memory.json, ns-3 topos, Dockerfile.bigmem
├── hybrid_net/
│   ├── trace_loader.py         # parse AstraSim trace log → per-node timeline
│   ├── ocs_penalty.py          # isolate OCS-tier flows, concurrency profile
│   ├── ocs_replay.py           # delay-propagation C-sweep (capacity contention)
│   ├── coupled_sim.py          # coupled forward DAG scheduler (direct algo + PP bubble)
│   ├── dag_sim.py              # roofline + collective math (used by replay/coupled)
│   ├── et_loader.py            # Chakra ET reader
│   ├── collectives.py          # direct-algorithm collective cost model
│   ├── tdm_model.py            # TDM/rotor OCS bandwidth derating
│   ├── scheduler.py            # circuit scheduling
│   └── presets.py              # sirius_like, rotornet_like, etc.
├── workloads/                  # legacy per-rank workload descriptors
└── results/                    # gitignored (ETs, logs, traces, ns-3 fct/qlen)
lab-notes/                      # experiment writeups, learnings, result tables
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
