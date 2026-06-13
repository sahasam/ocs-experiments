# Exp 3 — ns-3 fat-tree PS baseline: run attempt and topology findings

**Date:** 2026-06-11 → 2026-06-12  
**Status:** Terminated (incomplete). Yields a qualitative finding; no numeric result.

## Where this fits in the project

The OCS analytical sweeps (Exp 2, OCS replay) show how OCS degrades vs a theoretically uncongested PS reference. To argue OCS *advantage*, we need a PS baseline that includes congestion. This experiment attempts to provide that using ns-3 on a 128-GPU fat-tree.

Background on the OCS side: see memory `ocs-replay-toolchain.md` and `ocs-vs-ps-congestion.md`.

---

## Goal

Establish a packet-switched (PS) baseline *with congestion* to compare against the OCS results from Exp 2. The prior OCS sweeps only measure degradation vs an uncongested PS reference (C=∞, frictionless). To show OCS *advantage* we need PS *with* contention.

## What we built

- **ns-3 binary:** `ns3.42-AstraSimNetwork-optimized` was pre-built in the `astra-sim` repo and confirmed working in the `astra-sim-bigmem:latest` Docker container.
- **Topology:** `stage_configs/ns3_topo_128_fat_400g.txt` — 128-GPU 2-tier fat-tree: 8 leaf switches × 16 GPUs, 8 spine switches, 400 Gbps / 0.5 µs links, **2:1 oversubscription**. Bandwidth matches the OCS analytical runs.
- **Run script:** `run_astrasim_ns3.sh` (parallel to `run_astrasim_stage.sh` for the analytical backend).
- **Workload:** `llama3_8b_tp8_pp2_dp8` @ 128 ranks, same STAGE-generated Chakra ETs used in Exp 2.

Gotchas found during setup: topology file must have no `#` comment lines (ns-3 parser segfaults); `FLOW_FILE` is an input (background flows to inject), not an output.

## What happened

The run was started and left for ~24 hours. It never completed.

| Wall clock | Simulated time | Flows completed |
|---|---|---|
| 15 min | 35 ms | 2,090 |
| 1 hr | 103 ms | ~7,500 |
| 4 hr | 408 ms | ~60,000 |
| 12 hr | 1,107 ms | 121,126 |
| 24 hr | 2,051 ms | 226,093 |

The analytical (uncongested) baseline for this workload is **778 ms**. After 24 hours, ns-3 simulated time was 2,051 ms — 2.6× the uncongested baseline — with no rank having finished and new flows still being dispatched at the same rate. FCTs grew from ~0.3 ms early in the run to 1–5 ms by hour 24, indicating increasing congestion throughout.

Run was killed at 24 hours. Estimated time to completion: **several more days**.

## Root cause: direct all-reduce on an oversubscribed fat-tree

The core problem is a mismatch between the collective algorithm and the topology:

**`system.json` uses `"all-reduce-implementation": ["direct"]`** — every rank sends to every other rank in the group simultaneously. This algorithm is designed for fully-connected (OCS) fabrics where each pair has a dedicated circuit.

With DP=8 and 128 ranks, there are 16 DP groups each doing 8×7 = 56 simultaneous 8 MB flows = **896 concurrent cross-switch flows**. These all traverse the leaf→spine tier.

The fat-tree has 8 leaf switches × 8 uplinks = **64 leaf-spine links at 400 Gbps each**. With 896 flows:

```
896 flows / 64 links = ~14 flows per uplink
14 × 400 Gbps = 5.6 Tbps demand on a 400 Gbps link → 14:1 overload
```

HPCC (the congestion control algorithm) backs all senders down toward `MIN_RATE = 100 Mb/s`, then slowly ramps back up, creating oscillations that inflate per-flow completion time 10–15× and cascade through the entire training step.

## Key insight for the experiment design

**This is not a valid PS baseline for comparison with OCS.** Direct all-reduce is the wrong algorithm for a fat-tree. A real PS cluster running on a fat-tree would use:

- **Ring all-reduce** — each rank communicates with only 2 neighbors, 2 concurrent flows per rank instead of 7. Total concurrent cross-switch flows: ~128 (one ring-step at a time), well within the fat-tree's capacity.
- Or **recursive halving-doubling** — log₂(8) = 3 steps, each step at most 64 concurrent flows.

The OCS side legitimately uses `direct` because OCS provisions a dedicated circuit per communicating pair — simultaneous all-to-all is OCS's core value proposition.

A fair comparison is therefore:
- **OCS:** direct all-reduce on FullyConnected (current Exp 2 results) 
- **PS fat-tree:** ring all-reduce on the fat-tree (not yet measured)

Note that even this comparison understates OCS advantage, because OCS also enables direct (lower-latency) collectives that ring cannot match. The correct framing is: *OCS lets you run the more efficient algorithm; PS forces you to run ring*.

## What to do next

### Option A — Fix the ns-3 run (proper baseline)
Change `system.json` collective algorithm to `ring` for the ns-3 run:
```json
"all-reduce-implementation": ["ring"],
"all-gather-implementation": ["ring"],
"reduce-scatter-implementation": ["ring"]
```
With ring, ns-3 simulation time should be dramatically shorter (concurrent flows drop from 896 to ~128 per step). Expect hours, not days.

### Option B — Analytical congestion-aware + Switch (fast, conservative)
Use `AstraSim_Analytical_Congestion_Aware` with `topology: [Switch]` — single non-blocking switch, captures incast, runs in the same ~seconds as the existing analytical runs. Conservative (optimistic for PS) but scientifically valid as a lower bound on OCS advantage.

### Recommended: do Option B first, then Option A for validation
Option B gives results today; Option A gives a more realistic number to cross-check.

## Artifacts

- `stage_configs/ns3_topo_128_fat_400g.txt` — topology file
- `stage_configs/ns3_logical_128.json` — logical dims config
- `run_astrasim_ns3.sh` — run script
- `results/llama3_8b_tp8_pp2_dp8/ns3_output/` — partial fct.txt, qlen.txt, pfc.txt from the 24-hour run (not committed; gitignored)
- `results/llama3_8b_tp8_pp2_dp8/logs/astrasim_ns3.log` — 72-line init log (ring topology setup)
