# FTS paper — ns-3 trial definitions (the congestion-multiplier points)

**Date:** 2026-06-15
**Purpose:** ns-3 is the only engine that models multilevel-fabric congestion (see
`learnings/2026-06-15-ocs-engine-capability-audit.md`), so the OCS *congestion win* in the
FTS paper must come from packet-level runs. The prior runs (Exp 4 fat-tree, Exp 5 thin Clos)
are **partial/confounded** — stage-0 was killed mid-run in both. These trials produce
**paper-grade** points: ring/ring, single-variable, stage-0 actually completing.

## Shared setup (all trials)

- **Workload:** `llama3_8b_tp1_pp2_dp8` (16 ranks). TP=1 ⇒ all collectives (DP all-reduce +
  PP sends) traverse the outer fabric — clean 1-dim. ETs already generated under
  `results/llama3_8b_tp1_pp2_dp8/`.
- **PS algorithm:** ring (`stage_configs/system_ns3.json`). **OCS uses ring too** —
  ring/ring kills the Exp-4 algorithm confound and is a *conservative* OCS bound (isolates
  congestion/topology only; FullyConnected serializes receive NICs so direct looks
  artificially bad — stated as a known limitation).
- **OCS reference (analytical, already have):** `run_astrasim_stage.sh` →
  FullyConnected + Congestion_Unaware → stage-0 4783 ms / stage-1 3203 ms. Also
  `fct.txt` `ideal_fct` column = OCS line-rate per flow, free.
- **Logical topo:** `stage_configs/ns3_logical_16.json`. **CC:** HPCC (CC_MODE 3) — prior
  runs used it; realistic.
- **Driver:** `run_astrasim_ns3.sh <workload> <npus>` with env overrides.

### Run hygiene (non-negotiable — these are why prior runs failed)
- **Run DETACHED** (`docker run -d`) so the container outlives the launching shell
  (foreground runs get SIGTERM'd on shell teardown → only finished ranks logged).
- **Do NOT stop when stage-1 logs "finished."** Stage-0 (ranks 0–7) completes ~1.5 s of
  *simulated* time later; **stage-0 wall time IS the step time.** Watch for stage-0 stats.
- **Disable qlen monitor:** `QLEN_MON_START 99999999999999` (24 MB+/run, only for diag).
- **Packet size:** keep `PACKET_PAYLOAD_SIZE 1000` (realism). 9000 = ~9× faster but changes
  realism — only if wall time forces it.

## Trials

### N1 — fat-tree 4:1, ring/ring, run to completion  *(triangulation point)*
```
NS3_TOPO_FILE=.../stage_configs/ns3_topo_16_fat_400g.txt \
SYSTEM_CFG=.../stage_configs/system_ns3.json \
NS3_OUT_SUBDIR=ns3_output_fat LOG_NAME=ns3_fat_4to1 \
bash run_astrasim_ns3.sh llama3_8b_tp1_pp2_dp8 16   # detached
```
- Completes the Exp-4 stage-0 number. Primary use: **cross-fidelity agreement** —
  analytical ≈ AstraSim-trace ≈ ns-3 on one point (the methodology's credibility claim).

### N2 — thin Clos 8:1, ring/ring, run to completion  *(headline +3.8% point)*
```
NS3_TOPO_FILE=.../stage_configs/ns3_topo_16_clos_thin.txt \
SYSTEM_CFG=.../stage_configs/system_ns3.json \
NS3_OUT_SUBDIR=ns3_output_clos LOG_NAME=ns3_clos_8to1 \
bash run_astrasim_ns3.sh llama3_8b_tp1_pp2_dp8 16   # detached
```
- Completes the Exp-5 stage-0 (currently killed/partial). Hardens the headline congestion
  point.

**N1 vs N2 = the clean single-variable oversubscription A/B** (4:1 vs 8:1; same workload,
algorithm, bandwidth — only spine uplink count differs). Directly answers *"when does PS
suffice."* This is the cleanest "multilevel congestion" axis we can isolate.

### N3 — lower link bandwidth (100–200 Gbps) to expose comm  *(STRETCH, only if N1/N2 land)*
- Pushes into the OCS-favorable regime (exposed comm grows). **Blocker:** HPCC rejects
  non-standard link rates (see [[ns3-toolchain]]) — needs DCQCN (CC_MODE 1) or a new
  standard-rate topo file. Risky tonight; do only with time to spare.

## Outputs to harvest (per run)
- **stage-0 & stage-1 wall** from the log → PS step = stage-0 wall.
- **Per-flow FCT slowdown** via `analyze_fct.py` on `fct.txt` (PP send vs DP all-reduce
  multipliers — expect PP ≫ DP).
- **OCS-vs-PS overhead** = (PS stage0 − OCS stage0) / OCS stage0.

## Acceptance
A trial is paper-grade only if **stage-0 logs completion** (not killed). Partial = discard.

## Time budget
~20–40 min wall per 16-rank run; detached, can overlap. N1 + N2 are the must-haves;
launch both first so they cook while the analytical exposed-comm envelope is built.
