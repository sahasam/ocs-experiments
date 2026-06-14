# Exp 5 — Coupled forward DAG scheduler: OCS direct vs ring vs ideal floor

**Date:** 2026-06-13
**Status:** Complete. OCS side gate-validated; PS thin-Clos ns-3 baseline finished.
**Builds on:** `learnings/2026-06-13-why-redesign-ocs-experiment.md`, Exp 4.

---

## Why this experiment

Exp 4 measured OCS vs PS *ring-on-both-sides* — a congestion-only lower bound that
throws away the headline OCS advantage (direct all-reduce + dedicated circuits). The
redesign note explained why we couldn't just re-run: AstraSim's analytical backend
serialises direct AR (artifact), and `ocs_replay` is delay-only (can't make a node
finish earlier). The fix was to build a **third engine**: a coupled forward DAG
scheduler on `dag_sim`'s already-correct direct cost model.

## What we built

`hybrid_net/coupled_sim.py` — a global, event-driven list scheduler over **every
rank's ET at once**, driven by `run_coupled.py`. It combines:

- **dag_sim's per-node cost model** — roofline COMP, `direct_collective_phases`
  (bandwidth-optimal), p2p SENDs on the inter-node tier.
- **the cross-rank coupling `ocs_replay.build_graph` already knew how to extract**,
  but wired into a *forward scheduler* instead of a delay propagator:
  - PP send→recv: a RECV is a zero-cost sync gated on its matched SEND (matched on
    src,dst,tag). The PP bubble now **emerges from the graph** — no more `pp_recv_stall`
    hack.
  - collective barrier: a collective instance (same pg_name+name across its group)
    starts when its slowest member is ready and all members finish together.
- **a switchable collective algorithm** (`--collective-impl`): `direct` (OCS) or
  `ring` (PS), so the same engine produces both sides of the comparison.

This is the engine `dag_sim`'s docstring foreshadowed as "--collective-barrier mode."

## Gate validation (both pass)

**Gate A — infinite bandwidth must reproduce the AstraSim ideal floor.**
Ran AstraSim analytical on a flat FullyConnected with bandwidth 1e6 GB/s
(`stage_configs/network_fc_16_ideal.yml`):

| stage | AstraSim ideal floor | coupled (inf bw) | Δ |
|-------|----------------------|------------------|---|
| 0 (ranks 0–7)  | 4758.1 ms | 4765.8 ms | +0.16% |
| 1 (ranks 8–15) | 3181.180614 ms | 3181.2 ms | **exact to the ns** |

**Gate B — ring @ 50 GB/s must reproduce Exp 4's coupled OCS, and direct ≤ ring.**

| stage | Exp 4 AstraSim (FC, ring) | coupled ring @ 50 | Δ |
|-------|---------------------------|-------------------|---|
| 0 | 4783 ms | 4794.9 ms | +0.25% |
| 1 | 3203 ms | 3191.9 ms | −0.35% |

Direct comes in ≤ ring on every stage. Independent agreement with AstraSim to ~0.3%
on both gates means the coupling and the cost model are faithful.

The **1584 ms stage0−stage1 gap** the coupled scheduler produces matches the ~1.58 s
PP bubble observed directly in Exp 4 — the coupling is real, not assumed.

## Headline result (OCS fabric, 16 GPUs, llama3_8b tp1/pp2/dp8)

| config | stage 0 | stage 1 |
|--------|---------|---------|
| ideal (inf bw)   | 4765.8 ms | 3181.2 ms |
| **direct (OCS)** | **4779.2 ms** | **3191.9 ms** |
| ring (PS algo)   | 4794.9 ms | 3191.9 ms |

**At a 50 GB/s OCS fabric, direct beats ring by only +0.3% on the step.** The DP
all-reduce overlaps almost entirely with backward compute, so the step is
PP-bubble / compute-bound and the collective *algorithm* barely touches the critical
path. The engine is not blind to the algorithm — it responds correctly as the fabric
narrows and comm gets exposed:

| OCS bandwidth | direct vs ring (stage 0) |
|---------------|--------------------------|
| 50 GB/s | +0.3% |
| 10 GB/s | +2.0% |
|  5 GB/s | +4.1% |

## What this means for the OCS story

This **refines the redesign's hypothesis.** The headline OCS advantage was framed as
two-dimensional — (1) no fabric congestion *and* (2) direct instead of ring. For this
workload at a realistic 50 GB/s fabric, dimension (2) is worth ~0.3%: direct AR is a
rounding error once DP comm hides behind backward. The advantage that actually matters
is dimension (1): **congestion on the PP critical path.** Exp 4 already showed PP sends
run 4.1× slower than ideal on the shared fat-tree spine. OCS's congestion-free number
*is* essentially the ideal floor (direct @ 50 ≈ floor + 0.3%), so the whole OCS-vs-PS
gap will come from PS congestion, not the algorithm.

## The PS thin-Clos baseline (complete)

To isolate "PP traffic on a thin shared spine" (the case OCS's dedicated circuits win),
built `stage_configs/ns3_topo_16_clos_thin.txt`: a 3-switch Clos (2 leaf + 1 spine).
Each PP stage's 8-rank DP group sits on its own leaf (DP all-reduce stays intra-pod),
so **all cross-stage PP traffic funnels through the single spine**.

Originally specced at 4:1 (800 Gbps leaf uplinks), but this ns-3 build's HPCC INT
telemetry rejects non-standard link rates (`Error: IntHeader unknown rate:
800000000000`), corrupting the congestion model. Dropped the uplink to a single
**400 Gbps** spine link per leaf → **8:1 oversubscription**, deliberately thinner than
Exp 4's 4:1 fat-tree and all-standard-rate so HPCC stays valid. (8:1 vs 4:1 means this
isn't a controlled A/B against Exp 4 — it's a deliberately stressed thin fabric to show
the OCS-favorable regime.)

Run via the updated `run_astrasim_ns3.sh` with `DETACH=1` (container outlives the shell —
see Exp 4), ring (`system_ns3.json`), payload 1000 + HPCC to match Exp 4's settings so
topology is the only changed variable vs the fat-tree. Compare the resulting PP-send
slowdown (Exp 4 fat-tree: 4.1×) against this OCS floor.

Invocation:
```bash
DETACH=1 SYSTEM_CFG=.../system_ns3.json \
  NS3_TOPO_FILE=.../ns3_topo_16_clos_thin.txt \
  NS3_OUT_SUBDIR=ns3_output_clos LOG_NAME=astrasim_ns3_clos \
  bash run_astrasim_ns3.sh llama3_8b_tp1_pp2_dp8 16
```

## Results — PS thin-Clos vs OCS floor

The run completed in ~40 min wall (4.96 s simulated, 15,248 flows). As in Exp 4,
stage 1 finished first; stage 0 (the bottleneck) finished 1.64 s of simulated time
later — and this time we **got the complete stage-0 number** (Exp 4's was killed).

**Step time (Wall):**

| stage | OCS floor (coupled direct) | PS thin-Clos 8:1 (ns-3) | overhead |
|-------|----------------------------|-------------------------|----------|
| 0 (ranks 0–7)  | 4779.2 ms | **4962.5 ms** | **+3.8%** |
| 1 (ranks 8–15) | 3191.9 ms | **3318.3 ms** | **+4.0%** |

**Per-flow congestion (fct/ideal_fct), thin-Clos 8:1 vs Exp 4 fat-tree 4:1:**

| flow type | size | count | ideal | fat-tree 4:1 | **thin-Clos 8:1** |
|-----------|------|-------|-------|--------------|-------------------|
| PP cross-stage send | 256 MiB | 16 | 5630 µs | 4.13× (max 5.42×) | **5.63× (max 8.67×)** |
| DP AR — embedding | 15.7 MB | 896 | 346 µs | 2.49× | 2.70× |
| DP AR — transformer | 5.5 MB | 14336 | 123 µs | 3.05× | **2.40×** |

### Reading the result

1. **The thin shared spine hits exactly the traffic OCS isolates.** Halving the spine
   (2 spines → 1, 4:1 → 8:1) pushes **PP-send** slowdown from 4.13× to **5.63×** (worst
   flow 8.67×). Meanwhile **DP all-reduce is intra-pod** (each stage's 8-rank DP group
   lives on one leaf, never crossing the spine) so it's comparable or even *better* on
   the Clos (transformer AR 3.05× → 2.40×, since fewer spine-transiting PP flows disturb
   the leaf). This is the OCS thesis in one table: **the contention is on the shared
   spine, and it's PP traffic that pays.** OCS gives PP its own circuits and erases it.

2. **But at the step level it's only ~+4%, because the workload is bubble/compute-bound.**
   A 5.63× PP-send penalty translates to just +183 ms on the 4779 ms stage-0 step — most
   PP/DP comm overlaps with compute, so only the exposed tail on the critical path counts.
   This is the same lesson as the OCS-side direct-vs-ring result (+0.3%): **for an 8B
   model at 400 Gbps-class links, the network is a few-percent effect, whichever way you
   cut it** — algorithm (direct vs ring) ~0.3%, fabric congestion (OCS vs thin-Clos PS)
   ~3.8%. The congestion dimension is ~10× the algorithm dimension, confirming where the
   OCS value is, but both are small here.

3. **Where OCS would win bigger:** the +3.8% is gated by comm-hiding, not by the size of
   the congestion. Workloads that expose more comm on the critical path — lower
   bandwidth, larger DP degree, smaller compute per step, deeper pipelines with tighter
   bubbles — would convert the 5.6× PP-send penalty into a larger step-time gap. The
   bandwidth sweep above (direct−ring 0.3%→4.1% as bw drops 50→5 GB/s) is the same knob.

### Honest caveats

- 8:1 vs Exp 4's 4:1 means this isn't a controlled 1-variable delta against the fat-tree;
  it's a deliberately stressed thin fabric. The 4:1 → 8:1 row still reads cleanly as
  "thinner spine ⇒ worse PP congestion," but the absolute +3.8% is specific to 8:1.
- The OCS side is analytical/coupled (congestion-free by construction); the PS side is
  packet-level ns-3. They share the same ETs, compute model, and ring algorithm, and are
  anchored to a common ideal floor, so the comparison is apples-to-apples up to the
  backend (analytical vs packet) — the same caveat as Exp 4.

## Artifacts

- `hybrid_net/coupled_sim.py` — the coupled forward DAG scheduler
- `run_coupled.py` — driver (ideal / direct / ring on one engine)
- `analyze_fct.py` — per-flow-size congestion slowdown from an ns-3 fct.txt
- `stage_configs/network_fc_16_ideal.yml` — Gate A ideal-floor network
- `stage_configs/ns3_topo_16_clos_thin.txt` — thin Clos PS topology (2 leaf + 1 spine, 8:1)
- `run_astrasim_ns3.sh` — now supports `DETACH=1`, `NS3_OUT_SUBDIR`, `LOG_NAME`
- `results/llama3_8b_tp1_pp2_dp8/ns3_output_clos/` — thin-Clos fct.txt / qlen.txt
- `results/llama3_8b_tp1_pp2_dp8/logs/clos_stdout_durable.log` — ns-3 Wall-time log
