# Exp 5 — Coupled forward DAG scheduler: OCS direct vs ring vs ideal floor

**Date:** 2026-06-13
**Status:** OCS side complete and gate-validated. PS thin-Clos ns-3 run still pending.
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

## What's next — the PS thin-Clos baseline

To isolate "PP traffic on a thin shared spine" (the case OCS's dedicated circuits win),
built `stage_configs/ns3_topo_16_clos_thin.txt`: a 3-switch Clos (2 leaf + 1 spine).
Each PP stage's 8-rank DP group sits on its own leaf (DP all-reduce stays intra-pod),
so **all cross-stage PP traffic funnels through the single 800 Gbps spine** (4:1
oversubscribed). Run it with `/run-ps` (detached `docker run -d` — see Exp 4 for why),
then compare its PP-send congestion against this OCS floor.

## Artifacts

- `hybrid_net/coupled_sim.py` — the coupled forward DAG scheduler
- `run_coupled.py` — driver (ideal / direct / ring on one engine)
- `stage_configs/network_fc_16_ideal.yml` — Gate A ideal-floor network
- `stage_configs/ns3_topo_16_clos_thin.txt` — thin Clos PS topology (PS run pending)
