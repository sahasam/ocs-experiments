# Why the OCS experiment needs a redesign

**Date:** 2026-06-13
**Sources:** `experiments/2026-06-13-exp4-16gpu-ocs-ps-ns3.md`, `2026-06-13-ocs-ps-ns3-comparison-learnings.md`
**Leads to:** Exp 5 design (beefy OCS direct AR vs thin Clos PS ring AR + ideal floor)

---

## The problem with what we have

Exp 4 gave us a *defensible but deliberately weak* result: ring-on-both-sides, +2.7% PS
overhead on stage 1, stage 0 never completed. We chose ring on both sides specifically to
avoid an algorithm confound and an AstraSim artifact. That makes Exp 4 a **congestion-only
lower bound** — it isolates the topology effect but throws away the headline OCS advantage,
which is *two-dimensional*: (1) no fabric congestion, **and** (2) the ability to run direct
all-reduce instead of ring. We can't make the real OCS claim from a ring-vs-ring run.

To make the real claim we need **OCS+direct vs PS+ring**. That immediately runs into three
tool problems, which is why the experiment has to be redesigned rather than just re-run.

## Problem 1 — AstraSim mis-models direct all-reduce (but it's an AstraSim bug, not physics)

AstraSim's analytical Congestion_Unaware + FullyConnected backend models direct all-reduce
as each rank broadcasting the full gradient to all N-1 peers, then **serializing the N-1
incoming flows at the receive NIC** (7×16 GB @ 50 GB/s ≈ 2.24 s exposed comm). That makes
direct look *slower* than ring — the opposite of reality. If we naively "run OCS with direct"
in AstraSim, OCS loses, by artifact.

**The key discovery during planning:** this serialization lives **only in AstraSim's backend**.
Our own Python model `hybrid_net/dag_sim.py` already models direct correctly — its
`direct_collective_phases` decomposes the all-reduce into sharded reduce-scatter + all-gather
phases (parallel, bandwidth-optimal). Its docstring even says so: it "matches AstraSim's
'direct' all-reduce; the ring model overshoots ~7x." So the fix is not to patch a
serialization out of anything — it's to **stop using AstraSim's analytical backend for the
OCS number and use `dag_sim`'s cost model instead.**

## Problem 2 — `ocs_replay` is delay-only and can't model a faster algorithm

`ocs_replay.py` is our main OCS tool, but it is a **delay propagator seeded by an AstraSim
trace**: at C=∞ it reproduces the trace exactly, and finite capacity only *adds* delay. It
can never make a node finish *earlier* than the baseline trace. So it structurally cannot
turn a ring trace into a (faster) direct result, nor correct an inflated AstraSim direct
trace downward. It's the right tool for "how much does capacity-C contention stretch the
step," and the wrong tool for "what does a better algorithm buy." Different question, needs
a different engine.

## Problem 3 — `dag_sim` has the right cost model but no cross-rank coupling

`dag_sim` models direct AR correctly and models per-tier link contention, but it simulates
**each rank independently**. It does not model the PP coupling where stage 0 idles until
stage 1's backward gradient arrives. That bubble is large (≈ stage-1 compute) and **both**
the ideal floor (AstraSim, coupled) and the PS ground truth (ns-3, coupled) include it. An
OCS number that omits it isn't comparable. So per-rank `dag_sim` alone is too optimistic on
stage 0.

`ocs_replay.build_graph` already extracts the cross-rank wiring we'd need (send→recv
`sr_edges`, collective barriers `coll_inst`) — but it's bolted to the delay-only engine.

## What the redesign therefore requires

1. **A coupled forward DAG scheduler** = `dag_sim`'s correct direct cost model + the
   cross-rank send→recv / collective-barrier edges `build_graph` already knows how to
   extract. This is the "--collective-barrier mode" the `dag_sim` docstring foreshadows.
   It gives a coupled, artifact-free OCS+direct number, self-validated by two cheap gates:
   at infinite bandwidth it must reproduce the AstraSim ideal floor; with ring at 50 GB/s
   it must reproduce Exp 4's coupled OCS numbers (4783 / 3203 ms), and direct must come in
   ≤ ring.
2. **An ideal floor** (infinite bandwidth, 500 ns latency) so OCS and PS are each reported
   as overhead *above* the unavoidable compute+latency floor, anchoring the two different
   backends (analytical OCS vs packet-level PS) to a common reference.
3. **A thinner PS fabric** (3-switch Clos: 2 leaf + 1 spine, 4:1) with DP intra-pod so the
   experiment cleanly isolates *PP traffic on a thin shared spine* — the case OCS's
   dedicated circuits are supposed to win.

## One-line takeaway

Exp 4's ring-vs-ring result is real but conservative; the headline OCS advantage (direct AR
+ dedicated circuits) can't be measured with AstraSim's analytical backend (it serializes
direct) or with `ocs_replay` (delay-only, can't speed anything up). The redesign moves the
OCS number onto a **coupled forward scheduler built on `dag_sim`'s already-correct direct
cost model**, against an ideal floor and a thin Clos PS baseline.
