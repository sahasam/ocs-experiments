# OCS engine capability audit — which engine can show OCS *beating* baseline

**Date:** 2026-06-15
**Kind:** decision / conceptual
**Trigger:** While planning the FTS paper, asked the precise question: does the OCS
analytical modeling allow OCS to come out *better than baseline*, or is it a pure
derating (penalty-only) model? Audited the actual code, not the notes.

## The question

For the OCS-vs-PS story to exist at all, *some* engine must be able to produce
`OCS_step < PS_step` — a genuine win, not just "less penalty." Which of our engines can,
and which structurally cannot?

## Findings (read from code)

**1. `ocs_replay.py` / `ocs_penalty.py` — pure derating, by construction. Cannot show a win.**
- `ocs_replay.py:152`: `extra = (f - 1.0) * (end - start)` with `f = max(1.0, k/C) ≥ 1`
  (`_oversub_factor`, line 134). Propagated delay is `rn.extra + max(0.0, in_delay)` —
  **non-negative always** (line 194).
- Docstring (line 13): at `C ≥ peak` the stretch is 1.0 and it "reproduces the trace oracle
  EXACTLY." The seed trace is the **frictionless C=∞ PS** (Congestion_Unaware).
- ⇒ OCS can *equal* the baseline at best, never beat it. These answer **"what does finite
  circuit capacity cost,"** not "does OCS win." There is no congestion in their baseline for
  OCS to remove. (Correct for capacity dimensioning; useless for a win-claim.)

**2. `coupled_sim.py` — NOT a derating model; OCS *can* finish earlier — but only on algorithm.**
- Built specifically because `ocs_replay` "can never make a node finish earlier than the
  trace" (`coupled_sim.py:6-8`). Rebuilds the schedule from scratch with a switchable cost
  model `direct` vs `ring` (lines 71-88); gate B asserts **direct ≤ ring** (line 28).
- **Catch:** no cross-rank link-contention model. One serial link per `(rank,tier)`,
  `n_parallel=1` (lines 195-197); contention is "already in each node's duration." So it
  captures the *algorithm* win but **not** fabric congestion — and the algorithm win is only
  **~0.3% at 50 GB/s** (note-to-ahmad). The ~3.8% that actually matters is invisible to it.

**3. `Switch + Congestion_Aware` (AstraSim analytical) — abandoned, correctly.**
- Single non-blocking switch: models **endpoint incast only** (many DP senders → one
  receiver downlink). 1-dim hard limit (`Helper.cpp:26`) ⇒ no Clos/fat-tree, ever.
- Structurally **cannot represent spine oversubscription**, which is the actual mechanism:
  the 5.6× PP penalty comes from cross-stage sends crossing the 2-uplink leaf-spine
  bottleneck. Switch would put the congestion in the wrong place (DP incast) and miss the
  spine. Not just optimistic — it **mismodels the mechanism**. This is why we dropped it.

## Conclusion

**Across the whole stack, ns-3 is the *only* engine that models multilevel-fabric
congestion.** Therefore the OCS *congestion* win is **inherently a packet-level
measurement.** There is no fast analytical engine that produces it truthfully:
- derating engines (`ocs_replay`/`ocs_penalty`) can't show a win;
- `coupled_sim` shows only the ~0.3% algorithm sliver, no contention;
- `Switch+CA` mismodels where the congestion is.

## Consequence for the FTS experiment plan

Do **not** build the "crossover map" of the *win* on any analytical engine. Decompose:
1. **Exposed-comm envelope** (analytical, dense, cheap): how much comm sits on the critical
   path vs bandwidth × DP × model. Not derating, not a win-claim — the **opportunity size**.
2. **Congestion multiplier** (ns-3, sparse, expensive): the 5.6×-class inflation an
   oversubscribed multilevel fabric applies to those exposed flows, + a 4:1-vs-8:1
   oversubscription A/B.

OCS win ≈ (exposed-comm fraction) × (congestion multiplier removed). Both must be
non-trivial for hybrid to pay — that is the actionable rule.

**This turns the limitation into the paper's methodology contribution: fidelity selection.**
Exposed-comm and capacity dimensioning are answerable analytically in seconds; the
multilevel-congestion win is *not* (single-tier analytical misplaces the congestion), so a
designer must escalate to packet-level. The methodology states the boundary of where each
fidelity tier is valid.

See [[ocs-engine-capability]] (memory), ns-3 trial spec in
`experiments/2026-06-15-fts-ns3-trials.md`.
