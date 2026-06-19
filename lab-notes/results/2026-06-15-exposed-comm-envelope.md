# Exposed-comm envelope — step overhead vs outer-fabric bandwidth (8B, DP=16)

**Date:** 2026-06-15
**For:** FTS paper, methodology layer 1 (the analytical "opportunity size" map).
**Engine:** `hybrid_net.cli` analytical (congestion-UNAWARE) — bandwidth-limited exposed
comm, no contention. This is the opportunity size, not a win-claim (see
`learnings/2026-06-15-ocs-engine-capability-audit.md`).

## Setup

- Trace: `llama3_8b_dp16_2node_ib` (16 ranks, DP=16, 2-node) — the canonical analytical
  reference (compute floor 435.67 ms).
- Network: `npus_count=[8,2]`, NVLink tier fixed at **450 GB/s**, outer/inter-node tier
  **swept**, outer latency 500 ns. System: ring (`system_16gpu.json`).
- Command pattern (per outer bw): `python -m hybrid_net.cli --et-dir
  results/llama3_8b_dp16_2node_ib --num-npus 16 --network-yml <net> --system-json
  system_16gpu.json`. Raw data + per-bw YAMLs/CSVs under `results/envelope/`.

## Result

| outer GB/s | outer Gbps | step (ms) | overhead % | exposed comm (ms) |
|---|---|---|---|---|
| 50    | 400 | 436.89 | 0.28  | 1.22   |
| 25    | 200 | 438.10 | 0.56  | 2.43   |
| 12.5  | 100 | 440.82 | 1.18  | 5.15   |
| 6.25  | 50  | 467.34 | 7.27  | 31.67  |
| 3.125 | 25  | 778.73 | 78.74 | 343.06 |
| 1.5625| 12  | 1401.50| 221.69| 965.83 |

**Validation anchor:** the 400 Gbps row (0.28%, 1.22 ms) reproduces the prior
`infiniband_baseline` point exactly (workload-baseline-insights).

## Interpretation

- **Super-linear knee at ~50–100 Gbps.** Above ~100 Gbps exposed comm is <1.2% — the
  network is hidden behind compute and **PS alone suffices**. Below ~50 Gbps exposed comm
  explodes (7% → 79% → 222%) — this is the regime where fabric improvements (hybrid /
  circuits) have something to bite on.
- **This is the opportunity-size envelope.** It bounds how much *any* network optimization
  (OCS included) can win: the win is capped by the exposed-comm fraction at that operating
  point.
- **It explains the small ns-3 congestion win at 400 Gbps.** The packet-level 5.6× PP
  congestion multiplier acts on only ~1.2 ms of exposed comm there → a few-percent step
  effect. The combined methodology claim: **OCS win ≈ (exposed fraction, this table) ×
  (congestion multiplier, ns-3 N1/N2).** Both must be non-trivial for hybrid to pay; the
  envelope says you must be at/under the knee for the congestion multiplier to matter.

## Caveats / next
- DP-degree dominance (a higher-DP trace shifts the knee right) is established separately by
  the synthetic C-sweeps ([[ocs-replay-toolchain]]); not re-swept here.
- Single model (8B). 70B/405B need matching tiered YAMLs; deferred.
- Latency fixed 500 ns; the curve is bandwidth-driven (exposed comm ≫ latency until <25 Gbps).
