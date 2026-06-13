# Workload baseline insights — Llama-3 8B

**Date:** 2026-06-10  
**Sources:** `../results/2026-06-10-astrasim-baselines.md`, `../results/2026-06-10-hybrid-net-preset-survey.md`

## The workload is strongly compute-bound

97% of communication is hidden behind compute. `comm-overlap fraction = 0.9688` across all runs. Even if the network doubles its latency, training slows by a fraction of a percent.

Consequence: we are measuring fractions of a percent of overhead, not 10–20% gaps. Precision matters more here than it would for a comm-heavy workload.

## The compute floor is 435.67 ms

All overhead percentages in this project are relative to `compute_time = 435.67 ms`. This number is invariant — same workload, same H100 roofline, same batch size — regardless of network config.

## OCS has a very low bar to clear

The IB analytical baseline exposes only 1.07–2.26 ms of communication (depending on DP degree). OCS only needs to keep exposed comm ≤ 2–3 ms to be effectively competitive with InfiniBand for this workload.

## sirius_like ≈ infiniband_baseline

| config | overhead | exposed comm |
|---|---|---|
| `infiniband_baseline` (50 GB/s packet) | 0.28% | 1.22 ms |
| `sirius_like` (100 GB/s circuit / 10 GB/s packet) | 0.24% | 1.07 ms |

OCS at Sirius-like parameters adds no measurable cost vs IB — it is actually 0.04% *better* because the 100 GB/s circuit path serves the few large chunks faster than the 50 GB/s IB path.

## Reference configuration

DP=16, 2-node (NVLink switch + outer IB-like ring), `npus_count=[8, 2]` is the canonical config for all analytical sweeps. The 0.24% baseline from `sirius_like` is the "OCS reference" all subsequent experiments compare against.
