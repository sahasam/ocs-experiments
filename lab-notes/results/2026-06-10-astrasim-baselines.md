# AstraSim analytical baselines — Llama-3 8B at two scales

**Date:** 2026-06-10  
**Status:** Complete. These are the fixed reference points for all subsequent OCS/TDM comparisons.

## Why these were run

Before introducing any network model (TDM, OCS, congestion), we need to know:

1. What is the compute floor — the step time with a theoretically perfect network?
2. How much communication is already overlapped by the pipeline?
3. What overhead does a realistic (InfiniBand-like) network add?

AstraSim's analytical backend answers these precisely: it replays the Chakra execution trace with a parameterized bandwidth/latency model and reports per-rank step time, compute time, and exposed vs. hidden comm. Both runs use STAGE-generated traces.

The compute floor falls out as `compute_time` in the tables below: **435.67 ms** (invariant — same workload, same H100 roofline, same batch). Every subsequent experiment's "overhead %" is relative to this number.

## DP=8 flat — 8 ranks, single-node (or flat topology)

Source: `astrasim/synthetic/results/summary.md`  
Config: `llama3_8b_tp8_pp2_dp8`, 128 ranks (8 data-parallel groups of 16).  
Network: analytical backend, FullyConnected at H100 IB-like bandwidth.

**Headline:** step **437.93 ms**, compute **435.67 ms**, exposed comm **2.26 ms**, overhead **0.47%**.

| metric | min | mean | max |
|---|---|---|---|
| step time (real) | 437.931 ms | 437.931 ms | 437.931 ms |
| step time (ideal net) | 435.862 ms | 435.862 ms | 435.862 ms |
| compute time | 435.670 ms | 435.670 ms | 435.670 ms |
| total comm | 72.356 ms | 72.356 ms | 72.356 ms |
| exposed comm | 2.261 ms | 2.261 ms | 2.261 ms |
| hidden comm | 70.094 ms | 70.094 ms | 70.094 ms |
| comm-overlap fraction | 0.9688 | 0.9688 | 0.9688 |
| comm overhead % | 0.47 % | 0.47 % | 0.47 % |
| GPU util (real net) | 0.9948 | 0.9948 | 0.9948 |
| GPU util (ideal net) | 0.9996 | 0.9996 | 0.9996 |
| GPU-seconds/step | 3.5034 s | 3.5034 s | 3.5034 s |
| $/step | $0.003893 | $0.003893 | $0.003893 |
| J/step | 2452.414 J | 2452.414 J | 2452.414 J |

> P99 is deferred: AstraSim's analytical backend is deterministic per-rank. P99 only becomes meaningful once we capture a real GPU trace with step-to-step variance (GCP follow-up phase).

## DP=16 — 16 ranks, 2 nodes × 8 H100s, NVLink switch + InfiniBand

Source: `astrasim/synthetic/results/summary_16gpu_2node_ib.md`  
Config: Llama-3 8B, DP=16, `[Switch, Ring]` topology with npus_count=[8, 2].  
Tier 0 (Switch): NVLink at 400 GB/s. Tier 1 (Ring): InfiniBand at ~50 GB/s analytical.

**Headline:** step **436.94 ms**, compute **435.67 ms**, exposed comm **1.27 ms**, overhead **0.24%**.

| metric | min | mean | max |
|---|---|---|---|
| step time (real) | 436.936 ms | 436.936 ms | 436.936 ms |
| step time (ideal net) | 435.868 ms | 435.868 ms | 435.868 ms |
| compute time | 435.670 ms | 435.670 ms | 435.670 ms |
| total comm | 40.499 ms | 40.499 ms | 40.499 ms |
| exposed comm | 1.266 ms | 1.266 ms | 1.266 ms |
| hidden comm | 39.233 ms | 39.233 ms | 39.233 ms |
| comm-overlap fraction | 0.9688 | 0.9688 | 0.9688 |
| comm overhead % | 0.24 % | 0.24 % | 0.24 % |
| GPU util (real net) | 0.9971 | 0.9971 | 0.9971 |
| GPU util (ideal net) | 0.9995 | 0.9995 | 0.9995 |
| GPU-seconds/step | 6.9910 s | 6.9910 s | 6.9910 s |
| $/step | $0.007768 | $0.007768 | $0.007768 |
| J/step | 4893.679 J | 4893.679 J | 4893.679 J |

## Key takeaways

- **The workload is strongly compute-bound.** 97% of communication is hidden behind compute. Even if the network doubles its latency, training only slows by a fraction of a percent.
- **This sets a very low bar for OCS.** Any network that exposes ≤2–3 ms of communication is effectively competitive with InfiniBand for this workload. The OCS / TDM comparisons in subsequent experiments are all measuring fractions of a percent, not 10–20% gaps.
- **DP=16 (2-node) is the configuration all subsequent hybrid_net experiments use.** The 0.24% baseline is the "IB reference" against which TDM presets are compared.
- **435.67 ms is the compute floor.** All percentage overhead numbers throughout the project are relative to this.
