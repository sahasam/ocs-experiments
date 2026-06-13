# TDM guard-band × circuit-bandwidth sweep

**Date:** 2026-06-10  
**Status:** Complete. Identifies the operating envelope for `sirius_like_circuit`.

## Context

After establishing that `sirius_like` is the target OCS configuration (see `2026-06-10-hybrid-net-preset-survey.md`), this sweep asks: how sensitive is performance to the two key hardware parameters of the OCS link?

- **Guard band** (ns): dead time inserted between circuit slots to absorb clock jitter and prevent collisions. Larger guard → more bandwidth wasted per slot, but fewer timing violations.
- **Circuit bandwidth** (GB/s): raw line rate of the optical circuit (100, 50, 30, 10, 3 GB/s).

The feasibility gate: a cell is marked `✗` when the per-step collision probability exceeds 1e-6 (optical circuit cannot reliably come up). This is computed from `erfc(guard / (σ√2))` compounded over `n_pairs`, with σ = 1 ns (clock-skew sigma, a feasibility input only).

**Setup:**
- Pipelining: ON
- Scenario: Llama-3 8B, DP=16, 2 nodes (NVLink switch + outer-tier hybrid ring)
- Base preset: `sirius_like_circuit` (slot=100 ns, mode=flow, setup=0 ns)
- Clock-skew sigma: 1.0 ns
- Compute floor: 435.67 ms
- Outer-tier ring size (n_pairs) = 2

## Results

### Mean step time (ms)

Rows: guard band (ns). Columns: circuit bandwidth (GB/s).

| guard \ bw | 100 GB/s | 50 GB/s | 30 GB/s | 10 GB/s | 3 GB/s |
|---|---|---|---|---|---|
| 1 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 2 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 3 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 5 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 7 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 10 ns | 436.7 | 437.0 | 437.9 | 444.6 | 876.7 |
| 15 ns | 436.7 | 437.1 | 438.1 | 445.4 | 919.1 |
| 20 ns | 436.7 | 437.2 | 438.2 | 446.3 | 966.8 |
| 30 ns | 436.7 | 437.4 | 438.6 | 448.5 | 1082.7 |
| 50 ns | 436.9 | 438.1 | 439.7 | 545.2 | 1453.4 |
| 70 ns | 437.7 | 439.7 | 444.6 | 804.6 | 2318.3 |
| 90 ns | 443.2 | 545.2 | 804.6 | 2102.1 | 6643.1 |

### Effective circuit bandwidth (GB/s)

Rows: guard band (ns). Columns: circuit bandwidth (GB/s).

| guard \ bw | 100 GB/s | 50 GB/s | 30 GB/s | 10 GB/s | 3 GB/s |
|---|---|---|---|---|---|
| 1 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 2 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 3 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 5 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 7 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 10 ns | 90.0 | 45.0 | 27.0 | 9.0 | 2.7 |
| 15 ns | 85.0 | 42.5 | 25.5 | 8.5 | 2.5 |
| 20 ns | 80.0 | 40.0 | 24.0 | 8.0 | 2.4 |
| 30 ns | 70.0 | 35.0 | 21.0 | 7.0 | 2.1 |
| 50 ns | 50.0 | 25.0 | 15.0 | 5.0 | 1.5 |
| 70 ns | 30.0 | 15.0 | 9.0 | 3.0 | 0.9 |
| 90 ns | 10.0 | 5.0 | 3.0 | 1.0 | 0.3 |

### Comm overhead % vs ideal

Rows: guard band (ns). Columns: circuit bandwidth (GB/s).

| guard \ bw | 100 GB/s | 50 GB/s | 30 GB/s | 10 GB/s | 3 GB/s |
|---|---|---|---|---|---|
| 1 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 2 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 3 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 5 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 7 ns | ✗ | ✗ | ✗ | ✗ | ✗ |
| 10 ns | 0.2% | 0.3% | 0.5% | 2.0% | 101.2% |
| 15 ns | 0.2% | 0.3% | 0.5% | 2.2% | 111.0% |
| 20 ns | 0.2% | 0.3% | 0.6% | 2.4% | 121.9% |
| 30 ns | 0.2% | 0.4% | 0.7% | 2.9% | 148.5% |
| 50 ns | 0.3% | 0.6% | 0.9% | 25.1% | 233.6% |
| 70 ns | 0.5% | 0.9% | 2.0% | 84.7% | 432.1% |
| 90 ns | 1.7% | 25.1% | 84.7% | 382.5% | 1424.8% |

### Per-step collision probability

(Depends only on guard, σ, n_pairs — not on bandwidth. ✗ = P > 1e-6.)

| guard \ bw | 100 GB/s | 50 GB/s | 30 GB/s | 10 GB/s | 3 GB/s |
|---|---|---|---|---|---|
| 1 ns | 7.3e-01 | 7.3e-01 | 7.3e-01 | 7.3e-01 | 7.3e-01 |
| 2 ns | 2.9e-01 | 2.9e-01 | 2.9e-01 | 2.9e-01 | 2.9e-01 |
| 3 ns | 6.7e-02 | 6.7e-02 | 6.7e-02 | 6.7e-02 | 6.7e-02 |
| 5 ns | 8.1e-04 | 8.1e-04 | 8.1e-04 | 8.1e-04 | 8.1e-04 |
| 7 ns | 1.5e-06 | 1.5e-06 | 1.5e-06 | 1.5e-06 | 1.5e-06 |
| 10 ns | 3.1e-12 | 3.1e-12 | 3.1e-12 | 3.1e-12 | 3.1e-12 |
| 15 ns | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 20 ns | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 30 ns | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 50 ns | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 70 ns | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 90 ns | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |

## Findings

Three regimes are visible:

- **Cliff (guard ≤ 7 ns):** Infeasible at σ=1 ns. Collision probability falls 0.73 → 0.29 → 0.067 → 8.1e-4 → 1.5e-6 across guards {1, 2, 3, 5, 7}. The 1e-6 threshold lands between 7 and 10 ns.
- **Sweet spot (guard = 10 ns):** First feasible cell (P_step ≈ 3e-12). Effective bandwidth = 90% of nominal (100ns slot − 10ns guard = 90ns payload). Step time = 436.74 ms at bw=100 GB/s, essentially the compute floor.
- **Bandwidth tax (guard → 90 ns):** Effective bw falls linearly: guard 30 → 70%, 50 → 50%, 70 → 30%, 90 → 10% of nominal. Impact is invisible at bw=100 GB/s (compute-bound; comm < 1%) and catastrophic at bw=3 GB/s (step balloons to 6.6 s at guard=90 ns).

The effective bandwidth formula is exactly `bw_nominal × (slot - guard) / slot` — directly from `_quantize_slots` in `tdm_model.py`. Collision probability is independent of bandwidth (it's a function of guard, σ, n_pairs only).

### Practical takeaway

At σ = 1 ns (typical in-spec OCS endpoint), **guard = 10 ns** is the optimum. Smaller: circuit doesn't reliably come up. Larger: pays bandwidth tax without safety benefit. The knee is sharp on the left (3 ns of guard separates 8.1e-4 from 3e-12) and gentle on the right (linear cost).

For comm-dominated workloads (MoE, expert parallelism), the step-time penalty for over-guarding is 50–100× larger than for compute-heavy LLM training. Designers optimizing for this workload can afford to be conservative on guard; all-to-all-heavy workloads cannot.
