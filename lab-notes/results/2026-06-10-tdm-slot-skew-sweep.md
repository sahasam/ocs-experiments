# TDM slot-length × clock-skew sweep — interpretation and data

**Date:** 2026-06-10  
**Status:** Complete. Confirms slot length is the dominant lever; clock skew at σ ≤ 10 µs is invisible.

## Context

The guard × bandwidth sweep (see `2026-06-10-tdm-guard-bw-sweep.md`) showed that guard band and bandwidth are well-understood once you're past the feasibility cliff. This sweep asks the orthogonal question: how does the **TDM slot quantum** (independent of guard) and **clock-skew jitter** affect mean step time and P99?

This is relevant for hardware design: TDM slot size is chosen based on switch reconfiguration overhead; clock-skew σ depends on endpoint synchronization quality.

**Three variants were run:**
1. `sweep_slot_skew`: slot × skew, pipelining **ON** (realistic case)
2. `sweep_slot_skew_nopipe`: slot × skew, pipelining **OFF** (worst-case; shows skew without masking)
3. `sweep_largeskew`: slot × large skew (σ ∈ {0, 100µs, 1ms, 10ms}), pipelining **OFF**

All use Llama-3 8B, DP=16, 2 nodes, `sirius_like` preset (circuit_bw=100 GB/s, packet_bw=10 GB/s, guard=10 ns, setup=0 ns, threshold=1024 B). Monte Carlo: 50 trials (main), 100 trials (largeskew).

---

## How slot and skew enter the model

From `tdm_model.py:chunk_time`:

```
transfer_ns = B / circuit_bandwidth_GBs        # = B / 100 GB/s
n_slots     = ceil((transfer_ns + guard_band_ns) / slot_length_ns)
slot_wait   = slot_length_ns / 2               # avg alignment delay to next slot boundary
total       = circuit_setup_ns + slot_wait + n_slots * slot_length_ns + skew
```

A chunk waits an average `slot/2` for the next slot edge, then occupies an integer number of full slots. Guard band is per-slot dead time counted as waste. Sub-threshold chunks (< 1024 B) skip to packet mode entirely.

**Clock-skew** (`clock_skew_sigma_ns`) is σ of a zero-mean Gaussian added to each chunk's elapsed time — per-chunk, IID across chunks and trials. Since it's zero-mean, it doesn't affect mean step time until σ is large enough that the per-chunk tail events become critical-path. With N ≈ 64 outer chunks per step, step-time jitter ≈ σ√N ≈ 8σ.

---

## Data: sweep_slot_skew (pipelining ON)

Scenario: Llama-3 8B, DP=16, 2 nodes. Compute floor: **435.67 ms**. 50 trials/cell.

### Mean step time (ms)

| slot \ skew | 0 ns | 10 ns | 100 ns | 1 µs | 10 µs |
|---|---|---|---|---|---|
| 10 ns | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 100 ns | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 1 µs | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 10 µs | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 100 µs | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 1 ms | 438.7 | 438.7 | 438.7 | 438.7 | 438.7 |
| 10 ms | 1115.9 | 1115.9 | 1115.9 | 1115.9 | 1115.9 |

### P99 step time (ms)

| slot \ skew | 0 ns | 10 ns | 100 ns | 1 µs | 10 µs |
|---|---|---|---|---|---|
| 10 ns | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 100 ns | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 1 µs | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 10 µs | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 100 µs | 436.7 | 436.7 | 436.7 | 436.7 | 436.7 |
| 1 ms | 438.7 | 438.7 | 438.7 | 438.7 | 438.7 |
| 10 ms | 1115.9 | 1115.9 | 1115.9 | 1115.9 | 1116.1 |

### Comm overhead % vs ideal

| slot \ skew | 0 ns | 10 ns | 100 ns | 1 µs | 10 µs |
|---|---|---|---|---|---|
| 10 ns | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% |
| 100 ns | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% |
| 1 µs | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% |
| 10 µs | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% |
| 100 µs | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% |
| 1 ms | 0.7% | 0.7% | 0.7% | 0.7% | 0.7% |
| 10 ms | 156.1% | 156.1% | 156.1% | 156.1% | 156.1% |

### P99 exposed comm (ms)

| slot \ skew | 0 ns | 10 ns | 100 ns | 1 µs | 10 µs |
|---|---|---|---|---|---|
| 10 ns | 1.07 | 1.07 | 1.07 | 1.07 | 1.07 |
| 100 ns | 1.07 | 1.07 | 1.07 | 1.07 | 1.07 |
| 1 µs | 1.07 | 1.07 | 1.07 | 1.07 | 1.07 |
| 10 µs | 1.07 | 1.07 | 1.07 | 1.07 | 1.07 |
| 100 µs | 1.07 | 1.07 | 1.07 | 1.07 | 1.07 |
| 1 ms | 3.00 | 3.00 | 3.00 | 3.00 | 3.03 |
| 10 ms | 680.25 | 680.25 | 680.26 | 680.27 | 680.45 |

---

## Data: sweep_slot_skew_nopipe (pipelining OFF)

Same scenario, same 50 trials/cell, but `net.pipelined_hierarchy = False`. This forces phases (inner RS, outer AR, inner AG) to be summed rather than overlapped — exposing outer-tier cost directly.

### Mean step time (ms)

| slot \ skew | 0 ns | 10 ns | 100 ns | 1 µs | 10 µs |
|---|---|---|---|---|---|
| 10 ns | 438.4 | 438.4 | 438.4 | 438.4 | 438.4 |
| 100 ns | 438.4 | 438.4 | 438.4 | 438.4 | 438.4 |
| 1 µs | 438.4 | 438.4 | 438.4 | 438.4 | 438.4 |
| 10 µs | 438.4 | 438.4 | 438.4 | 438.4 | 438.4 |
| 100 µs | 438.7 | 438.7 | 438.7 | 438.7 | 438.7 |
| 1 ms | 441.3 | 441.3 | 441.3 | 441.3 | 441.3 |
| 10 ms | 1184.1 | 1184.1 | 1184.1 | 1184.1 | 1184.1 |

### P99 step time (ms)

| slot \ skew | 0 ns | 10 ns | 100 ns | 1 µs | 10 µs |
|---|---|---|---|---|---|
| 10 ns | 438.4 | 438.4 | 438.4 | 438.4 | 438.4 |
| 100 ns | 438.4 | 438.4 | 438.4 | 438.4 | 438.4 |
| 1 µs | 438.4 | 438.4 | 438.4 | 438.4 | 438.4 |
| 10 µs | 438.4 | 438.4 | 438.4 | 438.4 | 438.5 |
| 100 µs | 438.7 | 438.7 | 438.7 | 438.7 | 438.7 |
| 1 ms | 441.3 | 441.3 | 441.3 | 441.3 | 441.4 |
| 10 ms | 1184.1 | 1184.1 | 1184.1 | 1184.1 | 1184.3 |

### Comm overhead % vs ideal

| slot \ skew | 0 ns | 10 ns | 100 ns | 1 µs | 10 µs |
|---|---|---|---|---|---|
| 10 ns | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% |
| 100 ns | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% |
| 1 µs | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% |
| 10 µs | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% |
| 100 µs | 0.7% | 0.7% | 0.7% | 0.7% | 0.7% |
| 1 ms | 1.3% | 1.3% | 1.3% | 1.3% | 1.3% |
| 10 ms | 171.8% | 171.8% | 171.8% | 171.8% | 171.8% |

### P99 exposed comm (ms)

| slot \ skew | 0 ns | 10 ns | 100 ns | 1 µs | 10 µs |
|---|---|---|---|---|---|
| 10 ns | 2.74 | 2.74 | 2.74 | 2.74 | 2.77 |
| 100 ns | 2.74 | 2.74 | 2.74 | 2.74 | 2.77 |
| 1 µs | 2.74 | 2.74 | 2.74 | 2.74 | 2.77 |
| 10 µs | 2.76 | 2.76 | 2.76 | 2.76 | 2.79 |
| 100 µs | 3.03 | 3.03 | 3.03 | 3.03 | 3.06 |
| 1 ms | 5.67 | 5.68 | 5.68 | 5.68 | 5.72 |
| 10 ms | 748.43 | 748.43 | 748.43 | 748.45 | 748.63 |

---

## Data: sweep_largeskew (pipelining OFF, extended σ range)

Extends the skew axis to {0, 100 µs, 1 ms, 10 ms} to find where P99 finally diverges from mean. 100 Monte Carlo trials. Circuit_bw=100 GB/s.

### P99 − mean step time (ms)

| slot \ skew | 0 ns | 100 µs | 1 ms | 10 ms |
|---|---|---|---|---|
| 100 ns | +0.00 | +0.30 | +3.18 | +90.74 |
| 10 µs | +0.00 | +0.30 | +3.17 | +90.83 |
| 1 ms | −0.00 | +0.46 | +4.29 | +101.38 |
| 10 ms | +0.00 | +1.81 | +18.14 | +175.72 |

### Mean step time (ms)

| slot \ skew | 0 ns | 100 µs | 1 ms | 10 ms |
|---|---|---|---|---|
| 100 ns | 438.4 | 438.4 | 438.5 | 538.2 |
| 10 µs | 438.4 | 438.4 | 438.6 | 538.4 |
| 1 ms | 441.3 | 441.4 | 441.8 | 566.3 |
| 10 ms | 1184.1 | 1184.1 | 1184.2 | 1194.3 |

---

## Interpretation

### Three slot regimes (reading the mean step column, pipelining ON)

1. **slot ≤ 100 µs:** Slot is much smaller than chunk transfer time. Alignment (slot/2) and quantization overhead are negligible. Step ≈ 436.7 ms (~0.2% overhead). *Sirius operates here.*
2. **slot = 1 ms:** Slot ≈ chunk transfer. Step → 438.7 ms (~0.7% overhead). Quantization visible but small.
3. **slot = 10 ms:** Slot ≫ chunk transfer. Every small outer-ring chunk pays a full 10 ms slot. 64 chunks × 10 ms ≈ 640 ms of slot waste. Step → 1115.9 ms (+156%).

### Pipelining effect

`net.pipelined_hierarchy = True` (default) computes each allreduce layer as `max(T_inner, T_outer)` instead of `sum`. With pipelining ON:
- The outer-tier cost is partially hidden behind inner NVLink work.
- At slot=10 ms: overhead is 156% (ON) vs 172% (OFF).
- At slot=10 ns: overhead is 0.2% (ON) vs 0.6% (OFF) — the base floor is higher without pipelining because outer is no longer masked.

### Why σ ≤ 10 µs is invisible

With σ = 10 µs, step-time jitter ≈ σ√64 ≈ 80 µs ≈ 0.08 ms. The step itself is ~436 ms. The skew columns look identical until σ is pushed to 1 ms or above (see largeskew). At σ = 10 ms and slot = 10 ms: P99 − mean grows to +176 ms — but this is far outside any realistic OCS endpoint spec.

### Practical takeaway

- **Slot ≤ 100 µs is the only regime that matters for Sirius-like OCS.** Any slot below ~100 µs produces indistinguishable results for this workload.
- **Clock skew σ ≤ 10 µs (realistic hardware) has zero visible effect on training throughput or P99.**
- P99 divergence is a σ ≥ 1 ms phenomenon — well above typical OCS endpoint jitter specs.
- The 10 ms slot pathology (`coarse_tdm_stress`) is a useful stress test, not a realistic scenario.
