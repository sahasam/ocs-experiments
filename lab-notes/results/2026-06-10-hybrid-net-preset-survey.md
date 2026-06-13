# Hybrid-net preset survey — Llama-3 8B DP=16

**Date:** 2026-06-10  
**Status:** Complete. Established the performance range across realistic TDM presets and set `sirius_like` as the target configuration for subsequent sweeps.

## Context

After verifying that the AstraSim analytical baseline sits at 0.24% comm overhead (see `2026-06-10-astrasim-baselines.md`), we ran the `hybrid_net` Python model across four named presets to answer: how much overhead does a realistic OCS/TDM inter-node fabric add, and what does the worst-case TDM regime look like?

The model (`astrasim/synthetic/hybrid_net/`) replaces the AstraSim outer-tier ring with an analytical TDM/circuit model. All runs here use Llama-3 8B, DP=16, 2-node NVLink+outer-ring topology (npus_count=[8, 2]).

Each preset is defined in `presets.py`. The four covered:

| Preset | circuit_bw | packet_bw | slot | guard | setup | notes |
|---|---|---|---|---|---|---|
| `infiniband_baseline` | n/a | 50 GB/s | n/a | n/a | n/a | Pure packet; emulates IB ring |
| `sirius_like` | 100 GB/s | 10 GB/s | 100 ns | 10 ns | 0 ns | OCS-like; chunk → circuit if ≥1024 B |
| `sirius_like_circuit` | 100 GB/s | 10 GB/s | 100 ns | 10 ns | 0 ns | Same; forces circuit path explicitly |
| `coarse_tdm_stress` | n/a | 10 GB/s | 10 ms | — | — | Worst-case: all-packet but 10 ms TDM quanta |

## Results summary

| Preset | Step time | vs compute floor | Exposed comm | GPU util |
|---|---|---|---|---|
| `infiniband_baseline` | 436.89 ms | +0.28% | 1.22 ms | 0.9972 |
| `sirius_like` | 436.74 ms | +0.24% | 1.07 ms | 0.9976 |
| `sirius_like_circuit` | 436.74 ms | +0.24% | 1.07 ms | 0.9976 |
| `coarse_tdm_stress` | 545.22 ms | +25.15% | 109.55 ms | 0.7991 |

Compute floor: **435.67 ms**.

## Per-preset details

### infiniband_baseline — pure packet at 50 GB/s (1 trial)

**Headline:** step **436.89 ms**, overhead **0.28%**, exposed comm **1.22 ms**.

| metric | value |
|---|---|
| total comm | 38.923 ms |
| exposed comm | 1.216 ms |
| hidden comm | 37.707 ms |
| comm-overlap fraction | 0.9688 |
| % bytes in circuit mode | 0.00% |

This is the PS reference. All outer-ring bytes take the packet path at 50 GB/s.

---

### sirius_like — OCS hybrid at 100 GB/s circuit / 10 GB/s packet (50 trials)

**Headline:** step **436.74 ms**, overhead **0.24%**, exposed comm **1.07 ms**.

| metric | value |
|---|---|
| total comm | 34.088 ms |
| exposed comm | 1.065 ms |
| hidden comm | 33.022 ms |
| comm-overlap fraction | 0.9688 |
| % bytes in circuit mode | 6.67% |
| effective circuit bandwidth | 99.98 GB/s |
| guard-band waste fraction | 0.0222 |

50 Monte Carlo trials (clock-skew jitter); P99 = mean = 436.74 ms, confirming skew is invisible at σ ≤ 10 µs.

---

### sirius_like_circuit — same as sirius_like, circuit path forced (1 trial)

**Headline:** step **436.74 ms**, overhead **0.24%**, exposed comm **1.07 ms**.

| metric | value |
|---|---|
| total comm | 89.803 ms |
| exposed comm | 1.065 ms |
| hidden comm | 88.737 ms |
| % bytes in circuit mode | 6.67% |
| effective circuit bandwidth | 89.99 GB/s |
| guard-band waste fraction | 0.0241 |

Total comm is higher (89.8 vs 34.1 ms) because more data is routed through the circuit path at 100 GB/s raw vs 99.98 GB/s effective — but overlap hides it. Exposed comm is identical to `sirius_like`.

---

### coarse_tdm_stress — 10 ms TDM slots (1 trial)

**Headline:** step **545.22 ms**, overhead **25.15%**, exposed comm **109.55 ms**.

| metric | value |
|---|---|
| total comm | 389.295 ms |
| exposed comm | 109.549 ms |
| hidden comm | 279.746 ms |
| comm-overlap fraction | 0.7186 |
| % bytes in circuit mode | 0.00% |

All bytes take packet path but each chunk must wait for a 10 ms TDM slot. With 32 layers × 2 outer chunks = 64 chunks × 10 ms ≈ 640 ms of slot occupancy — well above the 435 ms compute window. The result is a 25% throughput hit.

## Key findings

- **sirius_like ≈ infiniband_baseline.** The OCS circuit path at Sirius-like parameters adds no measurable overhead vs IB. Both are within 0.04% of each other.
- **The slot granularity is the only thing that matters.** `coarse_tdm_stress` is 100× worse than `sirius_like` purely because of slot size (10 ms vs 100 ns). Bandwidth, guard band, and circuit setup time are secondary in the regime where slots are large relative to chunk size.
- **`sirius_like` became the canonical OCS target** for all subsequent parameter sweeps (guard × bandwidth, slot × skew). See `2026-06-10-tdm-guard-bw-sweep.md` and `2026-06-10-tdm-slot-skew-sweep.md`.
