# TDM operating envelope — slot, guard, skew

**Date:** 2026-06-10  
**Sources:** `../results/2026-06-10-tdm-guard-bw-sweep.md`, `../results/2026-06-10-tdm-slot-skew-sweep.md`

## Slot length

Three regimes (pipelining ON, Llama-3 8B DP=16):

| slot | overhead | verdict |
|---|---|---|
| ≤ 100 µs | 0.2% | invisible — Sirius operates here |
| 1 ms | 0.7% | visible but acceptable |
| 10 ms | +156% | catastrophic; every chunk pays a full slot |

**Practical rule: slot ≤ 100 µs is the only regime that matters.** Any value below ~100 µs produces indistinguishable step times for this workload.

## Guard band

At σ = 1 ns (clock-skew sigma):

- **≤ 7 ns:** infeasible — per-step collision probability > 1e-6. The cliff is sharp: 7 ns → P ≈ 1.5e-6, 10 ns → P ≈ 3e-12.
- **10 ns:** optimal. Effective bandwidth = 90% of nominal (100 ns slot − 10 ns guard = 90 ns payload). Step time ≈ compute floor at bw ≥ 30 GB/s.
- **> 10 ns:** linear bandwidth tax. guard=30 → 70% of nominal, guard=50 → 50%, guard=90 → 10%. Cost is invisible at bw=100 GB/s (compute-bound) but catastrophic at bw=3 GB/s.

**10 ns guard is the optimum.** Smaller: circuit doesn't reliably come up. Larger: pays bandwidth tax with no safety benefit (P is already 0 at 10 ns).

The knee is sharp on the left (3 ns of guard separates infeasible from rock-solid) and gentle on the right (linear tax).

## Clock-skew jitter

- **σ ≤ 10 µs:** zero visible effect on mean step time or P99. Step jitter ≈ σ√64 ≈ 80 µs for N=64 outer chunks — negligible against 436 ms step time.
- **σ = 1 ms:** mean step unchanged; P99 diverges by ~3–4 ms depending on slot.
- **σ = 10 ms:** P99 diverges by 90–175 ms — but this is far outside any realistic OCS endpoint spec.

**Practical hardware target:** σ ≤ 10 µs. Any tighter is free performance; any looser needs a guard-band reassessment.

## Workload-type caveat

For comm-dominated workloads (MoE, expert parallelism), the step-time penalty for over-guarding or large slots is **50–100× larger** than for compute-heavy LLM training. Designers optimizing for MoE cannot be conservative on guard; dense LLM training can afford to be.

## Sirius operating point

`sirius_like`: slot=100 ns, guard=10 ns, σ=1 ns. All three axes are well within safe territory.
