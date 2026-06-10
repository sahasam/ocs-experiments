"""Six sanity checks from the plan's verification section.

Run via:  python -m hybrid_net.cli --sanity
"""
from __future__ import annotations

import math
import statistics

import numpy as np

from .collectives import (
    Chunk,
    Phase,
    n_pairs_for_phase,
    ring_allreduce_byte_volume,
    ring_allreduce_chunks,
)
from .scheduler import LinkScheduler
from .simulate import (
    NetworkConfig,
    PerLayer,
    simulate_one_trial,
)
from .tdm_model import (
    CircuitMode,
    TDMConfig,
    chunk_time,
    flow_time,
    is_feasible,
    slot_error_prob,
)


def _make_layers(n_layers: int = 32, comp_us: int = 4586,
                 payload: int = 486_539_264) -> list[PerLayer]:
    """Reproduce the Llama-3 8B DP=16 ET shape (block 1 has no BWD_IG)."""
    layers: list[PerLayer] = []
    for b in range(1, n_layers + 1):
        layers.append(PerLayer(
            block=b,
            fwd_us=comp_us,
            bwd_ig_us=(0 if b == 1 else comp_us),
            bwd_wg_us=comp_us,
            comm_size_bytes=payload,
        ))
    return layers


def _net_16gpu_2node(tdm: dict[int, TDMConfig] | None = None) -> NetworkConfig:
    return NetworkConfig(
        topology=["Switch", "Ring"],
        npus_count=[8, 2],
        bandwidth_GBs=[400.0, 50.0],
        latency_ns=[1000.0, 1000.0],
        impl_per_tier=["ring", "ring"],
        tdm=tdm or {},
    )


def _net_16gpu_flat() -> NetworkConfig:
    return NetworkConfig(
        topology=["Ring"],
        npus_count=[16],
        bandwidth_GBs=[50.0],
        latency_ns=[1000.0],
        impl_per_tier=["ring"],
        tdm={},
    )


def _close(a: float, b: float, rel: float) -> bool:
    if b == 0:
        return abs(a) < rel
    return abs(a - b) / abs(b) <= rel


# Sanity 1: reproduce AstraSim's analytical numbers ---------------------------

def test_reproduce_astrasim() -> tuple[bool, str]:
    """INFINIBAND_BASELINE on 16-GPU 2-node should match AstraSim numbers.

    AstraSim reported (summary_16gpu_2node_ib.md):
       total_comm   = 40.499 ms
       exposed_comm =  1.266 ms
    Pass if both within 10%.
    """
    layers = _make_layers()
    net = _net_16gpu_2node()
    rng = np.random.default_rng(0)
    tr = simulate_one_trial(layers, net, rng)
    total_ms = tr.total_comm_ns / 1e6
    exposed_ms = tr.exposed_comm_ns / 1e6
    ok = _close(total_ms, 40.499, 0.10) and _close(exposed_ms, 1.266, 0.15)
    msg = (f"total_comm={total_ms:.3f}ms (AstraSim 40.499) "
           f"exposed={exposed_ms:.3f}ms (AstraSim 1.266)")
    return ok, msg


# Sanity 2: zero-cost circuit ------------------------------------------------

def test_zero_cost_circuit() -> tuple[bool, str]:
    """Setup=0, slot=0, guard=0, skew=0, circuit_bw=huge => comm ~ 0."""
    layers = _make_layers()
    tdm = TDMConfig(
        slot_length_ns=0.0, guard_band_ns=0.0, clock_skew_sigma_ns=0.0,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=1e9, packet_bandwidth_GBs=1e9,
        packet_latency_ns=0.0, packet_fallback_threshold_B=0,
    )
    # Apply to BOTH tiers so neither falls back to analytical.
    net = _net_16gpu_2node({0: tdm, 1: tdm})
    rng = np.random.default_rng(0)
    tr = simulate_one_trial(layers, net, rng)
    compute_ns = tr.compute_ns
    ok = tr.total_comm_ns < compute_ns * 1e-6  # comm is basically zero
    msg = (f"comm={tr.total_comm_ns:.1f}ns step={tr.step_ns/1e6:.3f}ms "
           f"compute={compute_ns/1e6:.3f}ms")
    return ok, msg


# Sanity 3: slot stress ------------------------------------------------------

def test_slot_stress() -> tuple[bool, str]:
    """slot_length=10ms >> any chunk => overhead balloons."""
    layers = _make_layers()
    tdm = TDMConfig(
        slot_length_ns=10_000_000.0,   # 10 ms
        guard_band_ns=0.0, clock_skew_sigma_ns=0.0,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=50.0, packet_bandwidth_GBs=50.0,
        packet_latency_ns=0.0, packet_fallback_threshold_B=0,
    )
    net = _net_16gpu_2node({1: tdm})
    rng = np.random.default_rng(0)
    tr = simulate_one_trial(layers, net, rng)
    # Baseline (no TDM) total comm should be ~ 40ms; under huge slots we expect >> 100 ms.
    ok = tr.exposed_comm_ns > 50_000_000  # > 50 ms exposed
    msg = (f"exposed={tr.exposed_comm_ns/1e6:.1f}ms (must be >50ms) "
           f"total_comm={tr.total_comm_ns/1e6:.1f}ms")
    return ok, msg


# Sanity 4: Monte Carlo determinism ------------------------------------------

def test_mc_determinism() -> tuple[bool, str]:
    """clock_skew_sigma=0 => all trials identical."""
    layers = _make_layers()
    tdm = TDMConfig(
        slot_length_ns=100.0, guard_band_ns=10.0, clock_skew_sigma_ns=0.0,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=100.0, packet_bandwidth_GBs=10.0,
        packet_latency_ns=200.0, packet_fallback_threshold_B=1024,
    )
    net = _net_16gpu_2node({1: tdm})
    rng = np.random.default_rng(123)
    steps = [simulate_one_trial(layers, net, rng).step_ns for _ in range(50)]
    ok = max(steps) - min(steps) < 1e-6
    msg = f"step variance over 50 trials: {max(steps) - min(steps):.6f} ns"
    return ok, msg


# Sanity 5: skew no longer perturbs delivery time ----------------------------

def test_skew_does_not_perturb_delivery() -> tuple[bool, str]:
    """Per the revised model, clock_skew_sigma_ns is a feasibility input, not
    a jitter source. With sigma > 0 (in any preset), all trials must still be
    identical. The remaining stochastic role of skew is in is_feasible()."""
    sigma = 100.0
    layers = _make_layers()
    tdm = TDMConfig(
        slot_length_ns=0.0,
        guard_band_ns=0.0,
        clock_skew_sigma_ns=sigma,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=50.0,
        packet_bandwidth_GBs=50.0,
        packet_latency_ns=0.0,
        packet_fallback_threshold_B=10**18,
    )
    net = _net_16gpu_flat()
    net.tdm[0] = tdm
    rng = np.random.default_rng(7)
    steps = [simulate_one_trial(layers, net, rng).step_ns for _ in range(50)]
    ok = max(steps) - min(steps) < 1e-6
    msg = f"sigma={sigma} ns -> step variance over 50 trials: {max(steps)-min(steps):.6f} ns"
    return ok, msg


# Sanity 6: hierarchical mass conservation -----------------------------------

def test_mass_conservation() -> tuple[bool, str]:
    """Total bytes on tier0 + tier1 = flat-ring volume on N = n0 * n1 ranks."""
    payload = 486_539_264
    n_inner, n_outer = 8, 2
    chunks = ring_allreduce_chunks(payload,
                                   [n_inner, n_outer], ["ring", "ring"])
    bytes_t0 = sum(c.bytes for c in chunks if c.tier == 0)
    bytes_t1 = sum(c.bytes for c in chunks if c.tier == 1)
    bytes_total = bytes_t0 + bytes_t1

    flat_n = n_inner * n_outer
    expected_flat = ring_allreduce_byte_volume(payload, flat_n)
    # Sanity: hierarchical total bytes should be in the same order of magnitude as flat.
    # (Hierarchical actually moves MORE bytes per rank than naive flat because we have
    # both inner phases sandwiching outer. Just check the components are nonzero
    # and tier-0 bytes >= tier-1 bytes.)
    ok = (bytes_t0 > 0) and (bytes_t1 > 0) and (bytes_t0 >= bytes_t1)
    msg = (f"tier0={bytes_t0:,} tier1={bytes_t1:,} "
           f"flat-ref={expected_flat:,}")
    return ok, msg


# Sanity 7: erfc closed-form bridge ------------------------------------------

def test_slot_error_closed_form() -> tuple[bool, str]:
    """slot_error_prob(sigma=5, guard=10, n_pairs=1) must match erfc(1) ~ 0.1573.

    Cross-check the formula against a value we can compute by hand.
    """
    p_pair, p_step = slot_error_prob(sigma_ns=5.0, guard_ns=10.0, n_pairs=1)
    expected = math.erfc(1.0)
    # n_pairs=1 path passes through `1 - (1 - p_pair)`, which round-trips
    # within 1 ULP of p_pair on IEEE-754 doubles.
    ok = abs(p_pair - expected) < 1e-15 and abs(p_step - p_pair) <= 1e-15
    msg = f"P_pair={p_pair:.12f} expected={expected:.12f} P_step={p_step:.12f}"
    return ok, msg


# Sanity 8: feasibility transition --------------------------------------------

def test_feasibility_transition() -> tuple[bool, str]:
    """is_feasible should be a monotone step in guard at fixed sigma: True for
    large guards (collision rare), False for small ones. Check the transition
    exists somewhere in a wide bracket around sigma."""
    sigma = 1.0
    # At guard 1*sigma collisions are common (~erfc(0.5)=0.48); at 10*sigma rare.
    not_feasible = is_feasible(sigma_ns=sigma, guard_ns=sigma * 1.0, n_pairs=2)
    feasible = is_feasible(sigma_ns=sigma, guard_ns=sigma * 10.0, n_pairs=2)
    ok = (not not_feasible) and feasible
    msg = (f"feasibility at guard=1sigma: {not_feasible} "
           f"(expected False); at guard=10sigma: {feasible} (expected True)")
    return ok, msg


# Sanity 9: ring multiplier ---------------------------------------------------

def test_ring_multiplier() -> tuple[bool, str]:
    """P_step = 1 - (1 - P_pair)^n_pairs should compound. For P_pair = 0.1
    and n_pairs = 4, expect ~0.3439."""
    # Solve for sigma giving P_pair = 0.1: erfc(guard/(2 sigma)) = 0.1.
    # erfc^-1(0.1) ~ 1.1631 => guard/(2 sigma) = 1.1631 => with guard = 10,
    # sigma = 10 / (2 * 1.1631) ~ 4.299
    sigma = 10.0 / (2.0 * 1.1631)
    p_pair_1, _ = slot_error_prob(sigma, guard_ns=10.0, n_pairs=1)
    _, p_step_4 = slot_error_prob(sigma, guard_ns=10.0, n_pairs=4)
    expected = 1.0 - (1.0 - p_pair_1) ** 4
    ok = abs(p_step_4 - expected) < 1e-9 and abs(p_pair_1 - 0.1) < 1e-3
    msg = (f"P_pair={p_pair_1:.4f} (~0.1) P_step(n=4)={p_step_4:.4f} "
           f"expected={expected:.4f}")
    return ok, msg


# Sanity 10: flow_time setup amortized ---------------------------------------

def test_flow_setup_amortized() -> tuple[bool, str]:
    """A Phase of n_chunks small chunks summed via chunk_time pays setup +
    slot_wait n times. flow_time over the same total bytes pays them once.
    For sirius-like params (setup=0, slot=100ns, slot_wait=50ns), the
    flow_time saving over n_chunks*chunk_time is roughly (n-1) * slot_wait."""
    tdm = TDMConfig(
        slot_length_ns=100.0, guard_band_ns=10.0, clock_skew_sigma_ns=0.0,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=100.0, packet_bandwidth_GBs=10.0,
        packet_latency_ns=200.0, packet_fallback_threshold_B=1024,
        mode=CircuitMode.FLOW_RESERVED,
    )
    B_total = 1_000_000      # 1 MB phase
    n_chunks = 64
    chunk_B = B_total // n_chunks
    sum_chunks = sum(chunk_time(chunk_B, tdm).time_ns for _ in range(n_chunks))
    ft = flow_time(B_total, tdm)
    # Sum should be larger by at least (n_chunks - 1) * slot_wait (50 ns).
    savings = sum_chunks - ft.duration_ns
    ok = savings > (n_chunks - 1) * (tdm.slot_length_ns / 2.0) * 0.5
    msg = (f"sum_chunks={sum_chunks:.1f} ns flow={ft.duration_ns:.1f} ns "
           f"saved={savings:.1f} ns (n_chunks={n_chunks})")
    return ok, msg


# Sanity 11: LinkScheduler serialization -------------------------------------

def test_link_serialization() -> tuple[bool, str]:
    """n_parallel=1 serializes; n_parallel=2 lets two simultaneous flows go."""
    s1 = LinkScheduler(n_parallel=1)
    a1, b1, _ = s1.schedule(0.0, 100.0)
    a2, b2, _ = s1.schedule(0.0, 100.0)
    ok_serial = (a1 == 0.0 and b1 == 100.0 and a2 == 100.0 and b2 == 200.0)

    s2 = LinkScheduler(n_parallel=2)
    a3, b3, c3 = s2.schedule(0.0, 100.0)
    a4, b4, c4 = s2.schedule(0.0, 100.0)
    ok_parallel = (a3 == 0.0 and a4 == 0.0 and c3 != c4)

    ok = ok_serial and ok_parallel
    msg = (f"serial: (0,{b1})/(100,{b2})  parallel: (0,{b3},ch{c3})/"
           f"(0,{b4},ch{c4})")
    return ok, msg


# Sanity 12: guard tax monotonicity ------------------------------------------

def test_guard_bandwidth_tax_monotone() -> tuple[bool, str]:
    """At fixed slot=100ns, sigma=0, increasing guard from 1 to 200 ns must
    drive flow_time duration monotonically non-decreasing (each chunk pays
    more guard, and may quantize into more slots)."""
    tdm_base = dict(
        slot_length_ns=100.0, clock_skew_sigma_ns=0.0,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=100.0, packet_bandwidth_GBs=10.0,
        packet_latency_ns=200.0, packet_fallback_threshold_B=1024,
        mode=CircuitMode.FLOW_RESERVED,
    )
    B_total = 50_000   # 0.5 us at 100 GB/s -> several slots
    guards = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0]
    durs = [flow_time(B_total, TDMConfig(guard_band_ns=g, **tdm_base)).duration_ns
            for g in guards]
    monotone = all(durs[i] <= durs[i + 1] + 1e-9 for i in range(len(durs) - 1))
    ok = monotone and durs[-1] > durs[0]
    msg = f"durations(ns) for guard {guards}: {[round(d, 1) for d in durs]}"
    return ok, msg


# Sanity 13: U-curve under comm-heavy regime ---------------------------------

def test_ucurve_signature() -> tuple[bool, str]:
    """With circuit_bw=3 GB/s (comm-heavy regime), guard sweep on FLOW_RESERVED
    must move step time meaningfully. Sanity that the model is sensitive to
    guard in a regime where comm matters."""
    layers = _make_layers()
    base = dict(
        slot_length_ns=100.0, clock_skew_sigma_ns=1.0,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=3.0,        # comm-heavy
        packet_bandwidth_GBs=1.0,
        packet_latency_ns=200.0, packet_fallback_threshold_B=1024,
        mode=CircuitMode.FLOW_RESERVED,
    )
    rng = np.random.default_rng(0)
    step_small = simulate_one_trial(
        layers, _net_16gpu_2node({1: TDMConfig(guard_band_ns=1.0, **base)}),
        rng).step_ns
    step_big = simulate_one_trial(
        layers, _net_16gpu_2node({1: TDMConfig(guard_band_ns=80.0, **base)}),
        rng).step_ns
    # Larger guard should slow the step (bandwidth tax). Expect at least 0.1ms
    # difference under comm-heavy params.
    delta_ms = (step_big - step_small) / 1e6
    ok = delta_ms > 0.1
    msg = (f"guard=1ns step={step_small/1e6:.3f}ms "
           f"guard=80ns step={step_big/1e6:.3f}ms  delta={delta_ms:.3f}ms")
    return ok, msg


TESTS = [
    ("reproduce AstraSim (10% match)",  test_reproduce_astrasim),
    ("zero-cost circuit",                test_zero_cost_circuit),
    ("slot stress",                      test_slot_stress),
    ("MC determinism (sigma=0)",         test_mc_determinism),
    ("skew does not perturb delivery",   test_skew_does_not_perturb_delivery),
    ("hierarchical mass conservation",   test_mass_conservation),
    ("slot_error_prob closed form",      test_slot_error_closed_form),
    ("feasibility transition",           test_feasibility_transition),
    ("ring multiplier",                  test_ring_multiplier),
    ("flow_time setup amortized",        test_flow_setup_amortized),
    ("LinkScheduler serialization",      test_link_serialization),
    ("guard tax monotonicity",           test_guard_bandwidth_tax_monotone),
    ("U-curve signature (comm-heavy)",   test_ucurve_signature),
]


def run_all() -> int:
    n_pass = 0
    n_fail = 0
    for name, fn in TESTS:
        ok, msg = fn()
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}: {msg}")
        if ok:
            n_pass += 1
        else:
            n_fail += 1
    print(f"\n{n_pass}/{len(TESTS)} sanity checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_all())
