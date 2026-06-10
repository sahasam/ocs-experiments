"""Per-link TDM + packet-fallback timing model.

A TDMConfig describes a single link tier. For each chunk of bytes that crosses
that link, chunk_time() decides whether to send via reserved circuit (slot-aligned
TDM) or pure packet mode, and returns the elapsed nanoseconds.

Unit convention: bandwidth in GB/s, latency / slot lengths in nanoseconds.
Because 1 GB/s = 1 byte/ns exactly, transfer time in ns is just
    bytes / bandwidth_GBs

Clock-skew semantics (revised):
    clock_skew_sigma_ns is the per-endpoint clock standard deviation. It does
    NOT add jitter to delivered transfer time -- in a real circuit-switched
    fabric, skew above the guard-band's tolerance makes the circuit fail to
    come up, it doesn't merely stretch delivery. Skew enters the model only
    through is_feasible(sigma, guard, n_pairs), used by sweeps as a binary
    mask on (guard, sigma) design points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class CircuitMode(str, Enum):
    PER_CHUNK_PACING = "per_chunk"   # legacy: every chunk independently slot-aligned
    FLOW_RESERVED    = "flow"        # one reservation amortised across a whole Phase


@dataclass
class TDMConfig:
    slot_length_ns:              float   # TDM slot granularity
    guard_band_ns:               float   # dead time per slot to absorb skew
    clock_skew_sigma_ns:         float   # per-endpoint clock std-dev; feasibility input
    circuit_setup_ns:            float   # one-time circuit reservation cost
    circuit_bandwidth_GBs:       float   # link rate when in circuit mode
    packet_bandwidth_GBs:        float   # link rate when in packet mode
    packet_latency_ns:           float   # one-way packet latency floor
    packet_fallback_threshold_B: int     # chunk < threshold => packet, else circuit
    mode:                CircuitMode = CircuitMode.PER_CHUNK_PACING
    n_parallel_circuits: int = 1         # WDM lanes / parallel channels on the tier


@dataclass
class ChunkTiming:
    """One per-chunk result. Lets simulate.py compute aggregate stats
    (slot utilization, mode mix, guard-band waste) without re-running."""
    time_ns:      float
    mode:         str           # 'circuit' or 'packet'
    bytes:        int
    n_slots:      int           # 0 in packet mode
    guard_waste_ns: float       # 0 in packet mode


@dataclass
class FlowTiming:
    """One Phase-level reservation: setup paid once, single slot alignment,
    integer slots of payload. Deterministic in (B_total, tdm)."""
    duration_ns:    float
    mode:           str           # 'circuit' or 'packet'
    bytes:          int
    n_slots:        int
    guard_waste_ns: float


def slot_error_prob(sigma_ns: float, guard_ns: float, n_pairs: int = 1
                    ) -> tuple[float, float]:
    """Return (P_pair, P_step) collision probability.

    Per-endpoint clock skew delta_i ~ N(0, sigma). For one TX-RX pair sharing a
    slot, pairwise skew is N(0, sigma * sqrt(2)). Slot fails iff the magnitude
    exceeds guard_band:
        P_pair = P(|N(0, sigma*sqrt(2))| > guard) = erfc(guard / (sigma * 2))
    A ring step on n_pairs simultaneous TX-RX pairs succeeds iff every pair
    succeeds: P_step = 1 - (1 - P_pair)^n_pairs.
    """
    if sigma_ns <= 0:
        return 0.0, 0.0
    p_pair = math.erfc(guard_ns / (sigma_ns * 2.0))
    p_step = 1.0 - (1.0 - p_pair) ** max(1, n_pairs)
    return p_pair, p_step


def is_feasible(sigma_ns: float, guard_ns: float, n_pairs: int = 1,
                threshold: float = 1e-6) -> bool:
    """A (guard, sigma, n_pairs) design point is feasible if its slot-collision
    probability is below `threshold`. Used by sweeps as a binary mask: above the
    threshold the optical circuit fails to come up reliably, and the simulated
    step time is not meaningful."""
    return slot_error_prob(sigma_ns, guard_ns, n_pairs)[1] <= threshold


def _quantize_slots(transfer_ns: float, slot_ns: float, guard_ns: float
                    ) -> tuple[int, float]:
    """Return (n_slots, occupied_ns) for a transfer that takes `transfer_ns`
    of line-rate time when there is a guard band of `guard_ns` per slot.

    Physical model: each `slot_ns` window dedicates `guard_ns` at the end
    to switching / clock-sync, leaving `(slot_ns - guard_ns)` ns of payload
    time. The flow needs `n_slots = ceil(transfer_ns / (slot_ns - guard_ns))`
    slots, each consuming `slot_ns` of wall time.

    Falls back to "no slot model" when slot_ns <= 0: one logical slot, no
    quantization, guard is paid once at the end.
    """
    if slot_ns <= 0:
        return 1, transfer_ns + guard_ns
    payload_window = slot_ns - guard_ns
    if payload_window <= 0:
        # Guard exceeds the slot: no payload fits.
        # Return a sentinel large duration so the caller / sweep can flag it.
        return 0, math.inf
    n_slots = max(1, math.ceil(transfer_ns / payload_window))
    return n_slots, n_slots * slot_ns


def chunk_time(B_bytes: int, tdm: TDMConfig, rng=None) -> ChunkTiming:
    """Time and metadata for delivering one chunk across a TDM/packet link.

    Deterministic in (B_bytes, tdm). The `rng` parameter is accepted for API
    compatibility with the legacy signature but is no longer consulted -- skew
    no longer perturbs delivery time (see module docstring).
    """
    if B_bytes <= 0:
        return ChunkTiming(0.0, "packet", 0, 0, 0.0)

    if B_bytes >= tdm.packet_fallback_threshold_B:
        if tdm.circuit_bandwidth_GBs <= 0:
            raise ValueError("circuit_bandwidth_GBs must be > 0 in circuit mode")
        transfer_ns = B_bytes / tdm.circuit_bandwidth_GBs
        n_slots, occupied = _quantize_slots(transfer_ns, tdm.slot_length_ns,
                                            tdm.guard_band_ns)
        slot_wait = tdm.slot_length_ns / 2.0 if tdm.slot_length_ns > 0 else 0.0
        total = tdm.circuit_setup_ns + slot_wait + occupied
        guard_waste = n_slots * tdm.guard_band_ns
        return ChunkTiming(total, "circuit", B_bytes, n_slots, guard_waste)

    # Packet mode
    if tdm.packet_bandwidth_GBs <= 0:
        raise ValueError("packet_bandwidth_GBs must be > 0 in packet mode")
    transfer_ns = B_bytes / tdm.packet_bandwidth_GBs
    total = transfer_ns + tdm.packet_latency_ns
    return ChunkTiming(total, "packet", B_bytes, 0, 0.0)


def flow_time(B_total: int, tdm: TDMConfig) -> FlowTiming:
    """One reservation across a Phase: setup once, one slot alignment, integer
    slots of the full payload. Captures the physics of an OCS where a single
    circuit carries the entire phase's bytes contiguously.

    Compared to summing chunk_time over the Phase's chunks (legacy
    PER_CHUNK_PACING), flow_time amortises:
        - circuit_setup_ns           paid once, not n_chunks times
        - slot_length_ns / 2 wait    paid once, not n_chunks times
    The slot quantization on the full payload is unchanged: n_slots is the
    minimum integer slots to fit B_total + guard at circuit bandwidth.
    """
    if B_total <= 0:
        return FlowTiming(0.0, "packet", 0, 0, 0.0)
    if B_total < tdm.packet_fallback_threshold_B:
        if tdm.packet_bandwidth_GBs <= 0:
            raise ValueError("packet_bandwidth_GBs must be > 0 in packet mode")
        return FlowTiming(
            B_total / tdm.packet_bandwidth_GBs + tdm.packet_latency_ns,
            "packet", B_total, 0, 0.0)
    if tdm.circuit_bandwidth_GBs <= 0:
        raise ValueError("circuit_bandwidth_GBs must be > 0 in circuit mode")
    transfer_ns = B_total / tdm.circuit_bandwidth_GBs
    n_slots, occupied = _quantize_slots(transfer_ns, tdm.slot_length_ns,
                                        tdm.guard_band_ns)
    slot_wait = tdm.slot_length_ns / 2.0 if tdm.slot_length_ns > 0 else 0.0
    duration = tdm.circuit_setup_ns + slot_wait + occupied
    guard_waste = n_slots * tdm.guard_band_ns
    return FlowTiming(duration, "circuit", B_total, n_slots, guard_waste)


def analytical_chunk_time(B_bytes: int, bandwidth_GBs: float, latency_ns: float) -> ChunkTiming:
    """Plain bytes / bw + latency timing for tiers WITHOUT a TDMConfig.

    Returned as a ChunkTiming so simulate.py treats both code paths uniformly.
    """
    if B_bytes <= 0:
        return ChunkTiming(0.0, "packet", 0, 0, 0.0)
    return ChunkTiming(B_bytes / bandwidth_GBs + latency_ns, "packet", B_bytes, 0, 0.0)
