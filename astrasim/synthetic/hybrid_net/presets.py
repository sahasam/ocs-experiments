"""Named TDMConfig presets. Each describes a single link tier's behavior.

INFINIBAND_BASELINE is the validation oracle: it should reproduce AstraSim's
16-GPU 2-node IB numbers when applied to tier 1 of a [Switch, Ring]
topology with bandwidth [400 GB/s, 50 GB/s]. The other presets are stylized
research configurations.
"""
from __future__ import annotations

from .tdm_model import CircuitMode, TDMConfig

PRESETS: dict[str, TDMConfig] = {

    # No slot/guard/skew; circuit never used (threshold > any chunk size).
    # Behaviorally identical to plain bytes / 50 GB/s + 0 ns -- so the hybrid
    # model on this preset must match AstraSim's analytical math on tier 1.
    "infiniband_baseline": TDMConfig(
        slot_length_ns=0.0,
        guard_band_ns=0.0,
        clock_skew_sigma_ns=0.0,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=50.0,
        packet_bandwidth_GBs=50.0,
        packet_latency_ns=0.0,
        packet_fallback_threshold_B=10**18,   # everything goes packet
    ),

    # Sub-microsecond optical TDM (Sirius-like).
    "sirius_like": TDMConfig(
        slot_length_ns=100.0,
        guard_band_ns=10.0,
        clock_skew_sigma_ns=5.0,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=100.0,
        packet_bandwidth_GBs=10.0,
        packet_latency_ns=200.0,
        packet_fallback_threshold_B=1024,     # < 1 KB chunks go packet
    ),

    # Periodic circuit rotation, no setup -- RotorNet style.
    "rotornet_like": TDMConfig(
        slot_length_ns=10_000.0,              # 10 us
        guard_band_ns=200.0,
        clock_skew_sigma_ns=20.0,
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=100.0,
        packet_bandwidth_GBs=10.0,
        packet_latency_ns=200.0,
        packet_fallback_threshold_B=10**18,   # circuit only
    ),

    # Sirius-like, but with FLOW_RESERVED reservation semantics: one circuit
    # per Phase, setup + slot alignment paid once across the whole AR phase.
    "sirius_like_circuit": TDMConfig(
        slot_length_ns=100.0,
        guard_band_ns=10.0,
        clock_skew_sigma_ns=1.0,              # in-spec; feasibility-only knob
        circuit_setup_ns=0.0,
        circuit_bandwidth_GBs=100.0,
        packet_bandwidth_GBs=10.0,
        packet_latency_ns=200.0,
        packet_fallback_threshold_B=1024,
        mode=CircuitMode.FLOW_RESERVED,
        n_parallel_circuits=1,
    ),

    # Deliberately bad: slot length >> any chunk transfer time.
    # Use to confirm slot quantization is in effect and to show
    # the regime where TDM hurts more than it helps.
    "coarse_tdm_stress": TDMConfig(
        slot_length_ns=1_000_000.0,           # 1 ms
        guard_band_ns=1_000.0,
        clock_skew_sigma_ns=0.0,
        circuit_setup_ns=10_000.0,
        circuit_bandwidth_GBs=50.0,
        packet_bandwidth_GBs=5.0,
        packet_latency_ns=1_000.0,
        packet_fallback_threshold_B=10**18,
    ),
}


def parse_overrides(spec: str | None) -> dict[str, object]:
    """Parse `--tdm-overrides 'slot_length_ns=200,clock_skew_sigma_ns=50'`.

    Values may have us/ms/s suffix on time-like fields; converted to ns.
    `mode=flow` / `mode=per_chunk` are passed through as strings and converted
    to `CircuitMode` in apply_overrides. Keys must match TDMConfig field names.
    """
    if not spec:
        return {}
    out: dict[str, object] = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"bad override (missing '='): {kv!r}")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        # mode is the only non-numeric override; preserve the string for
        # apply_overrides to coerce to CircuitMode.
        if k == "mode":
            out[k] = v
            continue
        # Allow us/ms/s suffixes for time-like fields
        mult = 1.0
        for suffix, m in (("ns", 1.0), ("us", 1e3), ("ms", 1e6), ("s", 1e9)):
            if v.endswith(suffix):
                v = v[:-len(suffix)]
                mult = m
                break
        out[k] = float(v) * mult
    return out


def apply_overrides(base: TDMConfig, overrides: dict[str, object]) -> TDMConfig:
    """Return a new TDMConfig with selected fields replaced."""
    from dataclasses import replace
    safe: dict[str, object] = {}
    for k, v in overrides.items():
        if not hasattr(base, k):
            raise ValueError(f"unknown TDMConfig field: {k!r}")
        cur = getattr(base, k)
        if isinstance(cur, CircuitMode):
            safe[k] = CircuitMode(v) if not isinstance(v, CircuitMode) else v
        elif isinstance(cur, bool):           # must precede int check
            safe[k] = bool(v)
        elif isinstance(cur, int):
            safe[k] = int(v)
        else:
            safe[k] = float(v)
    return replace(base, **safe)
