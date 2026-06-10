"""Per-rank step simulator.

Walks a Chakra ET in order, decomposes each ALLREDUCE into per-tier chunks
via collectives.py, times each chunk via tdm_model.chunk_time (or the plain
analytical timing for non-TDM tiers), then applies AstraSim-style overlap:
each layer's grad allreduce is hidden behind the NEXT layer's backward
compute (BWD_IG + BWD_WG). Monte-Carlos clock-skew sampling for P99.

Assumes the text-converter ET shape:
  forward in order:    COMP block_1_FWD ... block_N_FWD
  backward in reverse: BWD_IG[N], BWD_WG[N], ALLREDUCE[N], BWD_IG[N-1], ...
                      ..., BWD_WG[1], ALLREDUCE[1]   (block 1 has no BWD_IG)
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .collectives import Chunk, Phase, ring_allreduce_phases, total_bytes
from .et_loader import (
    ALL_REDUCE,
    COMM_COLL_NODE,
    COMP_NODE,
    EtNode,
    load_et,
)
from .scheduler import LinkScheduler
from .tdm_model import (
    ChunkTiming,
    CircuitMode,
    TDMConfig,
    analytical_chunk_time,
    chunk_time,
    flow_time,
)


@dataclass
class NetworkConfig:
    topology:      list[str]
    npus_count:    list[int]
    bandwidth_GBs: list[float]
    latency_ns:    list[float]
    impl_per_tier: list[str]                       # e.g. ["ring","ring"]
    tdm:           dict[int, TDMConfig] = field(default_factory=dict)
    # Roofline params (STAGE DAG path only; surfaced from system.json). STAGE
    # comp nodes carry num_ops/tensor_size but duration=0, so dag_sim computes
    # compute time itself with the same formula AstraSim uses.
    peak_perf_tflops:  float = 0.0     # "peak-perf" (TFLOP/s); scaled x1e12
    local_mem_bw_GBs:  float = 0.0     # "local-mem-bw" (GB/s); scaled x1e9
    roofline_enabled:  bool  = False
    # Pipeline inner and outer tiers of a hierarchical AR. With this on,
    # the collective's total time is approximately max(T_inner, T_outer)
    # instead of T_inner + T_outer + T_inner. This matches NCCL / AstraSim
    # behavior under `active-chunks-per-dimension > 1` and
    # `preferred-dataset-splits > 1`, which let inner RS/AG overlap with
    # the outer AR.
    pipelined_hierarchy: bool = True

    def n_tiers(self) -> int:
        return len(self.npus_count)

    def n_ranks(self) -> int:
        n = 1
        for c in self.npus_count:
            n *= c
        return n


@dataclass
class PerLayer:
    block:           int
    fwd_us:          int
    bwd_ig_us:       int
    bwd_wg_us:       int
    comm_size_bytes: int


_NAME_RE = re.compile(
    r"(?:COMP_NODE|COMM_COLL_NODE)_block_(\d+)_(FWD|BWD_IG|BWD_WG|ALLREDUCE)")


def parse_et_to_layers(nodes: list[EtNode]) -> list[PerLayer]:
    """Extract per-block (fwd, bwd_ig, bwd_wg, allreduce_size) from a Chakra ET.

    Iterates the node list once and fills a dict keyed by block id. Layers
    are returned in ascending block order (1..N).
    """
    by_block: dict[int, dict] = {}
    for n in nodes:
        m = _NAME_RE.search(n.name)
        if not m:
            continue
        block = int(m.group(1))
        phase = m.group(2)
        by_block.setdefault(block, {})
        if phase == "FWD":
            by_block[block]["fwd_us"] = n.duration_us
        elif phase == "BWD_IG":
            by_block[block]["bwd_ig_us"] = n.duration_us
        elif phase == "BWD_WG":
            by_block[block]["bwd_wg_us"] = n.duration_us
        elif phase == "ALLREDUCE":
            assert n.type == COMM_COLL_NODE
            assert n.comm_type == ALL_REDUCE, (
                f"only ALLREDUCE supported, got comm_type={n.comm_type}")
            by_block[block]["comm_size_bytes"] = n.comm_size

    layers: list[PerLayer] = []
    for block in sorted(by_block):
        d = by_block[block]
        layers.append(PerLayer(
            block=block,
            fwd_us=d.get("fwd_us", 0),
            bwd_ig_us=d.get("bwd_ig_us", 0),   # block 1 lacks this
            bwd_wg_us=d.get("bwd_wg_us", 0),
            comm_size_bytes=d.get("comm_size_bytes", 0),
        ))
    return layers


def time_chunk(c: Chunk, net: NetworkConfig, rng) -> ChunkTiming:
    """Dispatch one chunk to the appropriate link model for its tier."""
    if c.tier in net.tdm:
        return chunk_time(c.bytes, net.tdm[c.tier], rng)
    return analytical_chunk_time(
        c.bytes, net.bandwidth_GBs[c.tier], net.latency_ns[c.tier])


@dataclass
class TrialResult:
    step_ns:           float
    compute_ns:        float
    total_comm_ns:     float
    exposed_comm_ns:   float
    hidden_comm_ns:    float
    chunk_timings:     list[ChunkTiming]


def _any_flow_reserved(net: NetworkConfig) -> bool:
    return any(cfg.mode == CircuitMode.FLOW_RESERVED for cfg in net.tdm.values())


def simulate_one_trial(layers: list[PerLayer], net: NetworkConfig,
                       rng) -> TrialResult:
    """Dispatch by the strongest scheduling primitive any tier requires.

    - If any tier has CircuitMode.FLOW_RESERVED, use the timeline path that
      tracks per-tier LinkSchedulers and computes step end from an explicit
      event sequence.
    - Otherwise use the original layer-by-layer formula
      `step = compute + sum_layer max(0, comm - prev_bwd)` -- preserves
      existing behavior and the AstraSim regression baseline.
    """
    if _any_flow_reserved(net):
        return _simulate_trial_timeline(layers, net, rng)
    return _simulate_trial_legacy(layers, net, rng)


def _simulate_trial_legacy(layers: list[PerLayer], net: NetworkConfig,
                           rng) -> TrialResult:
    """Original per-chunk model. Deterministic now that skew jitter is gone
    -- the `rng` parameter is unused but kept for API stability."""
    compute_ns = sum(L.fwd_us + L.bwd_ig_us + L.bwd_wg_us
                     for L in layers) * 1e3   # us -> ns

    chunk_timings: list[ChunkTiming] = []
    comm_per_layer_ns: list[float] = []
    for L in layers:
        phases = ring_allreduce_phases(
            L.comm_size_bytes, net.npus_count, net.impl_per_tier,
            topology_per_tier=net.topology)
        phase_times_ns: list[float] = []
        for ph in phases:
            t = 0.0
            for c in ph.chunks:
                ct = time_chunk(c, net, rng)
                t += ct.time_ns
                chunk_timings.append(ct)
            phase_times_ns.append(t)
        if net.pipelined_hierarchy and len(phase_times_ns) > 1:
            layer_comm_ns = max(phase_times_ns)
        else:
            layer_comm_ns = sum(phase_times_ns)
        comm_per_layer_ns.append(layer_comm_ns)

    exposed_total_ns = 0.0
    for idx, L in enumerate(layers):
        if L.block == 1:
            hideable_ns = 0.0
        else:
            prev = next((Lp for Lp in layers if Lp.block == L.block - 1), None)
            hideable_ns = ((prev.bwd_ig_us + prev.bwd_wg_us) * 1e3
                           if prev is not None else 0.0)
        exposed_total_ns += max(0.0, comm_per_layer_ns[idx] - hideable_ns)

    total_comm_ns = float(sum(comm_per_layer_ns))
    hidden_ns = total_comm_ns - exposed_total_ns
    step_ns = compute_ns + exposed_total_ns
    return TrialResult(
        step_ns=step_ns,
        compute_ns=compute_ns,
        total_comm_ns=total_comm_ns,
        exposed_comm_ns=exposed_total_ns,
        hidden_comm_ns=hidden_ns,
        chunk_timings=chunk_timings,
    )


def _simulate_trial_timeline(layers: list[PerLayer], net: NetworkConfig,
                             rng) -> TrialResult:
    """Timeline path: walk events, schedule each AR Phase on its tier's
    LinkScheduler. Step end is `max(t_bwd_end_of_block_1, max(AR_end))`.

    AR[N] is launched at the end of BWD compute for block N (the layer's
    own gradients are ready). Subsequent BWDs proceed serially; ARs run
    concurrently with later BWDs except when LinkScheduler enforces
    serialization on a tier (e.g. n_parallel_circuits=1 and AR[N] still
    holding the circuit when AR[N-1] arrives).
    """
    compute_ns = sum(L.fwd_us + L.bwd_ig_us + L.bwd_wg_us
                     for L in layers) * 1e3

    schedulers: dict[int, LinkScheduler] = {
        tier: LinkScheduler(n_parallel=cfg.n_parallel_circuits)
        for tier, cfg in net.tdm.items()
        if cfg.mode == CircuitMode.FLOW_RESERVED
    }

    t_fwd_end = sum(L.fwd_us for L in layers) * 1e3

    # Walk backward in block order: N, N-1, ..., 1. BWD compute is serial.
    sorted_layers = sorted(layers, key=lambda L: -L.block)
    t_bwd_cursor = t_fwd_end

    chunk_timings: list[ChunkTiming] = []
    ar_ends: list[float] = []

    for L in sorted_layers:
        bwd_dur_ns = (L.bwd_ig_us + L.bwd_wg_us) * 1e3
        t_bwd_end = t_bwd_cursor + bwd_dur_ns
        t_bwd_cursor = t_bwd_end

        phases = ring_allreduce_phases(
            L.comm_size_bytes, net.npus_count, net.impl_per_tier,
            topology_per_tier=net.topology)
        if not phases:
            continue

        ar_arrival_ns = t_bwd_end
        phase_ends: list[float] = []
        seq_cursor_ns = ar_arrival_ns      # for sequential pipelining

        for ph in phases:
            tier = ph.tier
            arrival_ns = ar_arrival_ns if net.pipelined_hierarchy else seq_cursor_ns

            if tier in schedulers:
                ft = flow_time(total_bytes(ph.chunks), net.tdm[tier])
                start_ns, end_ns, _ch = schedulers[tier].schedule(
                    arrival_ns, ft.duration_ns)
                chunk_timings.append(ChunkTiming(
                    ft.duration_ns, ft.mode, ft.bytes, ft.n_slots,
                    ft.guard_waste_ns))
            elif tier in net.tdm:
                # Per-chunk pacing tier inside an otherwise FLOW_RESERVED net:
                # sum chunk_time across the phase. No per-rank link timeline
                # (PER_CHUNK semantics) -- it just runs from arrival.
                t = 0.0
                for c in ph.chunks:
                    ct = chunk_time(c.bytes, net.tdm[tier], rng)
                    t += ct.time_ns
                    chunk_timings.append(ct)
                start_ns = arrival_ns
                end_ns = arrival_ns + t
            else:
                tot = total_bytes(ph.chunks)
                if tot <= 0:
                    start_ns = arrival_ns
                    end_ns = arrival_ns
                else:
                    t = tot / net.bandwidth_GBs[tier] + net.latency_ns[tier]
                    chunk_timings.append(ChunkTiming(
                        t, "packet", tot, 0, 0.0))
                    start_ns = arrival_ns
                    end_ns = arrival_ns + t

            phase_ends.append(end_ns)
            seq_cursor_ns = end_ns

        if not phase_ends:
            continue
        if net.pipelined_hierarchy and len(phase_ends) > 1:
            ar_end_ns = max(phase_ends)
        else:
            ar_end_ns = phase_ends[-1]
        ar_ends.append(ar_end_ns)

    step_ns = max(t_bwd_cursor, max(ar_ends) if ar_ends else t_bwd_cursor)
    total_comm_ns = float(sum(ct.time_ns for ct in chunk_timings))
    exposed_comm_ns = max(0.0, step_ns - compute_ns)
    hidden_ns = max(0.0, total_comm_ns - exposed_comm_ns)
    return TrialResult(
        step_ns=step_ns,
        compute_ns=compute_ns,
        total_comm_ns=total_comm_ns,
        exposed_comm_ns=exposed_comm_ns,
        hidden_comm_ns=hidden_ns,
        chunk_timings=chunk_timings,
    )


@dataclass
class RankResult:
    rank:        int
    trials:      list[TrialResult]

    def step_ns(self, q: float) -> float:
        return _quantile([t.step_ns for t in self.trials], q)

    def comm_ns(self, q: float) -> float:
        return _quantile([t.total_comm_ns for t in self.trials], q)

    def exposed_ns(self, q: float) -> float:
        return _quantile([t.exposed_comm_ns for t in self.trials], q)


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    return float(np.quantile(xs, q))


def simulate_rank(et_path: Path, net: NetworkConfig, trials: int,
                  seed: int) -> RankResult:
    nodes = load_et(et_path)
    layers = parse_et_to_layers(nodes)
    rng = np.random.default_rng(seed)
    trial_results = [simulate_one_trial(layers, net, rng) for _ in range(trials)]
    return RankResult(rank=_rank_from_path(et_path), trials=trial_results)


def _rank_from_path(path: Path) -> int:
    # Files look like <basename>.<rank>.et
    stem = path.name
    parts = stem.split(".")
    for part in reversed(parts[:-1]):
        if part.isdigit():
            return int(part)
    return 0


def simulate_all_ranks(et_dir: Path, num_npus: int, net: NetworkConfig,
                       trials: int, base_seed: int = 1) -> list[RankResult]:
    # Find files of the form <basename>.<rank>.et
    et_files = sorted(et_dir.glob("*.et"),
                      key=lambda p: _rank_from_path(p))
    if not et_files:
        raise FileNotFoundError(f"no .et files in {et_dir}")
    # Filter to the configured ranks (in case extra files lurk).
    keep: dict[int, Path] = {}
    for p in et_files:
        r = _rank_from_path(p)
        if 0 <= r < num_npus:
            keep[r] = p
    if len(keep) != num_npus:
        raise FileNotFoundError(
            f"expected {num_npus} ET files in {et_dir}, found ranks {sorted(keep)}")
    results = []
    for r in range(num_npus):
        results.append(simulate_rank(keep[r], net, trials, seed=base_seed + r))
    return results


# Aggregate stats helpers ----------------------------------------------------

def aggregate_chunk_stats(results: list[RankResult]
                          ) -> dict[str, float]:
    """Across all ranks * trials * chunks: mode mix, guard waste, effective BW."""
    total_bytes = 0
    circuit_bytes = 0
    circuit_time_ns = 0.0
    total_guard_ns = 0.0
    total_comm_ns = 0.0
    for rr in results:
        for tr in rr.trials:
            for ct in tr.chunk_timings:
                total_bytes += ct.bytes
                if ct.mode == "circuit" and ct.time_ns > 0:
                    circuit_bytes += ct.bytes
                    circuit_time_ns += ct.time_ns
                    total_guard_ns += ct.guard_waste_ns
                total_comm_ns += ct.time_ns
    # Effective circuit bandwidth = total bytes delivered / total time the
    # circuit was occupied. 1 byte/ns == 1 GB/s.
    effective_circuit_GBs = (
        circuit_bytes / circuit_time_ns if circuit_time_ns > 0 else float("nan"))
    return {
        "bytes_circuit_pct":     (100.0 * circuit_bytes / total_bytes) if total_bytes else 0.0,
        "effective_circuit_GBs": effective_circuit_GBs,
        "guard_waste_frac":      (total_guard_ns / total_comm_ns) if total_comm_ns else 0.0,
    }


def compute_ideal_step_ns(layers: list[PerLayer]) -> float:
    """Compute-only step time -- the lower bound (perfect network)."""
    return sum(L.fwd_us + L.bwd_ig_us + L.bwd_wg_us for L in layers) * 1e3
