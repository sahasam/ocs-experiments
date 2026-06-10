"""Collective-algorithm decomposition: collective -> list of per-tier chunks.

We model only ring all-reduce, in two flavors:

  flat:           N ranks on a single tier, 2(N-1) chunks of P/N each
  hierarchical:   [n_inner, n_outer] tiers
                  - inner reduce-scatter: 2(n_inner-1) chunks of P/n_inner on tier 0
                  - outer allreduce:      2(n_outer-1) chunks of P/(n_inner*n_outer) on tier 1
                  - inner all-gather:     2(n_inner-1) chunks of P/n_inner on tier 0

The "hierarchical" decomposition is exactly what explains the 16-GPU paradox
(outer tier carries 1/n_inner the payload), so the model can reproduce
AstraSim's hierarchical-ring numbers.

Each chunk is (tier_index, bytes). simulate.py times each chunk via the
appropriate link model (TDM if a TDMConfig is set for that tier, else
analytical bytes / bw).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Chunk:
    tier:  int
    bytes: int


@dataclass
class Phase:
    """A pipeline-able phase of a collective (e.g. inner RS, outer AR, inner AG).

    All chunks within a phase share the same tier, but a phase may contain
    multiple chunks (one per ring step). In pipelined-hierarchy mode, the
    total time of an allreduce is max(time_of_each_phase); in sequential
    mode it is sum.
    """
    name:   str
    tier:   int
    chunks: list["Chunk"]


def ring_allreduce_phases(payload_bytes: int, npus_count: list[int],
                          impl_per_tier: list[str],
                          topology_per_tier: list[str] | None = None
                          ) -> list[Phase]:
    """Return the ordered list of phases (each containing chunks).

    For 1-tier flat ring: one Phase named "ar" with 2(n-1) chunks.
    For 2-tier hierarchical: three Phases ["inner_rs", "outer_ar", "inner_ag"]
    that can pipeline -- a pipelined scheduler will execute them concurrently
    and the layer time is max of the three.

    topology_per_tier: AstraSim YAML "topology" vector, e.g.
        ["Switch", "Ring"]. If a tier is a Switch, the AR completes in
        a single step with (n-1)/n * payload bytes per port (rather than
        2(n-1) ring steps). impl is ignored for Switch tiers since the
        switch reduction is implementation-agnostic in the analytical
        model.

    npus_count + impl_per_tier are the AstraSim system-config vectors;
    e.g. npus_count=[8,2], impl_per_tier=["ring","ring"] -> hierarchical
    ring. npus_count=[8], impl_per_tier=["ring"] -> flat 8-rank ring.
    """
    if len(npus_count) != len(impl_per_tier):
        raise ValueError("npus_count and impl_per_tier must have same length")
    if topology_per_tier is None:
        topology_per_tier = ["Ring"] * len(npus_count)
    if len(topology_per_tier) != len(npus_count):
        raise ValueError("topology_per_tier must match npus_count length")

    for impl, topo in zip(impl_per_tier, topology_per_tier):
        if topo != "Switch" and impl != "ring":
            raise NotImplementedError(
                f"only 'ring' supported on non-Switch tiers; got impl={impl!r}")

    if len(npus_count) == 1:
        chunks = _tier_allreduce_chunks(
            payload_bytes, npus_count[0], topology_per_tier[0], tier=0)
        return [Phase(name="ar", tier=0, chunks=chunks)]
    if len(npus_count) == 2:
        return _hierarchical_allreduce(payload_bytes,
                                       npus_count, topology_per_tier)
    raise NotImplementedError(
        f"only 1- or 2-tier topologies supported; got {len(npus_count)} tiers")


# Back-compat shim: previous callers asked for a flat list of chunks.
def ring_allreduce_chunks(payload_bytes: int, npus_count: list[int],
                          impl_per_tier: list[str],
                          topology_per_tier: list[str] | None = None
                          ) -> list[Chunk]:
    """Flat list of all chunks, ignoring phase structure (legacy callers)."""
    phases = ring_allreduce_phases(
        payload_bytes, npus_count, impl_per_tier, topology_per_tier)
    out: list[Chunk] = []
    for p in phases:
        out.extend(p.chunks)
    return out


def _tier_allreduce_chunks(payload_bytes: int, n: int, topology: str,
                           tier: int) -> list[Chunk]:
    """Decompose one all-reduce phase on a tier with `n` peers."""
    if n <= 1:
        return []
    if topology == "Switch":
        # On a Switch, each NPU's port carries (n-1)/n * payload in one
        # combined reduce+broadcast pass through the switch. Model as a
        # single chunk; the switch reduces in-network at no extra cost.
        switch_chunk = (payload_bytes * (n - 1)) // n
        return [Chunk(tier, switch_chunk)]
    # Default: ring AR = 2(n-1) steps of payload/n bytes per link.
    chunk_bytes = payload_bytes // n
    return [Chunk(tier, chunk_bytes) for _ in range(2 * (n - 1))]


def _tier_reduce_scatter_chunks(payload_bytes: int, n: int, topology: str,
                                tier: int) -> list[Chunk]:
    """Reduce-scatter only (half of a ring AR)."""
    if n <= 1:
        return []
    if topology == "Switch":
        # On a Switch, RS is a single upload of (n-1)/n * payload per port.
        switch_chunk = (payload_bytes * (n - 1)) // n
        return [Chunk(tier, switch_chunk)]
    chunk_bytes = payload_bytes // n
    return [Chunk(tier, chunk_bytes) for _ in range(n - 1)]


def _tier_all_gather_chunks(payload_bytes: int, n: int, topology: str,
                            tier: int) -> list[Chunk]:
    """All-gather only (the other half of a ring AR)."""
    return _tier_reduce_scatter_chunks(payload_bytes, n, topology, tier)


def _hierarchical_allreduce(payload_bytes: int, npus_count: list[int],
                            topology_per_tier: list[str]) -> list[Phase]:
    n_inner, n_outer = npus_count
    topo_inner, topo_outer = topology_per_tier
    outer_payload = payload_bytes // n_inner
    return [
        Phase(name="inner_rs", tier=0,
              chunks=_tier_reduce_scatter_chunks(
                  payload_bytes, n_inner, topo_inner, tier=0)),
        Phase(name="outer_ar", tier=1,
              chunks=_tier_allreduce_chunks(
                  outer_payload, n_outer, topo_outer, tier=1)),
        Phase(name="inner_ag", tier=0,
              chunks=_tier_all_gather_chunks(
                  payload_bytes, n_inner, topo_inner, tier=0)),
    ]


def direct_collective_phases(kind: str, payload_bytes: int, group_size: int,
                             tier: int) -> list[Phase]:
    """Direct/parallel collective decomposition for a flat FullyConnected fabric.

    This is the cost model that matches AstraSim's "direct" all-reduce on a
    FullyConnected topology (which is what STAGE was validated against), and it
    is fundamentally different from ring_allreduce_phases:

      ring (ring-on-torus):  2(n-1) chunks of P/n that traverse the ring serially
      direct (FullyConnected): every rank exchanges with every other AT ONCE, so a
        phase is a SINGLE parallel transfer of P/n bytes per rank.

    On STAGE traces the ring model overshoots comm time ~7x; the direct model
    matches AstraSim to ~1.6%. We keep ring_allreduce_phases untouched for the
    legacy PerLayer path and use this for the STAGE DAG path.

    Per-phase, per-rank bytes (n = group_size, P = payload_bytes):
      all_reduce      -> 2 phases (reduce-scatter then all-gather), each P/n
      reduce_scatter  -> 1 phase  P/n
      all_gather      -> 1 phase  P/n
      all_to_all      -> 1 phase  P*(n-1)/n

    Each phase carries one Chunk so the existing per-chunk / flow timers apply
    unchanged (TDM knobs, analytical bytes/bw). Returns [] for n <= 1 (a
    singleton group is a no-op, e.g. FSDP sharding placeholders in STAGE).
    """
    if group_size <= 1 or payload_bytes <= 0:
        return []
    n = group_size
    per = payload_bytes // n
    if kind == "all_reduce":
        return [Phase(name="rs", tier=tier, chunks=[Chunk(tier, per)]),
                Phase(name="ag", tier=tier, chunks=[Chunk(tier, per)])]
    if kind == "reduce_scatter":
        return [Phase(name="rs", tier=tier, chunks=[Chunk(tier, per)])]
    if kind == "all_gather":
        return [Phase(name="ag", tier=tier, chunks=[Chunk(tier, per)])]
    if kind == "all_to_all":
        return [Phase(name="a2a", tier=tier,
                      chunks=[Chunk(tier, (payload_bytes * (n - 1)) // n)])]
    raise ValueError(f"unknown collective kind {kind!r}")


def total_bytes(chunks: list[Chunk]) -> int:
    return sum(c.bytes for c in chunks)


def n_pairs_for_phase(phase: "Phase", npus_count: list[int],
                      topology_per_tier: list[str]) -> int:
    """Number of simultaneous TX-RX pairs active during one slot of `phase`.

    Used by feasibility checks: more pairs => P_step_collision compounds as
    1 - (1 - P_pair)^n_pairs.

    Switch tiers are treated as 1 logical pair (in-network reduction). Ring
    tiers contribute n_ranks pairs (each rank simultaneously sends to its
    neighbour).
    """
    return 1 if topology_per_tier[phase.tier] == "Switch" else npus_count[phase.tier]


def ring_allreduce_byte_volume(payload_bytes: int, n_ranks: int) -> int:
    """Theoretical per-rank byte volume of a flat ring AR.

    Used by the mass-conservation sanity check:
      flat ring on N ranks, payload P  =>  2 * (N-1) * P/N bytes per rank.
    """
    if n_ranks <= 1:
        return 0
    return 2 * (n_ranks - 1) * (payload_bytes // n_ranks)
