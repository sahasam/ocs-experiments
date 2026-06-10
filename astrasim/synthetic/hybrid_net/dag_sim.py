"""Per-rank general-DAG step simulator for STAGE Chakra ETs.

The PerLayer path in simulate.py assumes a DP-only shape (one all-reduce per
transformer block, hidden behind the previous block's backward). STAGE traces
are a general fine-grained DAG -- COMP + COMM_COLL (all-reduce / all-gather /
reduce-scatter / all-to-all) + point-to-point SEND/RECV, wired by data_deps --
so TP collectives and PP send/recv (the inter-node traffic the OCS study cares
about) cannot be expressed as PerLayer. This module simulates that DAG directly.

Model (per rank, ranks simulated independently -- see scheduler.py docstring on
why DP symmetry makes that defensible; cross-rank collective coupling is left to
AstraSim and an optional future --collective-barrier mode):

  * COMP nodes get a roofline duration from num_ops/tensor_size, using the same
    formula + unit scaling AstraSim uses (validated to 0.05% on the 64-rank 8B
    trace: sum = 595.7 ms vs AstraSim GPU 596 ms).
  * COMM_COLL nodes are decomposed with collectives.direct_collective_phases
    (the parallel / FullyConnected cost model that matches AstraSim's "direct"
    all-reduce; the ring model overshoots ~7x) and timed through the existing
    TDM / analytical link machinery, so every OCS knob carries over.
  * SEND/RECV are point-to-point transfers on the inter-node (OCS) tier.

Resources: one serial compute stream + one serial link per network tier (or a
LinkScheduler with n_parallel circuits for FLOW_RESERVED tiers). Compute and
each tier run concurrently, gated by data deps -- so comm/compute overlap
*emerges from the DAG* instead of a hardcoded "hide behind previous backward"
rule. Step time = critical path; exposed comm = step - compute (exactly
AstraSim's Wall = GPU + exposed identity).
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np

from .collectives import Chunk, direct_collective_phases
from .et_loader import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    REDUCE_SCATTER,
    EtNode,
    load_et,
)
from .scheduler import LinkScheduler
from .simulate import (
    NetworkConfig,
    RankResult,
    TrialResult,
    _rank_from_path,
    time_chunk,
)
from .tdm_model import ChunkTiming, CircuitMode

# Map Chakra CollectiveCommType -> the kind string direct_collective_phases wants.
_COMM_KIND = {
    ALL_REDUCE:     "all_reduce",
    ALL_GATHER:     "all_gather",
    REDUCE_SCATTER: "reduce_scatter",
    ALL_TO_ALL:     "all_to_all",
}


def roofline_duration_ns(num_ops: int | None, tensor_size: int | None,
                         peak_perf_tflops: float, local_mem_bw_GBs: float
                         ) -> float:
    """Roofline compute time in ns, matching AstraSim (Sys.cc / Workload.cc).

    perf = min(local_mem_bw * (num_ops/tensor_size), peak_perf); t = num_ops/perf.
    peak-perf is TFLOP/s (x1e12 -> FLOP/s); local-mem-bw is GB/s (x1e9 -> B/s).
    """
    if not num_ops or not tensor_size:
        return 0.0
    peak = peak_perf_tflops * 1e12
    membw = local_mem_bw_GBs * 1e9
    if peak <= 0 or membw <= 0:
        return 0.0
    oi = num_ops / tensor_size
    perf = min(membw * oi, peak)
    return (num_ops / perf) * 1e9


def load_comm_groups(path: Path) -> dict[str, list[int]]:
    """Load comm_group.json: {pg_name: [rank, ...]}."""
    return {str(k): list(v) for k, v in json.loads(Path(path).read_text()).items()}


def tier_for_collective(group_size: int, net: NetworkConfig,
                        tier_for_size: dict[int, int] | None = None) -> int:
    """Pick the network tier a collective of `group_size` ranks runs on.

    Membership-driven, not a fixed rule:
      * explicit --tier-for-size override wins;
      * a flat 1-tier net puts everything on tier 0 (the AstraSim flat baseline);
      * otherwise a collective spanning exactly the inner tier's NPU count
        (npus_count[0], e.g. an 8-rank TP group on an 8-GPU node) maps to the
        inner tier (NVLink); larger/other groups map to the outermost (OCS) tier.
    """
    if tier_for_size and group_size in tier_for_size:
        return tier_for_size[group_size]
    if net.n_tiers() <= 1:
        return 0
    if group_size == net.npus_count[0]:
        return 0
    return net.n_tiers() - 1


def _pp_tier(net: NetworkConfig) -> int:
    """Tier for PP point-to-point: the outermost (inter-node / OCS) tier."""
    return net.n_tiers() - 1


def _node_duration(node: EtNode, net: NetworkConfig, groups: dict[str, list[int]],
                   tier_for_size: dict[int, int] | None, rng,
                   pp_recv_stall_ns: float
                   ) -> tuple[float, str | tuple[str, int], list[ChunkTiming]]:
    """Return (duration_ns, resource_key, chunk_timings) for one node.

    resource_key is "compute" for COMP, ("net", tier) for comm/p2p, or None for
    zero-cost structural nodes (they only propagate dependencies).
    """
    t = node.type
    if t == COMP_NODE:
        dur = roofline_duration_ns(node.num_ops, node.tensor_size,
                                   net.peak_perf_tflops, net.local_mem_bw_GBs)
        return dur, "compute", []

    if t == COMM_COLL_NODE:
        ranks = groups.get(node.pg_name or "", [])
        gsize = len(ranks)
        tier = tier_for_collective(gsize, net, tier_for_size)
        kind = _COMM_KIND.get(node.comm_type)
        if kind is None or gsize <= 1 or not node.comm_size:
            return 0.0, ("net", tier), []
        phases = direct_collective_phases(kind, node.comm_size, gsize, tier)
        dur = 0.0
        cts: list[ChunkTiming] = []
        for ph in phases:                 # RS then AG run sequentially
            for c in ph.chunks:
                ct = time_chunk(c, net, rng)
                dur += ct.time_ns
                cts.append(ct)
        return dur, ("net", tier), cts

    if t in (COMM_SEND_NODE, COMM_RECV_NODE):
        tier = _pp_tier(net)
        if not node.comm_size:
            return 0.0, ("net", tier), []
        ct = time_chunk(Chunk(tier, node.comm_size), net, rng)
        dur = ct.time_ns
        if t == COMM_RECV_NODE:
            dur += pp_recv_stall_ns
        return dur, ("net", tier), [ct]

    # METADATA / MEM nodes: zero cost, dependency-only.
    return 0.0, None, []


def simulate_rank_dag(nodes: list[EtNode], net: NetworkConfig,
                      groups: dict[str, list[int]], rng,
                      tier_for_size: dict[int, int] | None = None,
                      pp_recv_stall_ns: float = 0.0) -> TrialResult:
    """List-schedule one rank's DAG -> TrialResult.

    Event-driven: a node becomes ready when all its data+ctrl deps have finished;
    among ready nodes the earliest-ready (id tie-break) is scheduled next, taking
    its resource's next free slot. Step time = max finish over all nodes.
    """
    by_id = {n.id: n for n in nodes}
    # Precompute durations / resources / chunk timings once (deterministic).
    dur: dict[int, float] = {}
    res: dict[int, str | tuple[str, int] | None] = {}
    chunk_timings: list[ChunkTiming] = []
    compute_ns = 0.0
    total_comm_ns = 0.0
    for n in nodes:
        d, r, cts = _node_duration(n, net, groups, tier_for_size, rng,
                                   pp_recv_stall_ns)
        dur[n.id] = d
        res[n.id] = r
        chunk_timings.extend(cts)
        if r == "compute":
            compute_ns += d
        elif isinstance(r, tuple):
            total_comm_ns += d

    # Build dependency structure (only deps that exist in this rank's node set).
    deps: dict[int, list[int]] = {}
    succs: dict[int, list[int]] = {nid: [] for nid in by_id}
    pending: dict[int, int] = {}
    for n in nodes:
        ds = [d for d in (list(n.data_deps) + list(n.ctrl_deps)) if d in by_id]
        deps[n.id] = ds
        pending[n.id] = len(ds)
        for d in ds:
            succs[d].append(n.id)

    # Resources: serial compute stream + a LinkScheduler per network tier.
    compute_free = 0.0
    link: dict[int, LinkScheduler] = {}

    def _scheduler_for(tier: int) -> LinkScheduler:
        if tier not in link:
            cfg = net.tdm.get(tier)
            n_par = (cfg.n_parallel_circuits
                     if cfg is not None and cfg.mode == CircuitMode.FLOW_RESERVED
                     else 1)
            link[tier] = LinkScheduler(n_parallel=n_par)
        return link[tier]

    finish: dict[int, float] = {}
    # Priority queue of (ready_time, id) for nodes whose deps are all satisfied.
    pq: list[tuple[float, int]] = []
    for nid, p in pending.items():
        if p == 0:
            heapq.heappush(pq, (0.0, nid))

    while pq:
        ready_ns, nid = heapq.heappop(pq)
        r = res[nid]
        if r == "compute":
            start = max(ready_ns, compute_free)
            end = start + dur[nid]
            compute_free = end
        elif isinstance(r, tuple):
            tier = r[1]
            start, end, _ch = _scheduler_for(tier).schedule(ready_ns, dur[nid])
        else:
            start = end = ready_ns
        finish[nid] = end
        for s in succs[nid]:
            pending[s] -= 1
            if pending[s] == 0:
                rt = max((finish[d] for d in deps[s]), default=0.0)
                heapq.heappush(pq, (rt, s))

    if len(finish) != len(by_id):
        # A cycle (or dep pointing outside the set was mis-handled): fail loud.
        raise ValueError(
            f"DAG schedule incomplete: {len(finish)}/{len(by_id)} nodes "
            f"(possible cycle)")

    step_ns = max(finish.values()) if finish else 0.0
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


def simulate_all_ranks_dag(et_dir: Path, num_npus: int, net: NetworkConfig,
                           groups: dict[str, list[int]], trials: int,
                           base_seed: int = 1,
                           tier_for_size: dict[int, int] | None = None,
                           pp_recv_stall_ns: float = 0.0) -> list[RankResult]:
    """Simulate every rank's DAG independently. Mirrors simulate_all_ranks."""
    et_files = sorted(Path(et_dir).glob("*.et"), key=_rank_from_path)
    if not et_files:
        raise FileNotFoundError(f"no .et files in {et_dir}")
    keep: dict[int, Path] = {}
    for p in et_files:
        r = _rank_from_path(p)
        if 0 <= r < num_npus:
            keep[r] = p
    if len(keep) != num_npus:
        raise FileNotFoundError(
            f"expected {num_npus} ET files in {et_dir}, found ranks {sorted(keep)}")
    results: list[RankResult] = []
    for r in range(num_npus):
        nodes = load_et(keep[r])
        rng = np.random.default_rng(base_seed + r)
        trs = [simulate_rank_dag(nodes, net, groups, rng,
                                 tier_for_size=tier_for_size,
                                 pp_recv_stall_ns=pp_recv_stall_ns)
               for _ in range(trials)]
        results.append(RankResult(rank=r, trials=trs))
    return results


# --- Pre-calculation gate (independent of AstraSim) -------------------------

def rank_roofline_compute_ns(nodes: list[EtNode], peak_perf_tflops: float,
                             local_mem_bw_GBs: float) -> float:
    """Sum of roofline COMP durations for one rank (hand-checkable)."""
    return sum(roofline_duration_ns(n.num_ops, n.tensor_size,
                                    peak_perf_tflops, local_mem_bw_GBs)
               for n in nodes if n.type == COMP_NODE)


def rank_comm_volume_bytes(nodes: list[EtNode], groups: dict[str, list[int]]
                           ) -> int:
    """Per-rank collective + p2p byte volume implied by the direct model.

    AR -> 2*P/n, RS/AG -> P/n, A2A -> P*(n-1)/n, SEND/RECV -> P. Independent of
    timing -- used to prove the loader reads comm_size/pg_name correctly."""
    total = 0
    for n in nodes:
        if n.type == COMM_COLL_NODE:
            gsize = len(groups.get(n.pg_name or "", []))
            kind = _COMM_KIND.get(n.comm_type)
            if kind is None or gsize <= 1 or not n.comm_size:
                continue
            for ph in direct_collective_phases(kind, n.comm_size, gsize, tier=0):
                total += sum(c.bytes for c in ph.chunks)
        elif n.type in (COMM_SEND_NODE, COMM_RECV_NODE) and n.comm_size:
            total += n.comm_size
    return total
