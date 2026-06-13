"""Coupled forward DAG scheduler -- the artifact-free OCS engine.

`dag_sim` has the right per-node cost model (roofline COMP, *direct* collective
decomposition, p2p sends) but schedules each rank independently, so it misses the
PP coupling where stage 0 idles until stage 1's backward gradient lands. `ocs_replay`
has the cross-rank coupling but is a delay propagator seeded by an AstraSim trace --
it can never make a node finish *earlier* than the trace, so it cannot turn a ring
trace into a (faster) direct result. See [[why-redesign-ocs-experiment]].

This module is the missing third engine: dag_sim's cost model run on a *global*
event-driven list scheduler over every rank's ET at once, with the two cross-rank
edge types `ocs_replay.build_graph` already knows how to extract:

  * PP send->recv: a recv completes when its matched SEND (matched on src,dst,tag)
    completes -- so the PP bubble *emerges* from the dependency graph instead of the
    hardcoded `pp_recv_stall` hack in dag_sim.
  * collective barrier: a collective instance (same pg_name+name across its group)
    starts when its slowest member is ready and all members finish together.

The collective cost model is switchable (`collective_impl`):
  * "direct"  -> direct_collective_phases  (FullyConnected / OCS: bandwidth-optimal,
                 the headline OCS algorithm)
  * "ring"    -> 2(g-1) chunks of P/g    (the PS algorithm, for the Exp4 gate)

Two self-validation gates (see simulate_coupled docstring):
  A. at infinite bandwidth the step must collapse to the compute+latency ideal floor.
  B. with collective_impl="ring" at 50 GB/s it must reproduce Exp4's coupled OCS
     numbers (stage0 ~4783 ms, stage1 ~3203 ms); direct must come in <= ring.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .collectives import Chunk, direct_collective_phases
from .dag_sim import _COMM_KIND, roofline_duration_ns, tier_for_collective
from .et_loader import (
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    load_et,
)
from .scheduler import LinkScheduler
from .simulate import NetworkConfig, time_chunk


@dataclass
class CNode:
    """One node in the global (all-ranks) DAG."""
    rank:   int
    nid:    int
    type:   int
    name:   str
    dur:    float                       # ns; 0 for recv/structural
    res:    tuple | None                # ("compute",rank) | ("net",rank,tier) | None
    deps:   list[tuple[int, int]] = field(default_factory=list)  # intra-rank preds
    sr_pred: tuple[int, int] | None = None   # recv -> matched send
    coll:   tuple | None = None         # (pg_name, name) instance key, or None


@dataclass
class CoupledGraph:
    nodes: dict[tuple[int, int], CNode]
    coll_instances: dict[tuple, list[tuple[int, int]]]   # key -> member (rank,nid)


def _collective_chunks(kind: str, payload: int, gsize: int, tier: int,
                       impl: str) -> list[Chunk]:
    """Per-rank chunk list for one collective under `impl` ("direct"|"ring")."""
    if gsize <= 1 or not payload:
        return []
    if impl == "direct":
        return [c for ph in direct_collective_phases(kind, payload, gsize, tier)
                for c in ph.chunks]
    if impl == "ring":
        per = payload // gsize
        if kind == "all_reduce":
            return [Chunk(tier, per) for _ in range(2 * (gsize - 1))]
        if kind in ("reduce_scatter", "all_gather"):
            return [Chunk(tier, per) for _ in range(gsize - 1)]
        if kind == "all_to_all":
            return [Chunk(tier, (payload * (gsize - 1)) // gsize)]
        raise ValueError(f"unknown collective kind {kind!r}")
    raise ValueError(f"unknown collective_impl {impl!r}")


def build_coupled_graph(et_dir: Path, workload: str, num_ranks: int,
                        groups: dict[str, list[int]], net: NetworkConfig,
                        rng, *, collective_impl: str = "direct",
                        tier_for_size: dict[int, int] | None = None
                        ) -> CoupledGraph:
    """Load every rank's ET, cost each node, and wire the cross-rank edges.

    Durations come from the same cost model as dag_sim (roofline COMP, collective
    decomposition timed through time_chunk, p2p SEND on the inter-node tier).
    RECV nodes are zero-cost sync points: the matched SEND carries the transfer
    (and its latency), so the recv just gates downstream compute on the send --
    this is what produces the PP bubble.
    """
    nodes: dict[tuple[int, int], CNode] = {}
    coll_instances: dict[tuple, list[tuple[int, int]]] = {}
    sends: dict[tuple[int, int, int], tuple[int, int]] = {}      # (src,dst,tag)->key
    recvs: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    pp_tier = net.n_tiers() - 1

    for rank in range(num_ranks):
        ets = load_et(Path(et_dir) / f"{workload}.{rank}.et")
        present = {n.id for n in ets}
        for n in ets:
            dur, res, coll = 0.0, None, None
            if n.type == COMP_NODE:
                dur = roofline_duration_ns(n.num_ops, n.tensor_size,
                                           net.peak_perf_tflops, net.local_mem_bw_GBs)
                res = ("compute", rank)
            elif n.type == COMM_COLL_NODE:
                members = groups.get(n.pg_name or "", [])
                gsize = len(members)
                tier = tier_for_collective(gsize, net, tier_for_size)
                kind = _COMM_KIND.get(n.comm_type)
                res = ("net", rank, tier)
                if kind and gsize > 1 and n.comm_size:
                    for c in _collective_chunks(kind, n.comm_size, gsize, tier,
                                                collective_impl):
                        dur += time_chunk(c, net, rng).time_ns
                    coll = (n.pg_name, n.name)
            elif n.type == COMM_SEND_NODE:
                res = ("net", rank, pp_tier)
                if n.comm_size:
                    dur = time_chunk(Chunk(pp_tier, n.comm_size), net, rng).time_ns
            elif n.type == COMM_RECV_NODE:
                res = None          # zero-cost sync; gated on matched send
            cn = CNode(rank=rank, nid=n.id, type=n.type, name=n.name,
                       dur=dur, res=res,
                       deps=[(rank, d) for d in (list(n.data_deps) + list(n.ctrl_deps))
                             if d in present],
                       coll=coll)
            nodes[(rank, n.id)] = cn
            if coll is not None:
                coll_instances.setdefault(coll, []).append((rank, n.id))
            if n.type == COMM_SEND_NODE and n.comm_dst is not None:
                sends[(rank, n.comm_dst, n.comm_tag or 0)] = (rank, n.id)
            elif n.type == COMM_RECV_NODE and n.comm_src is not None:
                recvs.append(((n.comm_src, rank, n.comm_tag or 0), (rank, n.id)))

    for key, recv_key in recvs:
        snd = sends.get(key)
        if snd is not None:
            nodes[recv_key].sr_pred = snd
    return CoupledGraph(nodes=nodes, coll_instances=coll_instances)


def simulate_coupled(graph: CoupledGraph, num_ranks: int) -> dict[int, float]:
    """Global list-schedule the coupled DAG -> per-rank step time (ns).

    Resources: one serial compute stream per rank + one LinkScheduler per
    (rank, tier). Units are single nodes, except a collective instance is one
    barrier unit: it becomes ready when its slowest member's deps are met, then
    occupies every member's link for the collective duration and finishes for
    all members together.

    Validation gates:
      A. infinite bandwidth -> every comm dur -> ~0, step collapses to the
         compute+latency ideal floor (== AstraSim ideal floor).
      B. ring @ 50 GB/s -> reproduces Exp4 coupled OCS (4783/3203 ms); direct
         comes in <= ring (the OCS algorithmic win).
    """
    nodes = graph.nodes
    member_of = {m: key for key, ms in graph.coll_instances.items() for m in ms}

    # Successors + pending-pred counts over intra-rank deps and recv->send edges.
    succs: dict[tuple[int, int], list[tuple[int, int]]] = {k: [] for k in nodes}
    pending: dict[tuple[int, int], int] = {}
    for k, cn in nodes.items():
        preds = list(cn.deps)
        if cn.sr_pred is not None:
            preds.append(cn.sr_pred)
        pending[k] = len(preds)
        for p in preds:
            succs[p].append(k)

    ready_time: dict[tuple[int, int], float] = {}
    # Per-instance count of members that have become individually ready.
    coll_ready: dict[tuple, int] = {key: 0 for key in graph.coll_instances}

    compute_free: dict[int, float] = {}
    link: dict[tuple[int, int], LinkScheduler] = {}

    def _link(rank: int, tier: int) -> LinkScheduler:
        # One serial link per (rank, tier). Bandwidth contention is already in
        # each node's duration; the scheduler just serialises the link timeline.
        if (rank, tier) not in link:
            link[(rank, tier)] = LinkScheduler(n_parallel=1)
        return link[(rank, tier)]

    finish: dict[tuple[int, int], float] = {}
    # PQ entries: (ready_time, tiebreak, kind, payload)
    #   kind "node" -> payload=node key ; kind "coll" -> payload=instance key
    pq: list[tuple[float, int, str, object]] = []
    seq = 0

    def _enqueue_node(k):
        nonlocal seq
        cn = nodes[k]
        if cn.coll is not None:
            key = cn.coll
            coll_ready[key] += 1
            if coll_ready[key] == len(graph.coll_instances[key]):
                rt = max(ready_time[m] for m in graph.coll_instances[key])
                heapq.heappush(pq, (rt, seq, "coll", key)); seq += 1
        else:
            heapq.heappush(pq, (ready_time[k], seq, "node", k)); seq += 1

    for k in nodes:
        if pending[k] == 0:
            ready_time[k] = 0.0
            _enqueue_node(k)

    def _run_on(cn: CNode, ready: float, dur: float) -> float:
        """Schedule one node's resource occupancy, return its finish time."""
        if cn.res is None:
            return ready
        kind = cn.res[0]
        if kind == "compute":
            r = cn.res[1]
            start = max(ready, compute_free.get(r, 0.0))
            end = start + dur
            compute_free[r] = end
            return end
        # net
        _r, tier = cn.res[1], cn.res[2]
        _start, end, _ch = _link(cn.res[1], tier).schedule(ready, dur)
        return end

    while pq:
        rt, _s, kind, payload = heapq.heappop(pq)
        if kind == "node":
            k = payload
            end = _run_on(nodes[k], rt, nodes[k].dur)
            finish[k] = end
            done = [k]
        else:  # collective barrier
            members = graph.coll_instances[payload]
            dur = max(nodes[m].dur for m in members)
            # all members start together at rt, each on its own link
            ends = [_run_on(nodes[m], rt, dur) for m in members]
            end = max(ends)
            for m in members:
                finish[m] = end
            done = members
        for k in done:
            for sk in succs[k]:
                pending[sk] -= 1
                if pending[sk] == 0:
                    scn = nodes[sk]
                    preds = list(scn.deps) + ([scn.sr_pred] if scn.sr_pred else [])
                    ready_time[sk] = max((finish[p] for p in preds), default=0.0)
                    _enqueue_node(sk)

    if len(finish) != len(nodes):
        raise ValueError(
            f"coupled schedule incomplete: {len(finish)}/{len(nodes)} nodes "
            f"(cycle or unmatched barrier?)")

    steps: dict[int, float] = {r: 0.0 for r in range(num_ranks)}
    for (rank, _nid), end in finish.items():
        if rank < num_ranks and end > steps[rank]:
            steps[rank] = end
    return steps


def fc_network(num_ranks: int, bandwidth_GBs: float, latency_ns: float,
               peak_perf_tflops: float, local_mem_bw_GBs: float) -> NetworkConfig:
    """Flat FullyConnected (OCS) net with roofline params surfaced for COMP."""
    return NetworkConfig(
        topology=["FullyConnected"],
        npus_count=[num_ranks],
        bandwidth_GBs=[bandwidth_GBs],
        latency_ns=[latency_ns],
        impl_per_tier=["direct"],
        tdm={},
        peak_perf_tflops=peak_perf_tflops,
        local_mem_bw_GBs=local_mem_bw_GBs,
        roofline_enabled=True,
    )
