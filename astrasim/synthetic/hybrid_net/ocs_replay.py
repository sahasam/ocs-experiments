"""Exposed-fraction OCS replay: how much capacity-C circuit contention extends
the step, swept over fabric capacity C.

Method -- delay propagation seeded by AstraSim's coupled trace:

  AstraSim's trace already gives a correct coupled schedule (every node's
  baseline start/end, including PP bubbles). We do NOT re-derive that coupling.
  Instead, under a finite OCS circuit capacity C, the DP-collective + PP flows on
  the OCS tier get *stretched* by their oversubscription (when k>C simultaneous
  flows share C circuits, each runs ~k/C slower). We then propagate ONLY that
  extra delay through the dependency graph and read off how much the step grows.

  Crucially, at C >= peak concurrency the stretch is 1.0, extra delay is 0, and
  the replay reproduces the trace oracle EXACTLY (Wall 1549/1058) by construction
  -- so the C-sweep deltas are trustworthy without re-implementing AstraSim.

Edges propagated:
  * intra-rank: ET data_deps + ctrl_deps
  * collective barrier: a collective instance (same pg_name+name across its group)
    completes together -> its delay = max member arrival delay + own stretch
  * PP send->recv: a recv's completion delay = matched send's completion delay

First-order: the oversubscription k(t) is taken from the BASELINE schedule
(contention can reshuffle it -- iterate for a fixpoint if needed; one pass is the
exposed-fraction estimate). See [[ocs-modeling-decision]], [[ocs-derating-model]].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .et_loader import (
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    load_et,
)
from .ocs_penalty import DP_GROUP_SIZE_DEFAULT
from .trace_loader import parse_trace


@dataclass
class RNode:
    rank:   int
    nid:    int
    type:   int
    name:   str
    start:  int            # baseline start tick
    end:    int            # baseline end tick
    deps:   list[tuple[int, int]] = field(default_factory=list)  # (rank,nid) preds
    is_ocs: bool = False   # DP collective or PP send (stretched under capacity C)
    extra:  float = 0.0    # added duration under current C (filled per sweep point)
    delay:  float = 0.0    # propagated delay (filled per sweep point)


def build_graph(trace_dir: Path, et_dir: Path, workload: str, num_ranks: int,
                groups: dict[str, list[int]],
                dp_group_size: int = DP_GROUP_SIZE_DEFAULT):
    """Build the global node table + cross-rank edges from trace + ETs."""
    trace = parse_trace(trace_dir)
    nodes: dict[tuple[int, int], RNode] = {}
    ets: dict[int, dict[int, object]] = {}
    # collective instances: (pg_name, name) -> [(rank,nid)]
    coll_inst: dict[tuple[str, str], list[tuple[int, int]]] = {}
    # send/recv matching: SEND key (src,dst,tag) and RECV key (src,dst,tag)
    sends: dict[tuple[int, int, int], tuple[int, int]] = {}
    recvs: list[tuple[tuple[int, int, int], tuple[int, int]]] = []

    for rank in range(num_ranks):
        et = {n.id: n for n in load_et(Path(et_dir) / f"{workload}.{rank}.et")}
        ets[rank] = et
        for nid, tn in trace.get(rank, {}).items():
            if tn.end is None:
                continue
            e = et.get(nid)
            is_ocs = False
            if e is not None:
                if tn.type == COMM_COLL_NODE and \
                        len(groups.get(e.pg_name or "", [])) == dp_group_size:
                    is_ocs = True
                elif tn.type == COMM_SEND_NODE:
                    is_ocs = True
            rn = RNode(rank=rank, nid=nid, type=tn.type, name=tn.name,
                       start=tn.start, end=tn.end, is_ocs=is_ocs)
            nodes[(rank, nid)] = rn
            # intra-rank deps (only those present in this rank's trace)
            if e is not None:
                rn.deps = [(rank, d) for d in (list(e.data_deps) + list(e.ctrl_deps))
                           if (rank, d) in nodes or d in trace.get(rank, {})]
                if tn.type == COMM_COLL_NODE and e.pg_name:
                    coll_inst.setdefault((e.pg_name, tn.name), []).append((rank, nid))
                elif tn.type == COMM_SEND_NODE and e.comm_dst is not None:
                    sends[(rank, e.comm_dst, e.comm_tag or 0)] = (rank, nid)
                elif tn.type == COMM_RECV_NODE and e.comm_src is not None:
                    recvs.append(((e.comm_src, rank, e.comm_tag or 0), (rank, nid)))

    # Resolve recv -> matched send edge (src,dst,tag).
    sr_edges: dict[tuple[int, int], tuple[int, int]] = {}  # recv -> send
    for key, recv_key in recvs:
        snd = sends.get(key)
        if snd is not None:
            sr_edges[recv_key] = snd
    return nodes, ets, coll_inst, sr_edges


def _concurrency_breakpoints(nodes: dict[tuple[int, int], RNode]):
    """Step-function of simultaneous OCS flows over the global clock."""
    evs = []
    for rn in nodes.values():
        if rn.is_ocs:
            e = rn.end if rn.end > rn.start else rn.start + 1
            evs.append((rn.start, 1)); evs.append((e, -1))
    evs.sort()
    # piecewise: list of (t0, t1, k)
    pieces = []
    cur = 0; prev = evs[0][0] if evs else 0
    for t, d in evs:
        if t > prev:
            pieces.append((prev, t, cur))
        cur += d; prev = t
    return pieces


def _oversub_factor(rn: RNode, pieces, capacity: int) -> float:
    """Time-avg max(1, k/C) over [start,end] -> the flow's stretch factor."""
    if capacity <= 0 or rn.end <= rn.start:
        return 1.0
    s, e = rn.start, rn.end
    acc = 0.0
    for (t0, t1, k) in pieces:
        lo, hi = max(s, t0), min(e, t1)
        if hi > lo:
            acc += (hi - lo) * max(1.0, k / capacity)
    return acc / (e - s) if e > s else 1.0


def replay(nodes, coll_inst, sr_edges, capacity: int, num_ranks: int,
           t_setup_ns: float = 0.0) -> dict[int, float]:
    """Propagate capacity-C contention + circuit-setup delay; return per-rank step time (ns).

    t_setup_ns: per-PP-send circuit establishment wait (ns).  Applied to every
    COMM_SEND_NODE on the OCS tier before bandwidth stretch.  Models rotor
    half-cycle or MEMS reconfiguration time on the PP critical path.
    0 = fast EO / Sirius-class (negligible); ~1e7 = 10 ms MEMS.
    """
    pieces = _concurrency_breakpoints(nodes)
    # extra duration per node under this C + t_setup
    for rn in nodes.values():
        if rn.is_ocs and capacity < 10**9:
            f = _oversub_factor(rn, pieces, capacity)
            rn.extra = (f - 1.0) * (rn.end - rn.start)
        else:
            rn.extra = 0.0
        # PP sends pay a per-circuit setup wait on top of bandwidth contention
        if t_setup_ns > 0 and rn.type == COMM_SEND_NODE and rn.is_ocs:
            rn.extra += t_setup_ns
        rn.delay = 0.0

    # collective-instance membership lookup: node -> its group members
    member_of: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for inst, members in coll_inst.items():
        for m in members:
            member_of[m] = members

    # Sort by end time (not start): AstraSim sets recv.end == send.end, but
    # recv.start=0 (pre-posted), so a start-time sort puts recvs before their
    # matched sends and breaks sr_edge delay propagation.  Tiebreak: sends
    # before recvs so recv picks up the send's delay when ends are equal.
    order = sorted(nodes.keys(),
                   key=lambda k: (nodes[k].end,
                                  0 if nodes[k].type == COMM_SEND_NODE else 1))
    baseline_end = {k: nodes[k].end for k in nodes}

    def slack(pred, node) -> float:
        return max(0.0, nodes[node].start - baseline_end[pred])

    for key in order:
        rn = nodes[key]
        in_delay = 0.0
        # intra-rank + recv->send predecessors
        preds = list(rn.deps)
        if key in sr_edges:
            preds.append(sr_edges[key])  # recv waits on matched send
        for p in preds:
            if p in nodes:
                in_delay = max(in_delay, nodes[p].delay - slack(p, key))
        # collective barrier: gated by the slowest group member's arrival delay
        if key in member_of:
            for m in member_of[key]:
                if m != key and m in nodes:
                    # member's arrival delay ~ its propagated delay minus own extra
                    in_delay = max(in_delay, nodes[m].delay - nodes[m].extra)
        rn.delay = rn.extra + max(0.0, in_delay)

    steps: dict[int, float] = {}
    for r in range(num_ranks):
        mx = 0.0
        for (rank, nid), rn in nodes.items():
            if rank == r:
                mx = max(mx, baseline_end[(rank, nid)] + rn.delay)
        steps[r] = mx
    return steps
