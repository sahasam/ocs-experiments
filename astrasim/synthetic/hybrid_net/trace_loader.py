"""Parse AstraSim's trace-enabled debug log into a per-node coupled timeline.

With `trace-enabled:1` in the system config, AstraSim's Workload emits, at debug
level to <logging-folder>/log.log (rotating), one line per node when it is issued
and one when it finishes:

    [ts] [workload] [debug] issue,sys->id=R, tick=T0, node->id=N, node->name=..., node->type=K
    [ts] [workload] [debug] callback,sys->id=R, tick=T1, node->id=N, node->name=..., node->type=K

The (issue, callback) tick pair gives each node's [start, end] under AstraSim's
*coupled* schedule -- including the PP stalls a per-rank model can't see. Joined
with the ET (comm_size/comm_type/pg_name/peer), this is the message timeline the
OCS replay consumes, and a node-level oracle to validate against AstraSim's own
per-rank Wall/Comm/GPU aggregates.

Ticks are AstraSim cycles == ns (Wall 1549076910 == 1.549 s), so durations are ns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .et_loader import (
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    EtNode,
    load_et,
)

_LINE_RE = re.compile(
    r"(issue|callback),sys->id=(\d+), tick=(\d+), node->id=(\d+), "
    r"node->name=(.*), node->type=(\d+)")


@dataclass
class TraceNode:
    rank:    int
    node_id: int
    type:    int
    name:    str
    start:   int          # issue tick (ns)
    end:     int | None    # callback tick (ns); None if never finished
    # joined from the ET (comm nodes only):
    comm_size: int | None = None
    comm_type: int | None = None
    pg_name:   str | None = None
    comm_peer: int | None = None   # comm_dst (SEND) or comm_src (RECV)

    @property
    def dur(self) -> int:
        return (self.end - self.start) if self.end is not None else 0


def _trace_files(trace_dir: Path) -> list[Path]:
    # spdlog rotation: log.log is newest, log.1.log older, etc. Order doesn't
    # matter -- we key by (rank, node_id) -- but read them all.
    fs = sorted(trace_dir.glob("log*.log"))
    if not fs:
        raise FileNotFoundError(f"no log*.log in {trace_dir}")
    return fs


def parse_trace(trace_dir: Path) -> dict[int, dict[int, TraceNode]]:
    """Parse the debug log into {rank: {node_id: TraceNode}} (no ET join yet)."""
    by_rank: dict[int, dict[int, TraceNode]] = {}
    for f in _trace_files(trace_dir):
        with open(f, "r", errors="replace") as fh:
            for line in fh:
                m = _LINE_RE.search(line)
                if not m:
                    continue
                kind, r, tick, nid, name, typ = m.groups()
                r = int(r); tick = int(tick); nid = int(nid); typ = int(typ)
                nodes = by_rank.setdefault(r, {})
                tn = nodes.get(nid)
                if tn is None:
                    tn = TraceNode(rank=r, node_id=nid, type=typ, name=name,
                                   start=tick, end=None)
                    nodes[nid] = tn
                if kind == "issue":
                    tn.start = tick
                else:  # callback
                    tn.end = tick
    return by_rank


def load_rank_timeline(trace_dir: Path, et_dir: Path, workload: str, rank: int
                       ) -> list[TraceNode]:
    """Parse one rank and join comm nodes with the ET for size/type/peer/group."""
    nodes = parse_trace(trace_dir).get(rank, {})
    et = {n.id: n for n in load_et(Path(et_dir) / f"{workload}.{rank}.et")}
    out: list[TraceNode] = []
    for nid, tn in nodes.items():
        e = et.get(nid)
        if e is not None and tn.type in (COMM_COLL_NODE, COMM_SEND_NODE,
                                         COMM_RECV_NODE):
            tn.comm_size = e.comm_size
            tn.comm_type = e.comm_type
            tn.pg_name = e.pg_name
            tn.comm_peer = e.comm_dst if e.comm_dst is not None else e.comm_src
        out.append(tn)
    return out


@dataclass
class RankAggregate:
    rank:     int
    wall_ns:  int      # max end - min start  (matches AstraSim Wall exactly)
    gpu_ns:   int      # sum of COMP durations (matches AstraSim GPU exactly)
    exposed_ns: int    # wall - gpu  (matches AstraSim exposed-comm exactly)
    comm_coll_ns: int  # sum of COMM_COLL durations (network collective load)
    comm_p2p_xmit_ns: int  # SEND durations = real PP transmission (not the wait)
    pp_bubble_ns: int  # RECV span beyond its transmission = pipeline stall
    n_nodes:  int
    n_unfinished: int


def rank_aggregates(trace_dir: Path) -> dict[int, RankAggregate]:
    """Reconstruct per-rank timing from the trace, to validate the oracle.

    Wall = span(start..end), GPU = sum COMP dur, exposed = Wall-GPU all match
    AstraSim's [statistics] lines exactly -- the load-bearing proof the timeline
    is faithful. COMM is split: collectives + p2p *transmission* (SEND dur) are
    the actual network load (what an OCS fabric re-times); the RECV span is mostly
    the PP *bubble* (posted at t0, waiting on the peer stage) -- a coupling stall,
    not network load. AstraSim's single "Comm time" stat blends these with overlap
    accounting, so we don't reproduce it directly (and don't need to).
    """
    out: dict[int, RankAggregate] = {}
    for rank, nodes in parse_trace(trace_dir).items():
        starts = [n.start for n in nodes.values()]
        ends = [n.end for n in nodes.values() if n.end is not None]
        wall = (max(ends) - min(starts)) if ends else 0
        gpu = sum(n.dur for n in nodes.values() if n.type == COMP_NODE)
        coll = sum(n.dur for n in nodes.values() if n.type == COMM_COLL_NODE)
        send = sum(n.dur for n in nodes.values() if n.type == COMM_SEND_NODE)
        # A RECV's transmission ~ the matching SEND's; its span beyond that is bubble.
        recv_span = sum(n.dur for n in nodes.values() if n.type == COMM_RECV_NODE)
        bubble = max(0, recv_span - send)
        out[rank] = RankAggregate(
            rank=rank, wall_ns=wall, gpu_ns=gpu, exposed_ns=wall - gpu,
            comm_coll_ns=coll, comm_p2p_xmit_ns=send, pp_bubble_ns=bubble,
            n_nodes=len(nodes),
            n_unfinished=sum(1 for n in nodes.values() if n.end is None),
        )
    return out
