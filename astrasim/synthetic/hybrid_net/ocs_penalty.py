"""Derive an OCS-tier contention penalty from AstraSim's coupled trace.

Feeds the lightweight trace-replay (Phase 2). Given the per-node coupled timeline
(trace_loader) we isolate the OCS-tier flows -- the DP collectives (size-4 groups;
TP size-8 stays on NVLink) plus PP point-to-point transmission -- on the GLOBAL
clock across all ranks, and measure how many circuits the fabric must carry at
once. Contention is a *cross-rank* phenomenon (many ranks' flows hitting the
optical switch simultaneously), so per-rank concurrency understates it.

Key outputs:
  * concurrency_profile: peak vs time-average simultaneous OCS flows. A large
    peak/avg gap means BURSTY traffic -- a single average-k derating (Probe A)
    misses the bursts where blocking actually happens.
  * fast-rotor derating B_eff/L_eff (ocs-derating-model) as a function of the
    fabric circuit capacity C: during a burst of k>C flows the fabric time-shares
    -> effective slowdown ~ k/C; for k<=C no contention. This is the knob the
    Go/No-Go gate turns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .et_loader import (
    COMM_COLL_NODE,
    COMM_SEND_NODE,
    load_et,
)
from .trace_loader import parse_trace

# A DP group has this many ranks (TP=8 stays intra-node/NVLink; DP=4 is the
# inter-node/OCS collective). Configurable for other parallelism layouts.
DP_GROUP_SIZE_DEFAULT = 4


@dataclass
class OcsFlow:
    kind:  str   # "DP" | "SEND"
    start: int
    end:   int
    bytes: int


def extract_ocs_flows(trace_dir: Path, et_dir: Path, workload: str,
                      num_ranks: int, groups: dict[str, list[int]],
                      dp_group_size: int = DP_GROUP_SIZE_DEFAULT) -> list[OcsFlow]:
    """All OCS-tier transmission flows across every rank, on the global clock.

    DP collectives (size==dp_group_size) + PP SEND transmission. RECV is excluded
    -- its span is mostly the pipeline bubble (waiting on the peer stage), not
    fabric load (see trace_loader: RECV span >> its transmission).
    """
    trace = parse_trace(trace_dir)
    flows: list[OcsFlow] = []
    for rank in range(num_ranks):
        et = {n.id: n for n in load_et(Path(et_dir) / f"{workload}.{rank}.et")}
        for nid, tn in trace.get(rank, {}).items():
            if tn.end is None:
                continue
            e = et.get(nid)
            if e is None:
                continue
            if tn.type == COMM_COLL_NODE and \
                    len(groups.get(e.pg_name or "", [])) == dp_group_size:
                flows.append(OcsFlow("DP", tn.start, tn.end, e.comm_size or 0))
            elif tn.type == COMM_SEND_NODE:
                flows.append(OcsFlow("SEND", tn.start, tn.end, e.comm_size or 0))
    return flows


@dataclass
class ConcurrencyProfile:
    peak:        int
    time_avg:    float
    span_ns:     int
    busy_ns:     int      # sum of flow durations
    n_flows:     int
    # fraction of busy-time spent at each concurrency level k -> ns
    level_ns:    dict[int, int]


def concurrency_profile(flows: list[OcsFlow]) -> ConcurrencyProfile:
    """Sweep-line peak + time-average simultaneous flows, and the dwell time at
    each concurrency level (for capacity-aware derating)."""
    evs: list[tuple[int, int]] = []
    for f in flows:
        e = f.end if f.end > f.start else f.start + 1
        evs.append((f.start, 1))
        evs.append((e, -1))
    if not evs:
        return ConcurrencyProfile(0, 0.0, 0, 0, 0, {})
    evs.sort()
    cur = peak = 0
    weighted = 0
    level_ns: dict[int, int] = {}
    prev = evs[0][0]
    for t, d in evs:
        if t > prev:
            level_ns[cur] = level_ns.get(cur, 0) + (t - prev)
            weighted += cur * (t - prev)
        cur += d
        peak = max(peak, cur)
        prev = t
    span = evs[-1][0] - evs[0][0]
    busy = sum((f.end if f.end > f.start else f.start + 1) - f.start for f in flows)
    return ConcurrencyProfile(
        peak=peak, time_avg=(weighted / span if span else 0.0),
        span_ns=span, busy_ns=busy, n_flows=len(flows), level_ns=level_ns)


def burst_blocking_time_ns(prof: ConcurrencyProfile, capacity: int) -> int:
    """Extra fabric-busy time from bursts exceeding circuit capacity C.

    At concurrency k>C the fabric serves C at a time, so the k flows take ~k/C
    longer during that window. Sums the excess over all dwell windows. This is the
    contention the average-k derating misses; 0 if C >= peak.
    """
    extra = 0
    for k, ns in prof.level_ns.items():
        if k > capacity > 0:
            extra += int(ns * (k - capacity) / capacity)
    return extra
