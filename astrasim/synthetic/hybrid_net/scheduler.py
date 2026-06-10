"""Per-rank link scheduler for circuit reservations.

When FLOW_RESERVED tiers are in play, each tier on each rank carries a
LinkScheduler that tracks per-channel availability. A flow requests a
circuit at `arrival_ns` for `duration_ns`; the earliest-available channel
serves it, possibly after a wait.

This models congestion within a rank's own flow timeline (e.g. AR[N]
extending past the start of AR[N-1]'s arrival). Cross-rank contention is
out of scope -- ranks are simulated independently and the workload's DP
symmetry keeps that defensible.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LinkScheduler:
    """Track when each parallel circuit channel on one (tier, rank) is next free.

    n_parallel: number of independent circuit channels (WDM lanes, distinct
        reservations that can coexist on the tier).
    cursors_ns: per-channel "next free at" timestamp. Initialised to zeros.
    """
    n_parallel: int = 1
    cursors_ns: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_parallel < 1:
            raise ValueError(f"n_parallel must be >= 1, got {self.n_parallel}")
        if len(self.cursors_ns) != self.n_parallel:
            self.cursors_ns = [0.0] * self.n_parallel

    def schedule(self, arrival_ns: float, duration_ns: float
                 ) -> tuple[float, float, int]:
        """Reserve `duration_ns` on the earliest-available channel from
        `arrival_ns`. Returns (start, end, channel_index)."""
        ch = min(range(self.n_parallel), key=lambda i: self.cursors_ns[i])
        start = max(arrival_ns, self.cursors_ns[ch])
        end = start + duration_ns
        self.cursors_ns[ch] = end
        return start, end, ch
