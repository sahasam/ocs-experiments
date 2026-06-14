#!/usr/bin/env python3
"""Bucket an ns-3 fct.txt by flow size and report per-type congestion slowdown.

ns-3 (HPCC) fct.txt columns (whitespace-separated, all times in ns):
    sip dip sport dport size start_time fct ideal_fct

`fct` is the actual flow completion time under congestion; `ideal_fct` is the
standalone line-rate time (what an uncongested / OCS fabric would see). The ratio
fct/ideal_fct is the per-flow congestion penalty -- so OCS-vs-PS per-flow slowdown
is readable from a single ns-3 run (see Exp 4). Flows bucket cleanly by size:
268 MB = PP cross-stage sends, ~5.8/16.4 MB = DP ring all-reduce chunks.

Usage:  python analyze_fct.py results/<workload>/ns3_output_clos/fct.txt
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path


def human(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "fct.txt")
    by_size: dict[int, list[tuple[int, int]]] = defaultdict(list)  # size -> [(fct, ideal)]
    n = 0
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 8:
            continue
        size, fct, ideal = int(f[4]), int(f[6]), int(f[7])
        by_size[size].append((fct, ideal))
        n += 1

    print(f"{path}  ({n} flows)\n")
    hdr = (f"{'flow size':>10} {'count':>7} {'ideal':>11} {'actual_mean':>12} "
           f"{'actual_max':>11} {'slow_mean':>9} {'slow_max':>9}")
    print(hdr)
    print("-" * len(hdr))
    for size in sorted(by_size, reverse=True):
        rows = by_size[size]
        ideals = [i for _, i in rows]
        fcts = [c for c, _ in rows]
        ideal = sum(ideals) / len(ideals)
        amean = sum(fcts) / len(fcts)
        amax = max(fcts)
        smean = amean / ideal if ideal else float("nan")
        smax = amax / ideal if ideal else float("nan")
        print(f"{human(size):>10} {len(rows):>7} {ideal/1e3:>9.0f}us "
              f"{amean/1e3:>10.0f}us {amax/1e3:>9.0f}us "
              f"{smean:>8.2f}x {smax:>8.2f}x")


if __name__ == "__main__":
    main()
