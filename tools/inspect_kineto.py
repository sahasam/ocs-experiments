"""Summarize a Kineto JSON trace.

Usage: python tools/inspect_kineto.py traces/kineto_rank0.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> None:
    with open(path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    by_cat: Counter[str] = Counter(e.get("cat", "?") for e in events)
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"]
    op_counts: Counter[str] = Counter(e.get("name", "?") for e in cpu_ops)

    total_cpu_us = sum(e.get("dur", 0) for e in cpu_ops)

    print(f"=== {path} ===")
    print(f"Total events:        {len(events)}")
    print(f"Events by category:  {dict(by_cat)}")
    print(f"cpu_op events:       {len(cpu_ops)}")
    print(f"CPU time covered:    {total_cpu_us / 1000:.1f} ms")
    print()
    print("Top 15 op names by count:")
    for name, n in op_counts.most_common(15):
        print(f"  {n:6d}  {name}")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    summarize(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
