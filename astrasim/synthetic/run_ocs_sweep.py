#!/usr/bin/env python3
"""OCS circuit-capacity C-sweep against an AstraSim trace.

Usage:
    python run_ocs_sweep.py <workload> <num_ranks>
    python run_ocs_sweep.py llama3_8b_tp8_pp2_dp4 64
    python run_ocs_sweep.py llama3_8b_tp8_pp2_dp4 64 --capacities inf,32,16,8,4,2,1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_net import dag_sim, ocs_replay


def main() -> None:
    p = argparse.ArgumentParser(description="OCS C-sweep")
    p.add_argument("workload", help="workload name (subdir under results/)")
    p.add_argument("num_ranks", type=int, help="total ranks = DP*TP*PP")
    p.add_argument("--results-dir",
                   default=str(Path(__file__).resolve().parent / "results"),
                   help="parent of the workload dir (default: <script-dir>/results/)")
    p.add_argument("--capacities", default="inf,32,16,8,4,2,1",
                   help="comma-separated circuit capacities; 'inf' = unconstrained baseline")
    args = p.parse_args()

    ed = Path(args.results_dir) / args.workload
    g = dag_sim.load_comm_groups(ed / "comm_group.json")
    nodes, _ets, coll, sr = ocs_replay.build_graph(
        ed / "logs" / "trace", ed, args.workload, args.num_ranks, g)

    caps: list[int] = []
    for tok in args.capacities.split(","):
        caps.append(10**9 if tok.strip().lower() == "inf" else int(tok.strip()))

    print(f"{'C':>14}  {'wall_ms':>10}")
    print("-" * 28)
    for C in caps:
        s = ocs_replay.replay(nodes, coll, sr, C, args.num_ranks)
        wall = max(s.values()) / 1e6
        label = "inf (baseline)" if C >= 10**9 else str(C)
        print(f"C={label:>13}  {wall:>9.1f}ms")


if __name__ == "__main__":
    main()
