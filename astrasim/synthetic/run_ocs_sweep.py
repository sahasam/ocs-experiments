#!/usr/bin/env python3
"""OCS circuit-capacity C-sweep against an AstraSim trace.

Sweeps two axes:
  C         -- concurrent circuit capacity (contention / bandwidth sharing)
  t_setup   -- per-PP-send circuit establishment wait in ms (latency floor)

Usage:
    python run_ocs_sweep.py <workload> <num_ranks>
    python run_ocs_sweep.py llama3_8b_tp8_pp2_dp4 64
    python run_ocs_sweep.py llama3_8b_tp8_pp2_dp8 128 --dp-group-size 8 \\
        --capacities inf,288,128,64,32 --t-setups 0,1,10,27
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_net import dag_sim, ocs_replay


def main() -> None:
    p = argparse.ArgumentParser(description="OCS C × T_setup sweep")
    p.add_argument("workload", help="workload name (subdir under results/)")
    p.add_argument("num_ranks", type=int, help="total ranks = DP*TP*PP*SP")
    p.add_argument("--results-dir",
                   default=str(Path(__file__).resolve().parent / "results"),
                   help="parent of the workload dir (default: <script-dir>/results/)")
    p.add_argument("--capacities", default="inf,288,128,64,32,16,8,4,2,1",
                   help="comma-separated circuit capacities; 'inf' = unconstrained baseline")
    p.add_argument("--t-setups", default="0",
                   help="comma-separated per-PP-send circuit wait in ms (default: 0)")
    p.add_argument("--dp-group-size", type=int, default=4,
                   help="DP collective group size (= DP degree); used to classify OCS vs NVLink flows (default: 4)")
    args = p.parse_args()

    ed = Path(args.results_dir) / args.workload
    g = dag_sim.load_comm_groups(ed / "comm_group.json")
    nodes, _ets, coll, sr = ocs_replay.build_graph(
        ed / "logs" / "trace", ed, args.workload, args.num_ranks, g,
        dp_group_size=args.dp_group_size)

    caps: list[int] = []
    for tok in args.capacities.split(","):
        caps.append(10**9 if tok.strip().lower() == "inf" else int(tok.strip()))

    t_setups_ms: list[float] = [float(t.strip()) for t in args.t_setups.split(",")]

    baseline_wall = max(ocs_replay.replay(nodes, coll, sr, 10**9, args.num_ranks).values())

    single_t = len(t_setups_ms) == 1

    if single_t:
        # original 1-D table
        t_ns = t_setups_ms[0] * 1e6
        print(f"  t_setup={t_setups_ms[0]:.0f}ms   dp_group_size={args.dp_group_size}")
        print(f"{'C':>14}  {'wall_ms':>10}  {'vs_PS':>8}")
        print("-" * 38)
        for C in caps:
            s = ocs_replay.replay(nodes, coll, sr, C, args.num_ranks, t_setup_ns=t_ns)
            wall = max(s.values()) / 1e6
            pct = (wall / (baseline_wall / 1e6) - 1) * 100
            label = "inf (baseline)" if C >= 10**9 else str(C)
            print(f"C={label:>13}  {wall:>9.1f}ms  {pct:>+7.1f}%")
    else:
        # 2-D grid: rows = t_setup, cols = C
        col_w = 12
        c_labels = ["inf" if C >= 10**9 else str(C) for C in caps]
        header = f"{'t_setup':>10}" + "".join(f"  {'C='+l:>{col_w}}" for l in c_labels)
        print(f"  dp_group_size={args.dp_group_size}   (cells = % vs PS baseline {baseline_wall/1e6:.0f}ms)")
        print(header)
        print("-" * len(header))
        for t_ms in t_setups_ms:
            t_ns = t_ms * 1e6
            row = f"{t_ms:>8.0f}ms"
            for C in caps:
                s = ocs_replay.replay(nodes, coll, sr, C, args.num_ranks, t_setup_ns=t_ns)
                wall = max(s.values()) / 1e6
                pct = (wall / (baseline_wall / 1e6) - 1) * 100
                cell = f"{pct:+.1f}%" if pct >= 0.05 else "  0.0%"
                row += f"  {cell:>{col_w}}"
            print(row)


if __name__ == "__main__":
    main()
