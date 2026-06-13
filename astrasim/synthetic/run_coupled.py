#!/usr/bin/env python3
"""Coupled forward DAG scheduler driver -- the artifact-free OCS number.

Runs hybrid_net.coupled_sim over a workload's ETs and reports per-PP-stage step
time for three configurations on the same engine:

  ideal   : infinite bandwidth, latency-only        -> the compute+latency floor
  direct  : direct (bandwidth-optimal) collectives   -> the headline OCS algorithm
  ring    : ring collectives                          -> the PS algorithm (Exp4 gate)

Self-validation gates (see coupled_sim.simulate_coupled):
  A. ideal step ~= AstraSim ideal floor
  B. ring @ 50 GB/s ~= Exp4 coupled OCS (stage0 4783 ms, stage1 3203 ms);
     direct <= ring.

Usage:
    python run_coupled.py llama3_8b_tp1_pp2_dp8 16
    python run_coupled.py <workload> <num_ranks> --pp-stages 0:0-7,1:8-15 \\
        --bandwidth 50 --latency 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_net import coupled_sim
from hybrid_net.dag_sim import load_comm_groups


def _parse_stages(spec: str, num_ranks: int) -> dict[str, list[int]]:
    """'0:0-7,1:8-15' -> {'stage 0':[0..7], 'stage 1':[8..15]}."""
    if not spec:
        return {"all ranks": list(range(num_ranks))}
    out: dict[str, list[int]] = {}
    for grp in spec.split(","):
        label, rng = grp.split(":")
        lo, hi = rng.split("-")
        out[f"stage {label}"] = list(range(int(lo), int(hi) + 1))
    return out


def _run(ed: Path, workload: str, num_ranks: int, groups, *, impl: str,
         bw: float, lat: float, peak: float, membw: float) -> dict[int, float]:
    net = coupled_sim.fc_network(num_ranks, bw, lat, peak, membw)
    rng = np.random.default_rng(1)
    g = coupled_sim.build_coupled_graph(ed, workload, num_ranks, groups, net,
                                        rng, collective_impl=impl)
    return coupled_sim.simulate_coupled(g, num_ranks)


def main() -> None:
    p = argparse.ArgumentParser(description="coupled forward DAG scheduler")
    p.add_argument("workload")
    p.add_argument("num_ranks", type=int)
    p.add_argument("--results-dir",
                   default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--pp-stages", default="0:0-7,1:8-15",
                   help="label:lo-hi,... groups to report (default PP2 8+8)")
    p.add_argument("--bandwidth", type=float, default=50.0, help="GB/s per link")
    p.add_argument("--latency", type=float, default=500.0, help="ns per hop")
    p.add_argument("--ideal-bandwidth", type=float, default=1e9)
    p.add_argument("--ideal-latency", type=float, default=500.0)
    p.add_argument("--system-json", default=None,
                   help="for peak-perf / local-mem-bw (default stage_configs/system.json)")
    args = p.parse_args()

    ed = Path(args.results_dir) / args.workload
    groups = load_comm_groups(ed / "comm_group.json")
    sysf = Path(args.system_json) if args.system_json else \
        Path(__file__).resolve().parent / "stage_configs" / "system.json"
    sysj = json.loads(sysf.read_text())
    peak = float(sysj.get("peak-perf", 300))
    membw = float(sysj.get("local-mem-bw", 900))

    stages = _parse_stages(args.pp_stages, args.num_ranks)

    configs = [
        ("ideal  (inf bw)", "direct", args.ideal_bandwidth, args.ideal_latency),
        ("direct (OCS)",    "direct", args.bandwidth,       args.latency),
        ("ring   (PS algo)", "ring",  args.bandwidth,       args.latency),
    ]

    print(f"workload={args.workload}  ranks={args.num_ranks}  "
          f"peak={peak} TFLOP/s  membw={membw} GB/s")
    print(f"OCS fabric: bw={args.bandwidth} GB/s  lat={args.latency} ns\n")
    header = f"{'config':<18}" + "".join(f"{lbl:>14}" for lbl in stages)
    print(header)
    print("-" * len(header))
    results: dict[str, dict[str, float]] = {}
    for label, impl, bw, lat in configs:
        steps = _run(ed, args.workload, args.num_ranks, groups,
                     impl=impl, bw=bw, lat=lat, peak=peak, membw=membw)
        row = {lbl: max(steps[r] for r in ranks) for lbl, ranks in stages.items()}
        results[label] = row
        print(f"{label:<18}" + "".join(f"{row[l]/1e6:>11.1f} ms" for l in stages))

    print()
    # Gate B echo: direct must be <= ring on every stage.
    d, r = results["direct (OCS)"], results["ring   (PS algo)"]
    for lbl in stages:
        adv = 100.0 * (r[lbl] - d[lbl]) / r[lbl] if r[lbl] else 0.0
        print(f"  {lbl}: direct is {adv:+.1f}% vs ring  "
              f"({'OK direct<=ring' if d[lbl] <= r[lbl] + 1 else 'FAIL direct>ring'})")


if __name__ == "__main__":
    main()
