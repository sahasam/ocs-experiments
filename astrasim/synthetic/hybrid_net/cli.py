"""CLI for the hybrid packet/circuit Python network model.

Reads an AstraSim-style network YAML (topology / npus_count / bandwidth /
latency) and system JSON (all-reduce-implementation), applies a named TDM
preset on the selected tier, runs Monte-Carlo step simulation per rank,
and writes a CSV + Markdown summary in the same shape as parse_results.py
output -- plus TDM-specific columns (P99, slot util, bytes-in-circuit %,
guard-band waste).

Run with --sanity to execute the six self-tests from the plan instead of
a normal simulation.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import yaml

# Allow running both as `python -m hybrid_net.cli` (package) and as a script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from hybrid_net.presets import PRESETS, apply_overrides, parse_overrides
    from hybrid_net.simulate import (
        NetworkConfig,
        RankResult,
        aggregate_chunk_stats,
        simulate_all_ranks,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gpu_constants import DOLLARS_PER_GPU_HOUR, WATTS_PER_GPU
else:
    from .presets import PRESETS, apply_overrides, parse_overrides
    from .simulate import (
        NetworkConfig,
        RankResult,
        aggregate_chunk_stats,
        simulate_all_ranks,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gpu_constants import DOLLARS_PER_GPU_HOUR, WATTS_PER_GPU


def load_network(yml_path: Path, system_json_path: Path) -> NetworkConfig:
    """Build a NetworkConfig from AstraSim's own input files.

    YAML schema (per astra-sim/inputs/network/*.yml):
        topology: [Switch, Ring]
        npus_count: [8, 2]
        bandwidth: [400.0, 50.0]
        latency: [1000.0, 1000.0]

    JSON schema (per system_*.json):
        "all-reduce-implementation": ["ring", "ring"]
    """
    net_raw = yaml.safe_load(yml_path.read_text())
    sys_raw = json.loads(system_json_path.read_text())
    impl = sys_raw["all-reduce-implementation"]
    if isinstance(impl, str):
        impl = [impl]
    return NetworkConfig(
        topology=net_raw["topology"],
        npus_count=net_raw["npus_count"],
        bandwidth_GBs=[float(b) for b in net_raw["bandwidth"]],
        latency_ns=[float(l) for l in net_raw["latency"]],
        impl_per_tier=impl,
        tdm={},
    )


def derive_row(rr: RankResult, ideal_step_ns: float, num_npus: int) -> dict:
    """Per-rank metrics matching parse_results.py columns + TDM extras."""
    mean_step = statistics.mean(t.step_ns for t in rr.trials)
    mean_comm = statistics.mean(t.total_comm_ns for t in rr.trials)
    mean_exposed = statistics.mean(t.exposed_comm_ns for t in rr.trials)
    mean_hidden = statistics.mean(t.hidden_comm_ns for t in rr.trials)
    mean_compute = statistics.mean(t.compute_ns for t in rr.trials)

    step_p99 = rr.step_ns(0.99)
    exposed_p99 = rr.exposed_ns(0.99)

    step_s = mean_step / 1e9
    gpu_seconds_per_step = step_s * num_npus

    return {
        "wall_real_ns":         mean_step,
        "wall_ideal_ns":        ideal_step_ns,
        "gpu_ns":               mean_compute,
        "comm_total_ns":        mean_comm,
        "comm_exposed_ns":      mean_exposed,
        "comm_hidden_ns":       mean_hidden,
        "overlap_frac":         (mean_hidden / mean_comm) if mean_comm else 0.0,
        "comm_overhead_pct":    100.0 * (mean_step - ideal_step_ns) / ideal_step_ns,
        "gpu_util_real":        mean_compute / mean_step,
        "gpu_util_ideal":       mean_compute / ideal_step_ns,
        "gpu_seconds_per_step": gpu_seconds_per_step,
        "dollars_per_step":     gpu_seconds_per_step * DOLLARS_PER_GPU_HOUR / 3600.0,
        "joules_per_step":      gpu_seconds_per_step * WATTS_PER_GPU,
        "step_p99_ns":          step_p99,
        "comm_exposed_p99_ns":  exposed_p99,
    }


def cross_rank_stats(rows: list[dict], key: str) -> tuple[float, float, float]:
    vals = [r[key] for r in rows]
    return min(vals), statistics.mean(vals), max(vals)


def write_csv(rows: list[dict], out: Path) -> None:
    fields = ["rank"] + sorted(set().union(*(r.keys() - {"rank"} for r in rows)))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_md(rows: list[dict], chunk_agg: dict, out: Path, title: str,
             num_npus: int, preset_name: str, trials: int) -> None:
    ms = lambda x: f"{x / 1e6:.3f} ms"
    pct = lambda x: f"{x:.2f} %"
    frac = lambda x: f"{x:.4f}"
    dol = lambda x: f"${x:.6f}"
    joule = lambda x: f"{x:.3f} J"
    seconds = lambda x: f"{x:.4f} s"

    def line(label: str, key: str, fmt) -> str:
        lo, mean, hi = cross_rank_stats(rows, key)
        return f"| {label} | {fmt(lo)} | {fmt(mean)} | {fmt(hi)} |"

    mean_step = statistics.mean(r["wall_real_ns"] for r in rows)
    mean_compute = statistics.mean(r["gpu_ns"] for r in rows)
    mean_comm = statistics.mean(r["comm_total_ns"] for r in rows)
    mean_exposed = statistics.mean(r["comm_exposed_ns"] for r in rows)
    mean_overhead = statistics.mean(r["comm_overhead_pct"] for r in rows)
    mean_p99 = statistics.mean(r["step_p99_ns"] for r in rows)

    md = []
    md.append(f"# {title}")
    md.append("")
    md.append(f"**Headline (mean across {num_npus} ranks, {trials} trials):** "
              f"step time **{mean_step/1e6:.2f} ms** "
              f"(P99 **{mean_p99/1e6:.2f} ms**), "
              f"compute **{mean_compute/1e6:.2f} ms**, "
              f"total comm **{mean_comm/1e6:.2f} ms** "
              f"(**{mean_exposed/1e6:.2f} ms** exposed). "
              f"Network slows training by **{mean_overhead:.2f}%** "
              f"vs compute-only ideal.")
    md.append("")
    md.append(f"Preset: `{preset_name}`")
    md.append("")
    md.append("## Per-rank panel (cross-rank min / mean / max)")
    md.append("")
    md.append("| metric | min | mean | max |")
    md.append("|---|---|---|---|")
    md.append(line("step time (real)",      "wall_real_ns",         ms))
    md.append(line("step time (ideal net)", "wall_ideal_ns",        ms))
    md.append(line("compute time",          "gpu_ns",               ms))
    md.append(line("total comm",            "comm_total_ns",        ms))
    md.append(line("exposed comm",          "comm_exposed_ns",      ms))
    md.append(line("hidden comm",           "comm_hidden_ns",       ms))
    md.append(line("comm-overlap fraction", "overlap_frac",         frac))
    md.append(line("comm overhead %",       "comm_overhead_pct",    pct))
    md.append(line("GPU util (real net)",   "gpu_util_real",        frac))
    md.append(line("GPU util (ideal net)",  "gpu_util_ideal",       frac))
    md.append(line("GPU-seconds/step",      "gpu_seconds_per_step", seconds))
    md.append(line("$/step",                "dollars_per_step",     dol))
    md.append(line("J/step",                "joules_per_step",      joule))
    md.append(line("**P99 step time**",     "step_p99_ns",          ms))
    md.append(line("**P99 exposed comm**",  "comm_exposed_p99_ns",  ms))
    md.append("")
    md.append("## TDM-specific aggregate (across all chunks / ranks / trials)")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    eff = chunk_agg["effective_circuit_GBs"]
    md.append(f"| % bytes in circuit mode | {chunk_agg['bytes_circuit_pct']:.2f} % |")
    if eff != eff:  # NaN
        md.append(f"| effective circuit bandwidth (GB/s) | n/a (no circuit chunks) |")
    else:
        md.append(f"| effective circuit bandwidth (GB/s) | {eff:.2f} |")
    md.append(f"| guard-band waste fraction | {chunk_agg['guard_waste_frac']:.4f} |")
    out.write_text("\n".join(md) + "\n")


def run_simulation(args) -> None:
    net = load_network(Path(args.network_yml), Path(args.system_json))
    if args.preset:
        base = PRESETS[args.preset]
        overrides = parse_overrides(args.tdm_overrides)
        tdm = apply_overrides(base, overrides)
        net.tdm[args.tier] = tdm
        preset_name = args.preset
        if overrides:
            preset_name += f" (+ overrides: {args.tdm_overrides})"
    else:
        preset_name = "no_tdm (purely analytical)"

    results = simulate_all_ranks(Path(args.et_dir), args.num_npus, net,
                                 trials=args.trials, base_seed=args.seed)

    # Derive ideal-step (compute-only) per rank.
    from .simulate import compute_ideal_step_ns, parse_et_to_layers
    from .et_loader import load_et
    et_files = sorted(Path(args.et_dir).glob("*.et"))
    rows = []
    for rr in results:
        # Find this rank's ET to get its ideal step.
        et = next(p for p in et_files
                  if p.name.endswith(f".{rr.rank}.et"))
        layers = parse_et_to_layers(load_et(et))
        ideal = compute_ideal_step_ns(layers)
        rows.append({"rank": rr.rank, **derive_row(rr, ideal, args.num_npus)})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_prefix}.csv"
    md_path = out_dir / f"{args.out_prefix}.md"
    chunk_agg = aggregate_chunk_stats(results)
    write_csv(rows, csv_path)
    write_md(rows, chunk_agg, md_path, args.title,
             args.num_npus, preset_name, args.trials)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")

    # Echo the headline so command output isn't silent.
    mean_step = statistics.mean(r["wall_real_ns"] for r in rows)
    mean_p99 = statistics.mean(r["step_p99_ns"] for r in rows)
    mean_overhead = statistics.mean(r["comm_overhead_pct"] for r in rows)
    print(f"  step mean={mean_step/1e6:.2f}ms  p99={mean_p99/1e6:.2f}ms  "
          f"overhead={mean_overhead:.2f}%")


def main() -> int:
    p = argparse.ArgumentParser(description="hybrid packet/circuit step sim")
    p.add_argument("--et-dir", help="dir of .et files")
    p.add_argument("--num-npus", type=int, help="number of ranks")
    p.add_argument("--network-yml", help="AstraSim network YAML (topology, npus_count, bw, lat)")
    p.add_argument("--system-json", help="AstraSim system JSON (all-reduce-implementation)")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="TDM preset to apply (omit for analytical-only)")
    p.add_argument("--tier", type=int, default=1,
                   help="tier index to apply TDM to (default 1 = outermost in 2-tier)")
    p.add_argument("--tdm-overrides", default="",
                   help="comma-separated key=val[ns|us|ms|s] overrides for the preset")
    p.add_argument("--trials", type=int, default=1,
                   help="Monte Carlo trials per rank (needed for P99)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--out-prefix", default="summary_hybrid")
    p.add_argument("--title", default="Hybrid-net step-time panel")
    p.add_argument("--sanity", action="store_true",
                   help="run the six self-tests and exit")
    args = p.parse_args()

    if args.sanity:
        from . import test_sanity
        return test_sanity.run_all()

    missing = [k for k in ("et_dir", "num_npus", "network_yml", "system_json")
               if getattr(args, k) is None]
    if missing:
        p.error(f"missing required: {missing}")
    run_simulation(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
