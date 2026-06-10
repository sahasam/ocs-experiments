"""Parse AstraSim logs (real + ideal network) into a metric panel.

Inputs:  results/logs/astrasim_real.log, astrasim_ideal.log
Outputs: results/summary.csv (per-rank), results/summary.md (cross-rank table)

AstraSim emits per-rank lines like:
  [statistics] [info] sys[0], Wall time: 437931112
  [statistics] [info] sys[0], Comm time: 72355584
  [statistics] [info] sys[0], GPU time: 435670000
  [statistics] [info] sys[0], Total compute-communication overlap: 70094472

Cycle counts are interpreted as nanoseconds (1 cycle = 1 ns at 1 GHz, which is
what text_converter -> AstraSim assumes when comp_time is given in microseconds).
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

from gpu_constants import DOLLARS_PER_GPU_HOUR, WATTS_PER_GPU

NS_PER_CYCLE = 1  # AstraSim assumption when converter inputs are microseconds

PATTERNS = {
    "wall_ns":    re.compile(r"sys\[(\d+)\], Wall time:\s+(\d+)"),
    "comm_ns":    re.compile(r"sys\[(\d+)\], Comm time:\s+(\d+)"),
    "gpu_ns":     re.compile(r"sys\[(\d+)\], GPU time:\s+(\d+)"),
    "overlap_ns": re.compile(r"sys\[(\d+)\], Total compute-communication overlap:\s+(\d+)"),
}


def parse_log(path: Path, num_npus: int) -> dict[int, dict[str, int]]:
    """Returns {rank: {wall_ns, comm_ns, gpu_ns, overlap_ns}}."""
    per_rank: dict[int, dict[str, int]] = {r: {} for r in range(num_npus)}
    text = path.read_text()
    for metric, pat in PATTERNS.items():
        for m in pat.finditer(text):
            rank = int(m.group(1))
            if rank >= num_npus:
                continue
            cycles = int(m.group(2))
            per_rank[rank][metric] = cycles * NS_PER_CYCLE
    # Default overlap to 0 if absent (AstraSim omits the line when overlap == 0).
    for r in per_rank:
        per_rank[r].setdefault("overlap_ns", 0)
    return per_rank


def derive_per_rank(real: dict[str, int], ideal: dict[str, int], num_npus: int) -> dict[str, float]:
    wall_real = real["wall_ns"]
    wall_ideal = ideal["wall_ns"]
    comm_real = real["comm_ns"]
    gpu_real = real["gpu_ns"]
    overlap_real = real["overlap_ns"]
    exposed_real = max(0, comm_real - overlap_real)

    step_s = wall_real / 1e9
    gpu_seconds_per_step = step_s * num_npus

    return {
        "wall_real_ns":         wall_real,
        "wall_ideal_ns":        wall_ideal,
        "gpu_ns":               gpu_real,
        "comm_total_ns":        comm_real,
        "comm_exposed_ns":      exposed_real,
        "comm_hidden_ns":       overlap_real,
        "overlap_frac":         (overlap_real / comm_real) if comm_real else 0.0,
        "comm_overhead_pct":    100.0 * (wall_real - wall_ideal) / wall_ideal,
        "gpu_util_real":        gpu_real / wall_real,
        "gpu_util_ideal":       gpu_real / wall_ideal,
        "gpu_seconds_per_step": gpu_seconds_per_step,
        "dollars_per_step":     gpu_seconds_per_step * DOLLARS_PER_GPU_HOUR / 3600.0,
        "joules_per_step":      gpu_seconds_per_step * WATTS_PER_GPU,
    }


def fmt_ms(ns: float) -> str:
    return f"{ns / 1e6:.3f}"


def write_csv(rows: list[dict], out: Path) -> None:
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank"] + list(rows[0].keys() - {"rank"}))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cross_rank_stats(rows: list[dict], key: str) -> tuple[float, float, float]:
    vals = [r[key] for r in rows]
    return min(vals), statistics.mean(vals), max(vals)


def write_summary_md(rows: list[dict], out: Path, title: str, num_npus: int) -> None:
    def line(label: str, key: str, fmt) -> str:
        lo, mean, hi = cross_rank_stats(rows, key)
        return f"| {label} | {fmt(lo)} | {fmt(mean)} | {fmt(hi)} |"

    ms = lambda x: f"{x / 1e6:.3f} ms"
    pct = lambda x: f"{x:.2f} %"
    frac = lambda x: f"{x:.4f}"
    dol = lambda x: f"${x:.6f}"
    joule = lambda x: f"{x:.3f} J"
    seconds = lambda x: f"{x:.4f} s"

    # Headline numbers (mean across ranks)
    mean_wall = statistics.mean(r["wall_real_ns"] for r in rows)
    mean_gpu = statistics.mean(r["gpu_ns"] for r in rows)
    mean_comm_total = statistics.mean(r["comm_total_ns"] for r in rows)
    mean_comm_exposed = statistics.mean(r["comm_exposed_ns"] for r in rows)
    mean_overhead_pct = statistics.mean(r["comm_overhead_pct"] for r in rows)

    md = []
    md.append(f"# {title}")
    md.append("")
    md.append(f"**Headline (mean across {num_npus} ranks):** step time **{mean_wall/1e6:.2f} ms**, "
              f"compute **{mean_gpu/1e6:.2f} ms**, total comm **{mean_comm_total/1e6:.2f} ms** "
              f"(**{mean_comm_exposed/1e6:.2f} ms** exposed after overlap). "
              f"Real network slows training by **{mean_overhead_pct:.2f}%** vs an ideal network.")
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
    md.append("| **P99 step time** | — | — | — |")
    md.append("")
    md.append("> P99 row is deferred: AstraSim's analytical backend is deterministic, so all "
              "ranks/passes give identical numbers. P99 becomes meaningful once we capture a "
              "real GPU trace with step-to-step variance (see plan: GCP follow-up phase).")
    out.write_text("\n".join(md) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir",  default="results/logs")
    p.add_argument("--out-dir",   default="results")
    p.add_argument("--num-npus",  type=int, default=8)
    p.add_argument("--title",     default="Llama-3 8B DP=8 — AstraSim metric panel")
    p.add_argument("--out-prefix", default="summary",
                   help="base name for summary.{csv,md} (e.g. 'summary_16gpu')")
    args = p.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    real_path = logs_dir / "astrasim_real.log"
    ideal_path = logs_dir / "astrasim_ideal.log"
    for path in (real_path, ideal_path):
        if not path.exists():
            sys.exit(f"missing {path}; run run_astrasim.sh first")

    real = parse_log(real_path, args.num_npus)
    ideal = parse_log(ideal_path, args.num_npus)

    rows = []
    for rank in range(args.num_npus):
        row = {"rank": rank, **derive_per_rank(real[rank], ideal[rank], args.num_npus)}
        rows.append(row)

    csv_path = out_dir / f"{args.out_prefix}.csv"
    md_path = out_dir / f"{args.out_prefix}.md"
    write_csv(rows, csv_path)
    write_summary_md(rows, md_path, args.title, args.num_npus)
    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
