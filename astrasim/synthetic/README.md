# Synthetic full-step Chakra ET -> AstraSim

This directory simulates a **full Llama-3 8B training step** end-to-end through
the AstraSim analytical backend, without ever touching a GPU. It exists because
the user's Path C nanoGPT traces are unusable for AstraSim (see top-level
`GOTCHAS.md` §5–6), and a real GPU capture needs hardware we don't have yet.

## How it works

```
[ compute_times.py ]              <- per-block roofline FLOPs -> H100 BF16 us
        |
        v
[ generate_et.sh ]                <- writes workloads/llama3_8b_dp8.txt
        |                            then runs chakra text_converter (via Docker)
        v
[ results/llama3_8b_dp8/*.et ]    <- 8 per-rank Chakra protobuf ETs
        |
        v
[ run_astrasim.sh ]               <- runs AstraSim twice:
        |                            - real network (HGX-H100-validated)
        |                            - ideal network (1e6 GB/s, 1 ns)
        v
[ results/logs/astrasim_*.log ]
        |
        v
[ parse_results.py ]              <- regex + derive -> summary.csv + summary.md
```

## Run it

```bash
bash generate_et.sh                # writes .et files (a few seconds)
bash run_astrasim.sh               # runs sim twice (a few seconds)
python parse_results.py            # prints summary.md to stdout
cat results/summary.md             # any time
```

## What the workload represents

- **Llama-3 8B shape**: 32 transformer blocks, d_model=4096, d_ff=14336
  (SwiGLU), n_heads=32. Tune in `generate_et.sh`.
- **Parallelism**: pure DP=8 (matches the validated HGX-H100 system config,
  which is 8 NPUs). One ALLREDUCE per transformer block on the
  backward-weight-grad path, no comm in fwd / no comm on the data-grad path.
- **Per-layer grad volume**: ~464 MB (attn 4*d^2 + SwiGLU MLP 3*d*d_ff,
  BF16 = 2 bytes/param).
- **Compute times**: roofline at 50% of H100 BF16 peak (989 TFLOPs/s) —
  see `compute_times.py`. Order-of-magnitude accurate; the comm/compute
  ratio is more trustworthy than absolute step time.
- **One pass per ET**: `num_passes=1`. The analytical backend is deterministic
  so additional passes don't reveal variance — they would just be repeated.

## Metric panel

Computed per rank, then aggregated (min / mean / max across the 8 ranks):

| metric | meaning |
|---|---|
| step time (real) | Wall time per rank on HGX-H100-validated network |
| step time (ideal net) | Same workload, infinite-bw / 1 ns network |
| compute time | GPU time per rank |
| total comm | Sum across all collectives |
| exposed comm | Comm NOT overlapped with compute (pure stall) |
| hidden comm | Comm masked by overlap with compute |
| comm-overlap fraction | `hidden / total` |
| comm overhead % | `(real - ideal) / ideal` |
| GPU util (real/ideal) | `compute_time / wall_time` — analytical-model util |
| GPU-seconds/step | step_time * num_NPUs |
| $/step | GPU-seconds/step * $4/hr / 3600 (H100 spot) |
| J/step | GPU-seconds/step * 700 W (H100 SXM TDP) |
| P99 step time | **deferred** — see below |

H100 cost and power constants live in `gpu_constants.py`.

## What's NOT meaningful here

- **P99 step time**: AstraSim's analytical backend is deterministic. Same
  workload -> same step time every run. P99 needs the real GPU trace in the
  follow-up phase.
- **Real SM utilization**: the analytical model has compute and comm only —
  no kernel launches, no memory stalls, no warp-level effects. Real
  utilization will be lower; needs nsight / dcgm during a real capture.
- **Step-to-step variance**: also requires real traces.

## Sanity-check thresholds

If you change the workload and rerun, check these against `summary.md`:

- Ideal step time should be within ~0.1% of compute time. (If much higher,
  `network_ideal.yml` isn't actually ideal — check its bandwidth/latency.)
- Comm overhead % must be positive and finite.
- GPU util (real) <= GPU util (ideal).
- $/step magnitude check: 8 * $4/hr * step_seconds / 3600 should match the
  reported number.

## Caveats

- **HGX-H100-validated** is the only config we use here and it is 8 NPUs.
  For >8 ranks you need a different validated config or one you build
  yourself.
- **No TP / no PP** in this iteration. Adding them is one more text-workload
  file per parallelism flavor (the `text_converter` supports MODEL and
  HYBRID_DATA_MODEL parallelism keywords). Defer until DP-baseline numbers
  are trusted.
- **Roofline compute times are coarse**. Real H100 kernel performance varies
  by ±20% depending on shapes, fusion, sequence length, and recomputation
  policy. The synthetic numbers are a planning/intuition tool, not a
  replacement for measurement.
- **Single training step** (`num_passes=1`). Steady-state behaviors like
  async optimizer step or gradient accumulation aren't modeled.

## Follow-up phase: real-trace calibration

The plan calls for a GCP free-credit run (`a2-highgpu-2g` or similar) to
capture real NCCL Chakra ETs from ~100 training steps of a DDP HuggingFace
or Megatron-LM workload. Then:

1. scp the per-rank `.et` files into `results/<workload-name>/`.
2. Re-run `run_astrasim.sh` against them (just change the workload path).
3. Extend `parse_results.py` with a `--multi-step` mode that reads per-step
   timings from the underlying Kineto trace -> compute true P99, real
   step-time histogram.
4. Compare synthetic numbers in this README against the measured ones to
   calibrate the roofline assumption.

See the top-level plan for end-to-end instructions.
