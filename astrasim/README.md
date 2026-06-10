# AstraSim (stage 4)

This directory contains the AstraSim configs that consume Chakra ETs produced
by stages 1-3 of the pipeline.

## Binary

Your astra-sim is already built at `/Users/sahas/workplace/astra-sim/`. The
binary inside `build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware`
is a **Linux ARM64 ELF** — it was built inside the `astra-sim:latest` Docker
image and cannot run natively on macOS. Always invoke via Docker.

## Configs in this directory

| file | purpose |
|---|---|
| `system.json` | 4-NPU ring system, ring collectives, mirrors `examples/system/native_collectives/Ring_4chunks.json` |
| `network.yml` | 4-NPU ring topology, 50 GB/s bandwidth, 500 ns latency — placeholder values for Path C plumbing validation |
| `no_memory_expansion.json` | Trivial remote-memory config (none) |
| `run_reference.sh` | Smoke test: runs the canonical 8-NPU all_reduce reference workload to prove the binary + Docker image work |
| `run_our_traces.sh` | Runs astra-sim against our pipeline output (see "Known limitations" below) |

## Quick start

```bash
# 1. Prove the binary works on the reference example
bash astrasim/run_reference.sh
# expect: "[system] [warning] Exiting" at the end

# 2. Try our traces
bash astrasim/run_our_traces.sh
# Path C: expected to fail mid-simulation, see Known limitations
# Path A: expected to succeed once GPU traces replace CPU traces
```

## Known limitations (Path C)

The chakra ETs we produce on CPU work for *parts* of the AstraSim pipeline but
not all the way through. Two related issues:

1. **`METADATA` nodes (type=1)** appear in our ETs (e.g. `## process_group:init ##`)
   because chakra's PyTorch converter, on CPU, preserves the raw ExecutionTrace.
   AstraSim's statistics layer asserts on unknown node types. Workaround: filter
   them out — but then dangling parent IDs cause the data-dep graph to fail
   integrity checks.

2. **Gloo collectives aren't recognized.** Chakra's `is_nccl_op` test is
   `"nccl:" in name`, which matches only NCCL ops. Our CPU/gloo run produces
   `gloo:all_reduce` ops that pass through as plain COMP nodes. The resulting
   Chakra ET has zero `COMM_COLL_NODE` entries, so AstraSim has no collectives
   to simulate.

Both go away on Path A (real GPU NCCL run): the chakra converter strips host-
only nodes during host+device merging, and NCCL ops trigger the collective
recognition path.

For Path C, treat `run_reference.sh` as the canonical proof that stage 4 works.
