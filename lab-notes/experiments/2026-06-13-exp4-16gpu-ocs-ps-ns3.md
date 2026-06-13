# Exp 4 — 16-GPU OCS vs PS ns-3 baseline

**Date:** 2026-06-13  
**Status:** Partial. Stage 1 complete; stage 0 cut off (container killed mid-run). Re-run needed for full step time.

## Where this fits

Exp 3 established that a 128-rank ns-3 run with direct all-reduce is infeasible (14:1 spine overload, never finished). This experiment retries at 16 GPUs with ring all-reduce — the correct algorithm for a fat-tree PS cluster — to get a valid PS congestion baseline.

Background: `learnings/2026-06-11-ocs-ps-comparison-learnings.md`, `learnings/2026-06-13-ocs-ps-ns3-comparison-learnings.md`.

---

## Goal

Measure PS step time with real packet-level congestion at a scale where ns-3 is tractable (~20–30 min), then compare against OCS analytical to quantify the OCS advantage.

## What we built

- **Workload:** `llama3_8b_tp1_pp2_dp8` (16 ranks). TP=1 so all collectives — DP all-reduce and PP sends — go through the outer fabric, which is exactly what both the fat-tree and FullyConnected model in AstraSim's 1-dim topology.
- **Topology:** `stage_configs/ns3_topo_16_fat_400g.txt` — 2-tier fat-tree, 2 leaf + 2 spine switches, 400 Gbps / 0.5 µs links, **4:1 oversubscription** (8 GPU downlinks, 2 spine uplinks per leaf).
- **PS config:** `stage_configs/system_ns3.json` — ring all-reduce, same compute params as `system.json`.
- **OCS config:** `system_ns3.json` + FullyConnected, analytical Congestion_Unaware.
- **Run scripts:** `run_astrasim_ns3.sh` (PS), `run_astrasim_stage.sh` (OCS).

## What happened

Both sides ran with ring all-reduce for a clean apples-to-apples comparison.

**OCS analytical (FullyConnected, ring):**

| Rank group | Wall time |
|------------|-----------|
| PP stage 0 (ranks 0–7) | 4,783 ms |
| PP stage 1 (ranks 8–15) | 3,203 ms |

**PS ns-3 (fat-tree 4:1, HPCC, ring) — stage 1 only:**

| Rank group | Wall time | vs OCS |
|------------|-----------|--------|
| PP stage 1 (ranks 8–15) | 3,289 ms | +2.7% |
| PP stage 0 (ranks 0–7) | n/a — killed | est. +3–5% |

The container was killed right as stage 0 began its ring all-reduce. Stage 0 finishes ~1.58s of simulated time after stage 1 in the OCS run — it must wait for stage 1's backward gradients before doing its own DP all-reduce. The simulation was at ~4.93s simulated when it stopped; stage 0's "finished" line never appeared.

**Why it died — the Claude session was the parent process.** The run was launched with a foreground `docker run` (no `-d`) from inside a Claude Code Bash tool call. `docker run` in foreground mode ties the container's lifetime to the launching client process. When the Claude session that owned that Bash subshell ended (turn/session teardown), the shell — and with it the foreground `docker run` client — was terminated, and Docker propagated SIGTERM to the container. AstraSim caught the signal and exited mid-run, flushing only the ranks that had already finished (stage 1), so stage 0's result was lost. Nothing was wrong with the simulation itself; it was killed by the harness lifecycle, not by an error. Fix: launch detached (`docker run -d`) so the container outlives the session, and tail the log file to monitor (see `/run-ps`).

**ns-3 flow-level breakdown (15,248 total flows):**

| Flow type | Size | Count | OCS ideal | PS actual | Slowdown |
|-----------|------|-------|-----------|-----------|----------|
| DP ring all-reduce (transformer layers) | 5.8 MB | 14,336 | 123 µs | 392 µs | 3.2× |
| DP ring all-reduce (embedding, stage 1) | 16.4 MB | 896 | 346 µs | 770 µs | 2.2× |
| PP activations / gradients (cross-stage) | 268 MB | 16 | 5,630 µs | 23,263 µs | **4.1×** |

Flow types identified from src→dst patterns in `fct.txt`: 5.8 MB ring within each PP stage (DP AR for transformer layers), 16.4 MB ring within stage 1 only (embedding layer gradients, larger), 268 MB cross-stage pairs (PP activation and gradient sends — all microbatches batched into one transfer per boundary).

## Key finding

**PP sends are the most congested flow type (4.1× slower than ideal)**, even though they're only 16 flows. They are 268 MB each (STAGE batches all microbatch activations into one transfer per stage boundary) and compete for the same 2 spine uplinks as the DP ring traffic. This directly validates the OCS hypothesis: DP traffic on a shared fabric degrades PP critical-path latency. OCS separates DP and PP onto different circuit slots, eliminating this interference.

The `ideal_fct` column in ns-3's `fct.txt` equals line-rate (no-congestion) performance — i.e., what OCS would see — so per-flow congestion penalties can be read from a single ns-3 run without a separate OCS simulation.

## What to do next

Re-run with the container in detached mode (`docker run -d`) so it isn't killed when the shell exits:

```bash
# In run_astrasim_ns3.sh, replace:  docker run --rm ...
# With:                              docker run -d --name astrasim-ns3 --rm ...
# Then tail the log file to monitor.
```

Also consider `PACKET_PAYLOAD_SIZE 9000` (jumbo frames) to cut simulation time ~9×. See `/run-ps` skill for options.

## Artifacts

- `stage_configs/ns3_topo_16_fat_400g.txt` — 16-node fat-tree topology
- `stage_configs/ns3_logical_16.json` — logical dims config
- `stage_configs/system_ns3.json` — ring all-reduce system config
- `results/llama3_8b_tp1_pp2_dp8/` — workload ETs, OCS log, ns-3 fct.txt / qlen.txt
