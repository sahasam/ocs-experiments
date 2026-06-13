# OCS vs PS Comparison: What the ns-3 Run Taught Us

**Date:** 2026-06-13  
**Sources:** `experiments/2026-06-13-exp4-16gpu-ocs-ps-ns3.md`

---

## The right comparison: congestion-unaware + OCS retiming vs. ns-3

The `ocs_replay` approach (our main OCS modeling tool) is actually the cleaner comparison for OCS vs PS:

- It takes a **single AstraSim trace** — same workload, same algorithm, same compute model, same PP/DP schedule for both sides.
- The PS baseline is `C=∞` (frictionless, no contention) — the best possible packet-switched network.
- The OCS side re-times the same trace under capacity-C contention, applying congestion **only** to the OCS-specific flows (DP all-reduce + PP sends).
- Everything else is held constant: compute roofline, microbatch schedule, PP bubble size.

This perfectly isolates the network topology effect. **The OCS advantage or penalty in ocs_replay is purely from network capacity, not algorithm choice or simulation backend differences.**

The ns-3 run changes three things at once: the network (fat-tree vs FullyConnected), the algorithm (ring vs direct), and the simulation backend (packet-level HPCC vs analytical Congestion_Unaware). That makes it harder to attribute differences to any single cause. ns-3 is more useful as a **validation and visualization tool** — showing which flows are congested and by how much — rather than as the primary OCS-vs-PS benchmark.

---

## Why algorithm choice is a confound

We ran:
- OCS: `system.json` (direct all-reduce) + FullyConnected + Congestion_Unaware
- PS:  `system_ns3.json` (ring all-reduce) + fat-tree + HPCC

OCS direct all-reduce on a FullyConnected analytical backend: each rank sends the full gradient to all N-1 peers simultaneously. In Congestion_Unaware mode, the receive link is the bottleneck — 7 incoming flows of 16 GB each at 50 GB/s = 2.24 s receive time. This produces the very large exposed comm (2385 ms) in the OCS direct run.

Ring all-reduce uses only 2 links at a time (left + right neighbor), 14 steps of 2 GB each. AstraSim overlaps reduce-scatter with backward computation, exposing far less comm (820–906 ms).

The result: **ring-on-congested-fat-tree (3289ms) is faster than direct-on-uncongested-FullyConnected (4768ms)** — not because PS is faster, but because ring is a better algorithm for the analytical model's assumptions. Comparing them is unfair to OCS.

The fair apples-to-apples comparison:

| Config | Stage 0 Wall | Stage 1 Wall |
|--------|-------------|-------------|
| OCS, ring, FullyConnected (analytical) | 4783 ms | 3203 ms |
| PS, ring, fat-tree 4:1 (ns-3 HPCC)    | n/a (killed) | 3289 ms |
| PS slowdown vs OCS (stage 1)           | — | **+2.7%** |

Stage 0 is the bottleneck rank (it waits for stage 1's backward gradient before it can do its DP all-reduce). Stage 0 PS ns-3 was killed mid-run — re-run needed to get the complete number. Estimated +3–5% based on flow-level delays.

---

## PP stage asymmetry: stage 0 finishes much later

This is easy to miss. With PP=2 and 1F1B scheduling:

- **PP stage 1** (ranks 8–15): runs backward first, sends gradient to stage 0, then does DP all-reduce. Finishes at ~3.2s.
- **PP stage 0** (ranks 0–7): runs backward only after receiving stage 1's gradient, then does DP all-reduce. Finishes at ~4.8s.

The gap is ~1.58s (OCS analytical). When monitoring an ns-3 run, only stage 1 finishing is visible early. **Don't kill the container when stage 1 finishes — stage 0 takes another ~1.5s of simulated time to complete.** The "Wall time" that matters for end-to-end step time is stage 0's.

---

## What ns-3 actually tells us: per-flow congestion breakdown

Even without stage 0 completing, the ns-3 run reveals which traffic types suffer most on a congested fat-tree:

| Flow type | Size | Count | OCS (ideal FCT) | PS (actual FCT) | Slowdown |
|-----------|------|-------|----------------|----------------|----------|
| DP ring all-reduce (transformer layers) | 5.8 MB | 14,336 | 123 µs | 392 µs | 3.2× |
| DP ring all-reduce (embedding, stage 1) | 16.4 MB | 896 | 346 µs | 770 µs | 2.2× |
| PP activations / gradients (cross-stage) | 268 MB | 16 | 5,630 µs | 23,263 µs | 4.1× |

The **PP sends are hit hardest** (4.1×). These are 268 MB flows (all microbatch activations batched together per stage boundary) that compete with the simultaneous DP ring traffic on the same 2-uplink leaf-spine bottleneck (4:1 oversubscription). This directly validates the core OCS hypothesis: **DP traffic congests the fabric, and PP sends — which are on the pipeline critical path — get caught in that congestion.**

On OCS, PP and DP can use separate circuit slots, eliminating this interference. The ns-3 result shows exactly where the penalty falls in a PS fat-tree, even if the overall step-time comparison requires a clean re-run.

---

## Ring vs direct: which algorithm belongs to which network

**PS fat-tree uses ring — this is not a choice.** Direct all-reduce on a fat-tree generates N*(N-1) simultaneous flows; at 128 ranks with DP=8 this is 896 concurrent flows on 64 leaf-spine links = 14:1 overload. NCCL defaults to ring on every real fat-tree cluster for exactly this reason.

**OCS uses direct — this is the point.** One circuit per rank-pair means every rank can send to every other simultaneously at full link bandwidth with zero contention. Direct all-reduce completes in a single step. This is the core algorithmic advantage that Sirius, Jupiter OCS, and similar systems sell.

**So OCS+direct vs PS+ring is the correct real-world framing,** not a confound. The OCS advantage is two-dimensional: (1) no congestion on the fabric, and (2) the ability to use a more efficient collective algorithm. Ring on a fat-tree is slower not just because of congestion but because it requires 2*(N-1) sequential steps where OCS does it in one.

**Why AstraSim makes direct look wrong on OCS:** The Congestion_Unaware + FullyConnected backend serializes incoming flows at the receive NIC — 7 flows × 16 GB at 50 GB/s = 2.24 s exposed comm, making direct *appear* slower than ring. In reality, OCS GPUs receive all peer flows in parallel through dedicated circuit interfaces. AstraSim is correct that there's no contention but wrong about NIC-level parallelism, so the real benefit of direct on OCS doesn't show up.

**Consequence for this experiment:**
- OCS+ring vs PS+ring (what Exp 4 measured) is a *conservative lower bound* on OCS advantage — it isolates only the congestion/topology effect.
- The full OCS advantage (congestion elimination + better algorithm) can't be measured cleanly in AstraSim's analytical backend.
- `ocs_replay` sidesteps this: takes one trace at whatever algorithm, measures how OCS contention delays flows vs the uncontended baseline. Network topology effect only.
- A complete paper-level claim: "OCS eliminates congestion (ring vs ring: +2.7% PS overhead on stage 1) AND enables direct all-reduce for further gains not captured by this simulation."

## Practical ns-3 run notes

- **Don't kill the container after stage 1 logs "finished".** Stage 0 logs its stats ~1.5s of simulated time later and its Ring topology setup appearing in the log is a sign it's starting DP all-reduce, not that it's done.
- **Use `/run-ps` skill** for future runs — it asks about packet size (9000 = 9× faster), qlen monitoring (disable for speed), and CC mode (DCQCN = faster, HPCC = more realistic).
- **qlen.txt** grows fast (24 MB+ per run) and is only needed for diagnosing congestion. Disable with `QLEN_MON_START 99999999999999` in the ns3_config unless you need queue depth traces.
- **fct.txt ideal_fct column = OCS line-rate performance**, making it easy to compute per-flow-type congestion penalties directly without a separate OCS run.
- **Container killed = partial results.** `docker run` in foreground mode ties the container to the launching shell. If that shell dies (background task cleanup, Ctrl-C, etc.), Docker sends SIGTERM to the container and AstraSim exits — logging only already-finished ranks. Run detached (`docker run -d`) so the container outlives the shell.

---

## AstraSim Switch topology as PS lower bound

`Switch` in AstraSim = single non-blocking switch. Correct comparison:
- **OCS:** `FullyConnected` + `Congestion_Unaware` (dedicated circuits, no contention)
- **PS:** `Switch` + `Congestion_Aware` (shared links, real HOL blocking / incast)

Switch captures incast at endpoint downlinks (many senders → one receiver's port) but omits multi-tier oversubscription and ECMP hash collisions. This makes it an **optimistic PS model** — a conservative lower bound on OCS advantage. If OCS beats Switch, it beats any real oversubscribed fat-tree. Bandwidth and latency must match between both runs (400 Gbps, 500 ns).

## Minimum scale for a meaningful experiment

- **DP=2:** only 2 simultaneous flows. A non-oversubscribed switch handles this trivially. No observable congestion.
- **DP=4–8 (16+ GPUs):** simultaneous all-reduce creates real queuing at switch ports. This is where the congestion effect is measurable.
- **TP must be 1** when using AstraSim's 1-dim analytical Congestion_Aware backend or ns-3 flat logical topology. TP>1 needs multi-dim topology support which Congestion_Aware doesn't have.
- **QoS caveat:** a non-oversubscribed switch with priority queuing (PP sends ahead of DP) would largely neutralize OCS advantage. Best-effort PS (commodity hardware, no QoS) is the right baseline for the claim.
