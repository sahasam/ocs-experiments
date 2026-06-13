# OCS vs PS comparison — learnings and framing

**Date:** 2026-06-11  
**Sources:** `../experiments/2026-06-11-exp3-ns3-fat-tree-baseline.md`, `../experiments/2026-06-12-ocs-ps-comparison-design.md`

## Direct all-reduce is the wrong algorithm for a fat-tree

With DP=8 and 128 ranks, `"all-reduce-implementation": ["direct"]` generates 896 concurrent cross-switch flows (16 DP groups × 56 simultaneous flows each). The fat-tree has 64 leaf→spine uplinks at 400 Gbps, so:

```
896 flows / 64 links → ~14 flows per uplink → 14:1 overload
```

HPCC backs all senders toward `MIN_RATE = 100 Mb/s`, creating oscillations that inflate per-flow completion time 10–15×. This is why the 24-hour ns-3 run never completed — it was measuring a pathological case, not a real PS cluster.

**Direct all-reduce is OCS's algorithm.** OCS provisions a dedicated circuit per pair; simultaneous all-to-all is its core value proposition. Comparing OCS+direct vs PS+direct is not a fair test.

## What a fair comparison looks like

| side | topology | algorithm |
|---|---|---|
| OCS | FullyConnected | direct all-reduce |
| PS fat-tree | fat-tree | **ring** all-reduce |

Ring all-reduce generates ~2 concurrent flows per rank at a time (~128 total), well within fat-tree capacity. Recursive halving-doubling (log₂N steps, ≤64 flows each) is the other option.

Even this comparison understates OCS advantage: OCS lets you run the more efficient (lower-latency) direct collective; PS forces you into ring. The correct framing is that the algorithm choice is part of the tradeoff, not just the network topology.

## AstraSim Switch topology as a PS lower bound

`Switch` in AstraSim = single non-blocking switch. It captures incast at endpoint downlinks (many senders competing for one receiver's port) but omits multi-tier oversubscription and ECMP hash collisions.

This makes it an **optimistic PS model**: if OCS beats Switch, it beats any real oversubscribed fat-tree too. Claims of OCS advantage against Switch are conservative lower bounds.

Correct comparison:
- OCS: `FullyConnected` + `Congestion_Unaware` (or Aware — identical for OCS)
- PS: `Switch` + `Congestion_Aware`

Bandwidth and latency must be matched between both runs (same 400 Gbps, 500 ns).

## Hypothesis being tested

**OCS prevents DP traffic from polluting PP pipeline-bubble latency.**

DP all-reduce is large and bursty — all replicas start sending simultaneously after backward, creating incast at the switch. PP sends cross the same switch during that burst and queue behind DP chunks. That wait directly extends the pipeline bubble. On OCS, PP and DP can be scheduled to separate circuit slots with no shared contention.

Not "OCS has lower base latency" — the hypothesis is about traffic isolation under simultaneous large all-reduce + critical-path small sends.

## Minimum viable scale

- **DP=2:** only 2 simultaneous flows. A non-oversubscribed switch handles this trivially; no observable congestion. Too small.
- **DP=4–8 (16+ GPUs):** simultaneous all-reduce creates real queuing at switch ports. This is where the effect is measurable.

**16 GPUs with PP=2, DP=8, TP=1 is the minimum experiment.** TP=1 keeps all traffic on the outer fabric — the cleanest fit for AstraSim's 1-dim `Congestion_Aware` backend.

## QoS caveat

A non-oversubscribed switch with priority queuing (PP sends ahead of DP) would largely neutralize the OCS advantage. This is a different experiment from best-effort PS (commodity hardware). Decide upfront which PS baseline is the claim.
