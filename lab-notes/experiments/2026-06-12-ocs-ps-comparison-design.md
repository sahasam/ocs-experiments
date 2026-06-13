# OCS vs PS Comparison Design & Parallelism Review
**Date:** 2026-06-12

## How to read the Switch topology result

The `Switch` topology in AstraSim is a **single non-blocking switch** — not a fat-tree. It captures incast at endpoint downlinks (many senders → one receiver competing for that link) but omits multi-tier oversubscription and ECMP hash collisions. This makes it an **optimistic PS model**: if OCS beats Switch, it beats any real oversubscribed fat-tree too. OCS advantage claims against Switch are conservative lower bounds.

Correct comparison:
- **OCS**: `FullyConnected` + `Congestion_Unaware` (dedicated circuits → no contention by construction; Aware backend gives identical result)
- **PS**: `Switch` + `Congestion_Aware` (shared links → real HOL blocking / incast)

Bandwidth and latency params must be matched between both runs (same 400 Gbps, 500 ns).

---

## Parallelism refresher

**Tensor Parallelism (TP):** Splits individual weight matrices across GPUs within a layer. Each GPU does a partial matmul then all-reduces. On the critical path of every layer — requires NVLink-class bandwidth. Stays intra-node.

**Pipeline Parallelism (PP):** Splits the model depth-wise across GPU groups (stage 0 = layers 1–N/k, stage 1 = layers N/k+1–2N/k, ...). Communication is small activations at stage boundaries (send/recv). Cost is the **pipeline bubble** — idle time at start/end of microbatch while stages drain.

**Data Parallelism (DP):** Identical model replicas on different data shards. After backward, all-reduce gradients across replicas (volume = full model size). Once per step, happens after compute — most tolerant of latency.

**Composition:** total GPUs = TP × PP × DP. Example: TP8/PP2/DP4 @ 64 GPUs → 8 GPUs share each tensor layer, 2 pipeline stages, 4 replicas.

**Which traffic crosses the fabric (switch/OCS):**
- TP → stays on NVLink, never touches outer fabric
- PP sends → cross stage boundaries through fabric; small tensors but on critical path
- DP all-reduces → cross all replica groups through fabric; large (full model), once per step

---

## 16-GPU minimum experiment design

**Goal:** apples-to-apples comparison of OCS vs PS that gives both a fair shot before a larger-scale cloud run.

**Setup:**
- 16 GPUs, PP=2, DP=4, TP=2
- Two 8-GPU NVLink meshes (one per PP stage)
- **PS:** 16-port non-oversubscribed switch
- **OCS:** 32-port OCS with 16 simultaneous circuits (16 GPUs × 1 port each, C=16 one-directional = 8 bidirectional pairs)

**Why bandwidth is matched:**
- Switch: 16 × 400 Gbps = 6.4 Tbps full bisection
- OCS: 8 pairs × 400 Gbps × 2 = 6.4 Tbps instantaneous

**The specific mechanism being tested:**  
DP all-reduce is large and bursty — all replicas start sending simultaneously after backward, creating incast at the switch. PP sends cross the same switch during that burst and queue behind DP chunks. That wait directly extends the pipeline bubble. On OCS, PP and DP can be scheduled to separate circuit slots — no DP chunk blocks a PP send.

The hypothesis: **OCS prevents DP traffic from polluting PP latency.** Not "OCS has lower base latency."

**Why 16 GPUs is the minimum:**  
DP=2 generates only 2 simultaneous flows — a non-oversubscribed switch handles that trivially, no observable congestion. DP=4–8 is where simultaneous all-reduce creates real queuing at switch ports.

**Modeling complication:**  
`Congestion_Aware` backend is 1-dim only. With TP=2 you'd need a 2-dim model (NVLink for TP, fabric for PP+DP) which isn't supported. **Cleanest workaround: use TP=1** (PP=2, DP=8). All 16 GPUs talk only through the outer fabric — exactly what Switch vs FullyConnected models in 1-dim. No asymmetry.

**Important caveat:** A non-oversubscribed switch with QoS priority queuing (PP sends jump the queue) would largely neutralize the OCS advantage. Decide upfront: best-effort PS (commodity hardware) or QoS-aware PS (harder benchmark for OCS). These are different experiments.

**With C=16 circuits (perfect matching):**  
Each GPU can talk to exactly one other simultaneously. For ring all-reduce over 16 ranks, one circuit schedule covers all 16 flows — essentially no oversubscription penalty. OCS at its best. The interesting axis to sweep: step time vs DP all-reduce message size, finding the crossover where PS wins.
