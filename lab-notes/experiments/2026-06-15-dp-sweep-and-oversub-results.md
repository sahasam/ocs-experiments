# FTS packet-level results — oversubscription A/B + DP-degree OCS-benefit sweep

**Date:** 2026-06-15
**Status:** N1, N2, DP=4 complete; DP=16 running.
**Spec:** `2026-06-15-fts-ns3-trials.md`. **Engine capability:** `../learnings/2026-06-15-ocs-engine-capability-audit.md`.
**Workload:** Llama-3 8B, TP=1/PP=2, ring/ring, ns-3 HPCC, thin-Clos (single 400 G spine uplink/leaf) unless noted. OCS floor = `run_coupled.py` direct @ 50 GB/s (Exp-5 method). PS = `run_astrasim_ns3.sh`. Per-flow = `analyze_fct.py`. sys[0] = stage-0 = the PP-bottleneck rank = the step time.

## Completed runs

| Run | Fabric | Oversub | DP | PS stage-0 (ns-3) | OCS floor (coupled direct) | OCS benefit |
|-----|--------|---------|----|-------------------|----------------------------|-------------|
| N1  | fat-tree | 4:1   | 8  | 4923.4 ms | 4779.2 ms | **+3.02%** |
| N2  | thin Clos| 8:1   | 8  | 4962.5 ms | 4779.2 ms | **+3.84%** |
| DP=4| thin Clos| 4:1   | 4  | 9641.6 ms | 9542.9 ms | **+1.03%** |
| DP=16| thin Clos| 16:1 | 16 | running   | (pending run_coupled) | pending |

Per-flow congestion (fct/ideal_fct, mean / max):

| run | PP cross-stage send | DP-AR (large) | DP-AR (small) |
|-----|---------------------|---------------|---------------|
| N1 fat 4:1 (256 MB PP) | 4.13× / 5.42× | 2.49× | 3.05× |
| N2 clos 8:1 (256 MB PP)| 5.63× / 8.67× | 2.70× | 2.40× |
| DP=4 clos 4:1 (512 MB PP)| 3.90× / 4.33× | 2.74× | 2.49× |

## Key results

1. **N2 reproduced Exp 5 to the digit** (4962.5 ms, PP 5.63×/8.67×) — toolchain is deterministic/reproducible.
2. **N1 completed the 4:1 fat-tree stage-0 that Exp 4 had killed** — PP 4.13× is now *measured*, not estimated. Gives a second completed oversubscription point (+3.02%) so the congestion claim no longer rests on one config.
3. **OCS benefit grows with DP degree:** +1.03% (DP=4) → +3.84% (DP=8) → DP=16 pending. Absolute congestion delay 99 ms → 183 ms. This is the thesis-advancing, packet-level result (vs the derating engine, which structurally can't show a win).

## Confounds — read carefully

- **Oversubscription A/B (N1 vs N2) is topology-confounded:** N1 is a 2-spine fat-tree, N2 a 1-spine thin Clos — they differ in topology *and* oversub, so it is *not* a clean single-variable A/B. Still shows "thinner spine ⇒ worse PP congestion" (4.13×→5.63×, +3.0%→+3.8%).
- **DP sweep is strong-scaled:** global batch fixed and split over DP, so halving DP **doubles per-rank compute** (DP=4 GPU 4757 ms = 2× DP=8 2382 ms) **and per-flow size** (PP 512→256→128 MB). It also couples oversubscription (4:1/8:1/16:1). So **Wall time is NOT comparable across DP** (compute-dominated; DP=4 Wall 9641 ms > DP=8 4962 ms).
  - **Why the OCS-benefit % is still valid:** overhead = (PS − OCS floor)/floor, and PS and floor share identical compute, so **compute cancels**. The per-flow multiplier is normalized by ideal FCT, so it is fair across flow sizes too. These two metrics are the defensible cross-DP signals; Wall is not.
  - A *clean* DP isolation would need **weak scaling** (fix per-rank microbatch) + **fixed oversub** (scale spine uplinks with DP). Not done; honest future work. The coupled DP+oversub growth is the realistic "scale DP on a fixed spine" scenario.

## Commands
- OCS floor (DP=4): `python run_coupled.py llama3_8b_tp1_pp2_dp4 8 --pp-stages 0:0-3,1:4-7 --bandwidth 50 --latency 500`
- PS DP=4: `DETACH=1 NS3_TOPO_FILE=.../ns3_topo_8_clos_thin.txt SYSTEM_CFG=.../system_ns3.json NS3_OUT_SUBDIR=ns3_output_clos bash run_astrasim_ns3.sh llama3_8b_tp1_pp2_dp4 8`
- New topo/logical files generated this session: `ns3_topo_{8,32}_clos_thin.txt`, `ns3_logical_{8,32}.json`.

## TODO when DP=16 lands
- `run_coupled.py llama3_8b_tp1_pp2_dp16 32 --pp-stages 0:0-15,1:16-31 --bandwidth 50 --latency 500` for its OCS floor.
- `analyze_fct.py` on its fct.txt for PP multiplier.
- Complete the DP-benefit curve (DP=4/8/16) → the paper's headline figure.
