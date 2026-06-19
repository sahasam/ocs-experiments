# OCS-for-LLM-training — literature, novelty verdict, and reading list

**Date:** 2026-06-16
**Source:** background research agent (web-verified citations; a few flagged "verify before citing").
**Why:** novelty check on the FTS paper + a reading list for the pivot direction (traffic pre-knowledge / prediction as the OCS feasibility gap).

## Bottom line (novelty)

- **Pre-pivot paper: moderately novel.** Strongest = the *measurement* contributions (0.23%-hardware-validated trace replay; the "DP all-reduce congested-but-hidden → circuits wasted on it; PP cross-stage is the exposed payoff" decomposition; packet-level scaling laws ~1%→9%→16%). The `exposed-fraction × congestion-multiplier` formula is useful but reads as *formalized folklore*. Methodology + precision-time claims are **integration/confirmation, not invention** — present them that way.
- **The pivot thesis IS novel — but the moat is precise.** Prior work (TopoOpt, Cassini, DELTA) exploits **structural** predictability (the DAG repeats) and **assumes timing is known / offline-computable**. The straggler/jitter literature proves **timing is hard**. **No one has joined these two threads.** Our defensible distinction: *structure is predictable; **timing/perturbation** (stragglers, jitter, MoE/dynamic shapes, failure-remapping, multi-tenancy) is the binding, unsolved constraint.* Make the structure-vs-timing distinction **explicit and early** or a reviewer says "TopoOpt/Cassini/DELTA already did predictable-traffic OCS."

## Biggest prior-art threats / MUST-CITE
1. **DELTA — "Beyond Traffic Matrix" (arXiv:2603.28096, 2026)** — DAG-aware OCS topology via MILP, "temporal slack of non-critical tasks." Closest to our decomposition + pivot; postdates our framing. **Read first, cite, differentiate sharply.**
2. **TopoOpt (NSDI'23, arXiv:2202.00433)** — offline one-shot topology+parallelization co-design; the predictable-traffic anchor.
3. **Cassini (NSDI'24, arXiv:2308.00852)** — offline time-shift scheduling assuming known iteration times.
4. **ACTINA (SC'25, DOI 10.1145/3712285.3759842)** — quantitative reconfiguration-latency vs benefit, in-workload vs one-shot.
5. **"Understanding Stragglers in Large Model Training" (OSDI'25, arXiv:2505.05713)** — evidence base for the pivot's timing-unpredictability claim.

## Reading list (next few days)

**TOP-5 read-first:** DELTA (2603.28096) · TopoOpt (NSDI'23) · Cassini (NSDI'24) · Understanding Stragglers (OSDI'25) · ACTINA (SC'25).

**A — Traffic predictability & perturbation**
- Reconfigurable-DCN survey (CACM / arXiv:2502.16228) — frames ML traffic as "predictable & periodic"; the claim to use *and qualify*.
- MegaScale (NSDI'24, arXiv:2402.15627) — overlap works AND stability/perturbation at 10k+ GPUs.
- Understanding Stragglers (OSDI'25, arXiv:2505.05713).
- Robust LLM Training Infra @ ByteDance (arXiv:2509.16293) — failures/restarts/remapping.
- Collective Comm Profiling of ML Workloads (arXiv:2507.07117) — empirical traffic structure.

**B — Reconfigurable/circuit scheduling for ML**
- TopoOpt (NSDI'23) · DELTA (2603.28096) · TPU v4 (ISCA'23, arXiv:2304.01433) · LumosCore (arXiv:2411.01503) · Sirius (SIGCOMM'20) · RotorNet (SIGCOMM'17, traffic-oblivious counter-argument).

**C — Scheduling/circuits under uncertainty**
- Cassini (NSDI'24) · Dynamic Demand-Aware Link Scheduling (arXiv:2301.05751) · SWOT (arXiv:2510.19322, overlap reconfig w/ transmission) · Semi-Oblivious Reconfigurable DCNs (HotNets'24, DOI 10.1145/3696348.3696860) · Throughput Bounds of Reconfigurable DCNs (arXiv:2405.20869).

**D — OCS control-plane feasibility**
- Jupiter Evolving (SIGCOMM'22) · Mission Apollo (arXiv:2208.10041, switching-time/radix) · Lightwave Fabrics (SIGCOMM'23) · Sundial (OSDI'20) · ACTINA (SC'25).

**Core toolchain / collectives refs (for the paper):** ASTRA-sim2.0 (ISPASS'23, arXiv:2303.14006) · Chakra (arXiv:2305.14516) · HPCC (SIGCOMM'19) · GPipe (arXiv:1811.06965) · PipeDream (SOSP'19) · Megatron-LM (SC'21, arXiv:2104.04473).

## Citation caveats (verify before they go in the bib)
- **"Graham" (clock sync): UNVERIFIABLE — drop unless confirmed.** (Likely confused with Sundial/DTP.)
- **Helios, c-Through, Mordia** — real & well-known but exact author lists/DOIs NOT locked down; verify the citation strings.
- DTP is **2016** (SIGCOMM), extended ToN 2019 — not 2018.
