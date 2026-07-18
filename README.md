# Constraint-Aware Neural Network Systems

UROP 1100 research (HKUST, Advisor: Prof. Wei Zhang) on **hardware-efficient neural networks** —
training and pruning networks so that FPGA cost (pruned fan-in, packable LUT structure) is part of
the objective, not bolted on after. Three connected parts:

| Folder | What it covers | Headline result |
|---|---|---|
| [**LutNet/**](LutNet) | Analysis + pruning-correctness fix for a LUT-CNN framework (CIFAR-10) | Fixed a tie-breaking pruning bug (682 → **691**, exact target); mapped the accuracy–hardware frontier (Spearman ρ = **−0.96**), up to **86% pin / 81% slice** reduction |
| [**RadioML/**](RadioML) | Conventional 1-D CNN modulation classification + structured failure analysis | Baseline **60.36%**, two-branch V3 **61.90%**; diagnosed AM-SSB "sink" (**72%**) and WBFM → AM-DSB confusion (**63.6%**) |
| [**RadioML-LUT/**](RadioML-LUT) | Porting the LUT hardware-aware workflow onto RadioML 1-D IQ | Ablation: pruning-first **40%** vs no-prune control **29%** under full hardening (**+11 pts**); ≈38% pin pruning |

The common question across all three:

> How should model structure and feature representation be designed to improve **both** learning
> behavior and hardware efficiency?

`LutNet/` establishes the hardware-aware method and its trade-offs; `RadioML/` builds intuition for
a new signal domain and where models fail; `RadioML-LUT/` brings the two together by running the
hardware-aware workflow on that domain.

---

## My contributions

- Reverse-engineered and analyzed an existing LUT-CNN training/pruning pipeline.
- Found and fixed a **sensitivity-pruning correctness bug** caused by threshold ties.
- Ran a 40+ configuration sweep of the accuracy ↔ hardware trade-off and produced the analysis figures.
- Ported the LUT operators and staged schedule to RadioML 1-D IQ, and ran the pruning-vs-no-prune ablation.
- Built the failure-analysis tooling (class/SNR breakdowns, confusion, confidence-margin analysis).

## Scope note

This repository emphasizes analysis, adaptation, and experiment structure. It does **not** include
the full original collaborative LUT framework; some code is reorganized to highlight my direct
contributions. All result files and figures are my own experiment outputs.

## Repository layout

```
LutNet/        analysis_code/  figures/  results/            + README
RadioML/       models/ data/ training/ analysis/ figures/ results/   + README
RadioML-LUT/   figures/ results/                             + README
```
