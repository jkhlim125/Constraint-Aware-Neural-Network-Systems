# LutNet: Hardware-Aware Pruning and FPGA Efficiency

Analysis and pruning-correctness work on a LUT-based neural-network training pipeline (CIFAR-10),
where the goal is FPGA efficiency — reducing LUTs, slices, and input pins — without collapsing
accuracy. On an FPGA, resource usage is **not** proportional to parameter count: LUT packing and
input fan-in dominate, so pruning has to be measured in hardware terms, not just sparsity.

> Scope note — this folder emphasizes analysis and the pruning fix. It does not include the full
> original collaborative LUT framework. Result files and figures are my own outputs.

---

## 1. The bug: naive threshold pruning silently misses its target

Global sensitivity pruning keeps weights whose score ≥ a quantile threshold. But many scores land
**exactly on** the threshold (a "tie region"), and naive selection doesn't decide among them — so
the actual prune count drifts off target.

![Naive vs tie-aware pruning](figures/pruning_logic.png)

Measured on one tensor (`results/threshold_check_summary.txt`):

| | value |
|---|---|
| tensor elements | 1,152 |
| sparsity target | 0.60 |
| threshold | 0.18 |
| **target prune count** | **691** |
| naive prune count | 682  (**−9**, 28 values tied at threshold) |
| tie-aware prune count | **691  (error 0)** |

**Fix:** explicitly rank and select within the tie region to hit the exact target. This restores
accurate sparsity control — which matters because packing/pin estimates downstream assume the
target was met.

## 2. The training behavior: pruning is a discrete structural shift

![Training phase transition](figures/training_phase_transition.png)

Training is not smooth. Warmup is stable; the pruning step causes a sudden accuracy drop; a recovery
phase lets the model adapt to the new structure. Treating pruning as a structural event (with a
recovery budget) — rather than a smooth regularizer — is what keeps the runs stable.

## 3. The trade-off: the prepared frontier beats the naive baseline

I swept sparsity × pack-ratio across 40+ runs and plotted accuracy against hardware reduction.

![Pareto: max accuracy vs pin reduction](figures/fig2_pareto_max_acc_vs_pin_reduction.png)

- The pruning/pairing-**prepared** frontier (green) stays **above** the naive baseline (blue) at
  every pin-reduction level — same hardware saving, more accuracy retained. Spearman ρ = **−0.96**
  (p < 1e-4) for max accuracy vs pin reduction.
- Aggressive configs reach **≈86% pin / ≈81% slice reduction**; the best run holds **80.96% accuracy
  at 84% pin / 71% slice reduction** (`results/log_comparison_table.csv`).

![Accuracy drop vs pin reduction](figures/fig4_accuracy_drop_vs_pin_reduction.png)
![Best per pack ratio](figures/fig5_best_per_pack_ratio.png)

## 4. Analysis code

| Script | Purpose |
|---|---|
| `analysis_code/check_pruning_consistency.py` | Detects threshold ties; compares target vs naive vs tie-aware prune counts (the bug above). |
| `analysis_code/compare_packratio_runs.py` | Compares runs across pack ratios; builds the trade-off tables. |
| `analysis_code/parse_train_log.py` | Parses accuracy/loss/pruning events from raw training logs. |
| `analysis_code/l1_paper_figures.py` | Generates the paper-style Pareto / trade-off figures. |

## 5. Results

| File | Contents |
|---|---|
| `results/threshold_check_summary.txt` | Tie statistics + prune-count errors (the bug fix). |
| `results/log_comparison_table.csv` | Per-run accuracy vs slice / pin reduction across configs. |
| `results/lutnet_summary_metrics.csv` | Best/final accuracy, sparsity flags per log. |

## Key takeaways

1. Naive global-threshold pruning fails when scores tie; exact sparsity needs explicit tie-breaking.
2. Pruning is a discrete structural shift — budget a recovery phase.
3. Hardware efficiency ≠ sparsity: measure pins and slices, and the prepared frontier dominates the baseline.

## Future directions

Structure-aware (block/channel) pruning · alternative backbones · extension to RadioML
(see [`../RadioML-LUT`](../RadioML-LUT)) · true FPGA-synthesis validation of the packing estimates.
