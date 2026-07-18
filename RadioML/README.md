# RadioML: CNN Modulation Classification & Failure Analysis

Modulation classification on RadioML 2016.10a (11 classes, `[B, 2, 128]` IQ, −20 to +18 dB SNR)
using 1-D CNNs. The emphasis is **structured failure analysis**, not chasing accuracy: overall
numbers move little, but *where* and *why* the model fails is highly systematic — and that diagnosis
is what motivates the LUT port in [`../RadioML-LUT`](../RadioML-LUT).

> Scope note — result files and figures are my own experiment outputs.

---

## 1. Models compared

| Model | Input | Test accuracy |
|---|---|---|
| Baseline 1-D CNN | raw IQ | **60.36%** |
| IF-feature CNN | + instantaneous frequency | 60.17% |
| Two-branch V3 | IQ + IF, separate branches | **61.90%** (best) |

![Baseline class × SNR](figures/analysis_heatmaps.png)

Accuracy is strongly **SNR-bound**: near-chance below −6 dB, 95%+ above +6 dB. Adding features
barely moves the overall number — the interesting result is the failure structure.

## 2. Structured failure modes (baseline)

The model does not fail randomly. From `results/summary_baseline.txt` and `results/results_comparison.csv`:

- **Confidence separates right from wrong** — correct predictions avg confidence **0.83** vs
  **0.42** for incorrect (margin 0.74 vs 0.23). Errors are uncertainty-driven, not confident-wrong.
- **AM-SSB is a "sink"** — under noise, other classes collapse *into* AM-SSB: **4,222 sink
  predictions vs 1,654 correct (≈72% sink rate)**, and sink rate rises to ~90% below −14 dB.
- **WBFM is the hardest class** — **28.1%** overall (40.6% high-SNR, **3.67%** low-SNR), and it
  leaks systematically into AM-DSB: **915 cases (63.6%)**.

![IF-feature heatmap](figures/analysis_heatmap_ifreq.png)
![Two-branch V3 heatmap](figures/analysis_heatmap_branch_v3.png)

The two-branch V3 lifts overall accuracy to 61.90% and most classes to 55–93%, but **WBFM stays
stuck at 29.6%** (AM-DSB confusion 60.3%) — confirming this is a **representation** limit, not a
capacity one. Raw IQ doesn't expose the frequency structure WBFM needs, and adding a branch only
partly helps.

## 3. Why this matters for the LUT work

These are the same fine amplitude/phase classes (AM-SSB, WBFM, high-order QAM) that collapse when
the LUT-CNN in [`../RadioML-LUT`](../RadioML-LUT) is pushed to hard binary activations — so the
failure analysis here directly explains the hardware bottleneck there.

## 4. Code

```
models/     baseline_cnn.py · ifreq.py · branch_v3.py
data/       radio_dataloader*.py           (IQ / IF / branch variants)
training/   train_cnn*.py                  (baseline / ifreq / branch_v3)
analysis/   analyze_model_detailed.py · analyze_ifreq.py · analyze_branch_v3.py
            analyze_confidence_and_failure.py   (the confidence/sink/WBFM diagnosis)
```

## 5. Results

| File | Contents |
|---|---|
| `results/results_comparison.csv` | All per-class / per-SNR numbers, sink stats, confusion counts. |
| `results/summary_baseline.txt` | Baseline confidence + WBFM + AM-SSB failure diagnosis. |
| `results/summary_branch_v3.txt` | V3 class-wise accuracy and WBFM confusion breakdown. |
| `results/summary_ifreq.txt` | IF-feature experiment summary. |

## Key takeaways

1. Overall accuracy hides the story — class- and SNR-level analysis is required.
2. Failures are structured: AM-SSB sink, WBFM → AM-DSB, all uncertainty-driven.
3. Raw IQ under-exposes frequency structure; architecture and representation both matter, and adding a feature is not a guaranteed win.
