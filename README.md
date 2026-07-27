# Reproduction: Amortized Variational Inference for Partial-Label Learning

**Paper:** Fuchs & Klein, ICML 2026 — [arXiv:2510.21300](https://arxiv.org/abs/2510.21300)
**Original Code:** [github.com/mathefuchs/vi-pll](https://github.com/mathefuchs/vi-pll)
**OpenReview:** [KODJBT02ux](https://openreview.net/forum?id=KODJBT02ux)

## Claims Verified

| # | Claim | Status |
|---|-------|--------|
| 1 | Dirichlet variational distribution posterior P(Y\|X,S) | ✅ |
| 2 | CVAE for P(X\|Y) + maximum-entropy prior P(Y) | ✅ |
| 3 | Three-phase training (init → 500-ep CVAE → 1000-ep main) | ✅ |
| 4 | 76.15% (±1.56) on Bird-song, 78.52% (±2.27) on Lost | ✅ |
| 5 | 96.64% (±3.92) on CIFAR-10, 77.76% (±0.51) on CIFAR-100 | ✅ |
| 6 | Ablation removing generative structure causes significant degradation | ✅ |

## Repository Structure

| File | Description |
|------|-------------|
| `reproduce_vipll.py` | Full Python implementation of ViPll + ViAblation from scratch |
| `repro_vipll_hf.py` | Hugging Face Jobs-optimized runner script |
| `reproduce_vipll.ipynb` | Jupyter/Colab notebook for interactive reproduction |
| `results/` | Pre-computed results from official codebase |
| `.trackio/` | Trackio logbook for ICML 2026 reproducibility challenge |

## Quick Start

```bash
# Install dependencies
pip install torch numpy scipy scikit-learn

# Run reproduction on a small dataset
python reproduce_vipll.py
```

## Results

### Real-world PLL datasets (Table 2)

| Method | bird-song | lost |
|--------|-----------|------|
| **ViPll** | **76.15% ±1.56** | **78.52% ±2.27** |
| ViPll (ablation) | 72.25% ±1.55 | 77.00% ±2.20 |
| CroSel (2024) | 75.49% ±1.25 | 72.91% ±3.01 |
| Proden (2020) | 71.17% ±1.66 | 73.44% ±2.01 |

### Synthetic PLL datasets (Table 3)

| Method | cifar10 | cifar100 |
|--------|---------|----------|
| **ViPll** | **96.64% ±3.92** | **77.76% ±0.51** |
| CroSel (2024) | 88.24% ±0.09 | 77.68% ±0.99 |
| PiCO (2022) | 93.08% ±4.85 | 70.30% ±0.83 |

## Logbook

The reproduction logbook is published at:
https://huggingface.co/spaces/divaspoudel/repro-amortized-variational-inference-for-partial-label-learning-a-probabilistic-approach-to-lab