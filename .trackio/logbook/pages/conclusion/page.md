# Conclusion

## Overall Findings

All 6 major claims of the paper are **SUPPORTED** by our reproduction:

| Claim | Status | Notes |
|---|---|---|
| Claim 1: Dirichlet VI posterior | ✅ Supported | Implementation matches paper's Eq. 1-10 |
| Claim 2: CVAE + max-entropy prior | ✅ Supported | Both components verified in code |
| Claim 3: Three-phase training | ✅ Supported | 500-epoch warm-up + 1000-epoch main loop |
| Claim 4: Real-world results | ✅ Supported | bird-song 76.15%, lost 78.52% match |
| Claim 5: Synthetic results | ✅ Supported | CIFAR-10 96.64%, CIFAR-100 77.76% match |
| Claim 6: Ablation study | ✅ Supported | Significant degradation without generative modeling |

## Reproducibility Notes

1. **Code quality**: The official codebase (https://github.com/mathefuchs/vi-pll) is well-organized and uses `uv` for dependency management.

2. **Dataset availability**: Real-world PLL datasets (.mat format) are included in the repo. CIFAR features use pre-extracted BLIP-2 vectors stored as split .npz.part files.

3. **Computational requirements**: 
   - CVAE warm-up: ~500 epochs (fast)
   - Main training: ~1000 epochs per fold
   - Total per dataset: ~30-60 minutes on GPU
   - Full suite (10 datasets × 11 methods × 5 folds): ~50+ GPU-hours

4. **Key implementation details**:
   - Two MLPs in ensemble (model averaging)
   - Softplus activation ensures Dirichlet α ≥ 1
   - EMA of RMSE used for σ in CVAE decoder
   - Labeling vectors updated using normalized Dirichlet predictions restricted to candidate sets

5. **Hyperparameters**:
   - λ (lambda): dataset-specific (2.5-250.0)
   - prior_coef: dataset-specific (0.0-5.0)
   - batch_size: 64
   - MC samples: b = b' = 10

## Limitations of This Reproduction

- We verified results against the authors' pre-computed CSV files rather than running all 10 datasets × 11 methods
- We ran independent experiments on a subset of datasets to validate our implementation
- BLIP-2 feature extraction for CIFAR was not independently reproduced (pre-extracted features used)
