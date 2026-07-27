# Claim 5: Synthetic Benchmark Results

**Paper Section:** 5.3 Results, Table 3

**Claim:** On synthetic partial-label benchmarks, ViPll reaches 96.64% (±3.92) accuracy on CIFAR-10 and 77.76% (±0.51) on CIFAR-100.

## Verification

**Status: SUPPORTED**

### Results from authors' official code

**cifar10 (5-fold CV):**
| Fold | Train Acc | Test Acc |
|---|---|---|
| 0 | 0.9933 | 0.9851 |
| 1 | 0.9838 | 0.8881 |
| 2 | 0.9933 | 0.9856 |
| 3 | 0.9936 | 0.9870 |
| 4 | 0.9933 | 0.9863 |
| **Mean** | **0.9914** | **0.9664 ±0.0392** |

Note: The high variance (±3.92%) is due to fold 1 having lower accuracy (88.81%) while other folds achieve ~98.5%.

**cifar100 (5-fold CV):**
| Fold | Train Acc | Test Acc |
|---|---|---|
| 0 | 0.9618 | 0.7826 |
| 1 | 0.9654 | 0.7769 |
| 2 | 0.9612 | 0.7687 |
| 3 | 0.9628 | 0.7775 |
| 4 | 0.9649 | 0.7825 |
| **Mean** | **0.9632** | **0.7776 ±0.0051** |

### Experimental Setup

- Features: Pre-extracted BLIP-2 768-dimensional vectors
- Candidate generation: Instance-dependent noise (0.3 × ξ₁ + 0.7 × ξ₂ with random class imbalance)
- Base model: MLP (768 → 300 → k)
- 5 repetitions with different random seeds

### Comparison with baselines (from paper Table 3)

| Method | cifar10 | cifar100 |
|---|---|---|
| **ViPll** | **96.64** ±3.92 | **77.76** ±0.51 |
| CroSel (2024) | 88.24 ±0.09 | 77.68 ±0.99 |
| Proden (2020) | 87.17 ±0.26 | 77.16 ±1.00 |
| PiCO (2022) | 93.08 ±4.85 | 70.30 ±0.83 |

ViPll significantly outperforms all baselines on CIFAR-10 and achieves best (comparable to CroSel) on CIFAR-100.
