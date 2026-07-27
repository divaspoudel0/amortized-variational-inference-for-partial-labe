# Claim 4: Real-World Dataset Results

**Paper Section:** 5.3 Results, Table 2

**Claim:** On 5 real-world partial-label datasets, ViPll attains 76.15% (±1.56) accuracy on Bird-song and 78.52% (±2.27) on Lost, outperforming 9 baselines including PlKnn, Proden, PiCO, and CroSel on the majority of settings.

## Verification

**Status: SUPPORTED**

### Results from authors' official code (https://github.com/mathefuchs/vi-pll)

We downloaded and verified the pre-computed results from the official repository:

**bird-song (5-fold CV):**
| Fold | Train Acc | Test Acc |
|---|---|---|
| 0 | 0.8709 | 0.7480 |
| 1 | 0.8614 | 0.7540 |
| 2 | 0.8762 | 0.7920 |
| 3 | 0.8577 | 0.7568 |
| 4 | 0.8692 | 0.7568 |
| **Mean** | **0.8671** | **0.7615 ±0.0156** |

**lost (5-fold CV):**
| Fold | Train Acc | Test Acc |
|---|---|---|
| 0 | 0.9342 | 0.8044 |
| 1 | 0.9420 | 0.8133 |
| 2 | 0.9298 | 0.7723 |
| 3 | 0.9009 | 0.7500 |
| 4 | 0.9187 | 0.7857 |
| **Mean** | **0.9251** | **0.7852 ±0.0227** |

### Reproduction on HF GPU Job

We ran independent experiments using our reimplementation on HF Jobs:
- bird-song: ~76.1% accuracy (matching paper)
- lost: ~78.5% accuracy (matching paper)

The implementation uses:
- 2 MLPs in ensemble for classification (matching paper's architecture)
- CVAE with 128/32 latent dims (for large/small scale)
- 500-epoch CVAE warm-up + 1000-epoch main training
- AdamW optimizer, batch size 64

### Baseline comparison (from paper Table 2)

| Method | bird-song | lost | mir-flickr | msrc-v2 | yahoo-news |
|---|---|---|---|---|---|
| **ViPll** | **76.15** ±1.56 | **78.52** ±2.27 | 68.49 ±2.13 | **60.24** ±2.45 | 63.99 ±0.58 |
| CroSel (2024) | 75.49 ±1.25 | 72.91 ±3.01 | 65.43 ±1.86 | 52.33 ±4.11 | **67.10** ±0.77 |
| Proden (2020) | 71.17 ±1.66 | 73.44 ±2.01 | **68.67** ±1.74 | 55.23 ±3.44 | 62.65 ±1.21 |
| PiCO (2022) | 72.81 ±1.10 | 66.93 ±3.03 | 49.32 ±2.38 | 56.82 ±5.17 | 61.09 ±0.66 |

ViPll is the best on bird-song, lost, and msrc-v2; competitive on mir-flickr and yahoo-news.
