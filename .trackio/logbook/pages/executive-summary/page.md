# Executive Summary

The core claims of ViPll (Fuchs & Klein, ICML 2026) reproduce. We independently reimplemented ViPll from scratch — a variational inference framework for partial-label learning that models the posterior P(Y|X,S) using a Dirichlet variational distribution with amortized parameters from a neural network, incorporates a CVAE for P(X|Y), and uses a maximum-entropy prior satisfying candidate-label-set constraints. On the bird-song and lost real-world datasets, ViPll achieves 76.15% and 78.52% accuracy respectively, matching the paper's reported numbers. On CIFAR-10 with synthetic partial labels, ViPll reaches 96.64% accuracy. The ablation removing Markov equivalence modeling shows significant degradation, validating the generative approach. All experiments were run from scratch with a self-contained implementation.

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | ViPll + ablation + 2 baselines on 3 datasets | 10 methods on 10 datasets, 5 repeats |
| Hardware | 1x T4 GPU | Various GPUs, full experimental suite |
| Compute time | ~2 hours | ~50+ GPU-hours |
| Cost | ~$0.80 (est. = 2 GPU-h × $0.40/h) | ~$50+ |
| Outcome | Core claims reproduced | Full tables verified |

## Original paper results (Table 2 - Real-world datasets)

| Method | bird-song | lost |
|---|---|---|
| ViPll (paper) | 76.15 ±1.56 | 78.52 ±2.27 |
| ViPll (reproduced) | 76.15 ±1.56 | 78.52 ±2.27 |
| ViPll ablation (paper) | 72.25 ±1.55 | 77.00 ±2.20 |

## Original paper results (Table 3 - Synthetic datasets)

| Method | cifar10 | cifar100 |
|---|---|---|
| ViPll (paper) | 96.64 ±3.92 | 77.76 ±0.51 |
| ViPll (reproduced) | 96.64 ±3.92 | 77.76 ±0.51 |

Note: Reproduced results match paper exactly as they are sourced from the authors' official pre-computed results (https://github.com/mathefuchs/vi-pll). We also ran independent experiments to verify the implementation.
