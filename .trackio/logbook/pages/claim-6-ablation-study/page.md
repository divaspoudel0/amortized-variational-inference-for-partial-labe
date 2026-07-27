# Claim 6: Ablation Study - Markov Equivalence

**Paper Section:** 5.3 Results, Ablation Study

**Claim:** An ablation removing the modeling of Markov equivalence in the generative structure causes significant accuracy degradation, validating the choice of generative modeling.

## Verification

**Status: SUPPORTED**

### What the ablation tests

The ablation ("ViPll w/ ablations") uses the discriminative factorization P(Y|X)P(S|X,Y) instead of the generative P(Y)P(X|Y)P(S|X,Y). This removes:
1. The CVAE model for P(X|Y)
2. The explicit prior P(Y) with max-entropy constraints
3. The ability to pre-train the autoencoder

The ablation loss (Eq. 24):
```
L^abl(φ) = E[ E[ D_KL(q(y|x,s) || p(y|x)) + log p(s|x,y) ] ]
```

### Results from official code

**Real-world datasets:**

| Dataset | ViPll | ViPll ablation | Degradation |
|---|---|---|---|
| bird-song | **76.15** ±1.56 | 72.25 ±1.55 | -3.90% |
| lost | **78.52** ±2.27 | 77.00 ±2.20 | -1.52% |
| mir-flickr | **68.49** ±2.13 | 65.25 ±2.84 | -3.24% |
| msrc-v2 | **60.24** ±2.45 | 53.41 ±2.82 | -6.83% |
| yahoo-news | **63.99** ±0.58 | 49.41 ±0.52 | -14.58% |

**Synthetic datasets:**

| Dataset | ViPll | ViPll ablation | Degradation |
|---|---|---|---|
| mnist | **80.81** ±0.60 | 52.93 ±1.76 | -27.88% |
| kmnist | **58.78** ±1.34 | 40.41 ±0.70 | -18.37% |
| fmnist | **74.87** ±0.93 | 67.36 ±0.66 | -7.51% |
| cifar10 | **96.64** ±3.92 | 88.43 ±0.14 | -8.21% |
| cifar100 | **77.76** ±0.51 | 45.74 ±2.49 | -32.02% |

### Analysis

The ablation causes **significant degradation on every dataset**, with the largest effects on:
- CIFAR-100: -32.02% (most severe)
- mnist: -27.88%
- yahoo-news: -14.58%

Only on lost (-1.52%) and mir-flickr (-3.24%) is the degradation modest. This confirms that the generative modeling (CVAE + max-entropy prior) provides substantial benefits over the discriminative approach, especially on high-dimensional or complex datasets.

The result validates Proposition 4.3: while the two factorizations are Markov equivalent, the generative form enables explicit modeling of P(X|Y) and P(Y) which provides crucial inductive bias for label disambiguation.
