# Claim 1: Dirichlet Variational Distribution Posterior

**Paper Section:** 4.1 Optimization Objective

**Claim:** ViPll models the posterior P(Y|X,S) over true labels using a Dirichlet variational distribution whose parameters are output by an amortized inference network, factorizing the generative process as P(Y)P(X|Y)P(S|X,Y) instead of the standard discriminative formulation P(Y|X)P(S|X,Y).

## Verification

**Status: SUPPORTED**

The implementation in `reproduce_vipll.py` directly implements this:

1. **Dirichlet variational distribution** (Eq. 1): `q(y|x,s) = Dir(y; α)` where `α = f_φ(x,s) + 1`, with `f_φ` being a neural network.

2. **Generative factorization** (Eq. 4): The model uses `P(X,Y,S) = P(Y)P(X|Y)P(S|X,Y)` instead of the standard `P(Y|X)P(S|X,Y)`.

3. **Amortized inference**: The variational parameters α are predicted by an MLP (two MLPs in ensemble), not optimized per data point.

Key code from our implementation:
```python
# Dirichlet variational distribution (Eq. 1)
alpha = 1 + self.logits_to_evidence(logits)  # softplus ensures α ≥ 1
dirichlet_dist = torch.distributions.Dirichlet(alpha)
q_y_x_sample = dirichlet_dist.rsample((10,))  # MC samples

# Generative factorization (Eq. 4)
# log P(Y) - prior term
log_p_y = torch.mean(torch.sum(q_y_x_sample * torch.log(prior + 1e-10), dim=-1), dim=0)
# log P(X|Y) - CVAE term  
log_p_x_y = self._compute_log_p_x_given_y(...)
# log P(S|X,Y) - candidate set term
log_p_s_xy = torch.mean(torch.sum(q_y_x_sample * torch.log(curr_yb + 1e-10), dim=-1), dim=0)
```

The paper's Proposition 4.3 proves the two factorizations are Markov equivalent, but the generative form P(Y)P(X|Y)P(S|X,Y) allows explicit modeling of P(X|Y) via CVAE and P(Y) via max-entropy prior, which the discriminative form does not support.
