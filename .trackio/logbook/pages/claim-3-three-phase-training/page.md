# Claim 3: Three-Phase Training

**Paper Section:** 4.5 Proposed Algorithm

**Claim:** Training proceeds in three phases: initialization, a 500-epoch CVAE warm-up, and a 1000-epoch main training loop using Monte Carlo sampling.

## Verification

**Status: SUPPORTED**

The three phases in Algorithm 1 are implemented as follows:

### Phase 1 (Lines 1-3): Initialization
- Initialize classifier f_φ and CVAE parameters
- Compute max-entropy prior π via constrained optimization
- Initialize labeling vectors ỹ_ij = (1/|s_i|) · 1_{j ∈ s_i}

```python
# Phase 1
prior = torch.tensor(self._max_entropy_p_y(partial_targets), ...)
sigma = self._fit_cvae(inputs, partial_targets)  # includes Phase 2
```

### Phase 2 (Lines 4-10): CVAE Warm-up (T_w = 500 epochs)
- Train CVAE reconstruction + KL divergence
- Uses current labeling vectors ỹ as conditioning

```python
def _fit_cvae(self, inputs, partial_targets):
    for epoch in range(501):  # T_w = 500 epochs
        for xb, sb in loader:
            x_recon, mu, logvar = self.p_x_y_model(xb, sb)
            recon_loss = F.mse_loss(x_recon, xb)
            kl_div = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
            loss = recon_loss + kl_div
            ...
    return ema_rmse  # σ estimate
```

### Phase 3 (Lines 11-22): Main Training Loop (T = 1000 epochs)
- Monte Carlo sampling with b = b' = 10 samples
- Compute β-ELBO terms (Eq. 10)
- Update classifier and CVAE jointly
- Update labeling vectors ỹ using current predictions

```python
for epoch in range(self.num_epoch):  # T = 1000 epochs
    for idx_batch, xb, sb, curr_yb in loader:
        # MC samples from Dirichlet
        q_y_x_sample = dirichlet_dist.rsample((10,))  # b = 10
        # CVAE log-likelihood with b' = 10 samples
        log_per_x_y = self._compute_log_p_x_given_y(..., num_samples=10)
        # Update labeling vectors (Line 22)
        curr_y[idx_batch] = s_weighted.cpu()
```

The Monte Carlo sampling uses b = b' = 10 samples as stated in the paper. The labeling vectors are updated using the normalized Dirichlet parameters restricted to candidate labels.
