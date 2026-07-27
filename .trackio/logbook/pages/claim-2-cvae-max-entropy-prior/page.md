# Claim 2: CVAE + Maximum-Entropy Prior

**Paper Sections:** 4.2 (CVAE) and 4.3 (Prior)

**Claim:** ViPll incorporates a conditional variational autoencoder to model P(X|Y) (Section 4.2) and a maximum-entropy prior satisfying candidate-label-set constraints (Section 4.3).

## Verification

**Status: SUPPORTED**

### 2a. CVAE for P(X|Y)

The CVAE architecture consists of:
- **Encoder** r_γ(z|x,y): Gaussian with parameters μ_γ(x,y), Σ_γ(x,y), predicted by NN
- **Decoder** p_θ(x|y,z): Gaussian with mean μ_θ(y,z), fixed variance σ²
- **Prior** p(z|y) = N(0, I_m)

Key implementation:
```python
class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim, y_dim, latent_dim, hidden_dim=128):
        self.fc1 = nn.Sequential(nn.Linear(input_dim + y_dim, hidden_dim), nn.ReLU())
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, y):
        h = self.fc1(torch.cat((x, y), dim=1))
        return self.fc2_mu(h), self.fc2_logvar(h)

class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, y_dim, output_dim, hidden_dim=128):
        self.recon = nn.Sequential(
            nn.Linear(latent_dim + y_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Hardsigmoid(),
        )
```

The CVAE is warmed up for 500 epochs (Phase 2), and σ is estimated as the EMA of the RMSE reconstruction loss.

### 2b. Maximum-Entropy Prior

The prior P(Y) = Dir(y; α^π) is computed by solving the constrained optimization (Eq. 22):
- Maximize entropy H(π) subject to:
  - π_y ≥ fraction of instances where s_i = {y} (singleton sets)
  - π_y ≤ fraction of instances where y ∈ s_i (any candidate set)
  - Σ π_y = 1

```python
def _max_entropy_p_y(self, partial_targets):
    num_inst = partial_targets.shape[0]
    min_cnt = partial_targets[partial_targets.sum(1) == 1].sum(0)  # singletons
    max_cnt = partial_targets.sum(0)  # all appearances
    min_freq = min_cnt / num_inst
    max_freq = max_cnt / num_inst
    # Maximize entropy subject to bounds and sum=1 constraint
    result = minimize(lambda x: -entropy(x), start_val, bounds=bounds, constraints=constraints)
    return result.x
```

This ensures the prior respects the candidate label set constraints rather than being uniform.
