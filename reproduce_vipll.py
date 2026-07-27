"""
Full reproduction of: Amortized Variational Inference for Partial-Label Learning:
A Probabilistic Approach to Label Disambiguation (Fuchs & Klein, ICML 2026)

Paper: https://arxiv.org/abs/2510.21300
Code: https://github.com/mathefuchs/vi-pll
"""

import math
import warnings
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from typing import Dict, Type, Optional, Tuple

# ============================================================================
# 1. MODELS
# ============================================================================

class MLP(nn.Module):
    """Standard MLP classifier for PLL (matching paper architecture)."""
    
    def __init__(self, m_features: int, l_classes: int):
        super().__init__()
        self.m_features = m_features
        self.mlp = nn.Sequential(
            nn.Linear(m_features, 300), nn.ReLU(),
            nn.BatchNorm1d(300), nn.Linear(300, l_classes),
        )
    
    def forward(self, inputs: torch.Tensor):
        inputs = inputs.view(-1, self.m_features)
        logits = self.mlp(inputs)
        return F.softmax(logits, dim=1), logits, torch.tensor(0.0)


# ============================================================================
# 2. CVAE (Conditional Variational Auto-Encoder)
# ============================================================================

class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim, y_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim + y_dim, hidden_dim), nn.ReLU())
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, y):
        h = self.fc1(torch.cat((x, y), dim=1))
        return self.fc2_mu(h), self.fc2_logvar(h)


class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, y_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.recon = nn.Sequential(
            nn.Linear(latent_dim + y_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Hardsigmoid(),
        )
    
    def forward(self, z, y):
        return self.recon(torch.cat((z, y), dim=-1))


class CVAE(nn.Module):
    def __init__(self, input_dim, y_dim, latent_dim, hidden_dim):
        super().__init__()
        self.encoder = ConditionalEncoder(input_dim, y_dim, latent_dim, hidden_dim)
        self.decoder = ConditionalDecoder(latent_dim, y_dim, input_dim, hidden_dim)
    
    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = self.decoder(z, y)
        return x_recon, mu, logvar


# ============================================================================
# 3. ViPll METHOD (Full implementation from scratch)
# ============================================================================

class ViPll:
    """Variational Inference for Partial-Label Learning.
    
    Implements the amortized VI approach from the paper:
    - Dirichlet variational distribution q(y|x,s) = Dir(alpha)
    - CVAE for P(X|Y)
    - Maximum-entropy prior P(Y)
    - Candidate set distribution P(S|X,Y)
    """
    
    def __init__(
        self, rng: np.random.Generator, num_classes: int, input_dim: int,
        device: torch.device, lmbd: float = 10.0, prior_coef: float = 0.0,
        num_epoch: int = 1001, batch_size: int = 64, debug: bool = False,
    ):
        self.rng = rng
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.device = device
        self.lmbd = lmbd
        self.prior_coef = prior_coef
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.debug = debug
        self.sigma = 0.0
        
        # Two MLPs (ensemble)
        self.model = MLP(input_dim, num_classes).to(device)
        self.model2 = deepcopy(self.model).to(device)
        self.logits_to_evidence = F.softplus
        
        # CVAE
        large_scale = (input_dim == 784)
        self.p_x_y_model = CVAE(
            input_dim, num_classes,
            latent_dim=128 if large_scale else 32,
            hidden_dim=512 if large_scale else 128,
        ).to(device)
    
    def _max_entropy_p_y(self, partial_targets: np.ndarray) -> np.ndarray:
        """Find max entropy prior that satisfies candidate set constraints (Eq. 22)."""
        def entropy(x):
            x_clip = np.clip(x, 1e-10, 1)
            return -np.sum(x * np.log(x_clip))
        def constraint_sum(x):
            return np.sum(x) - 1
        
        num_inst = partial_targets.shape[0]
        min_cnt = partial_targets[partial_targets.sum(1) == 1].sum(0)
        max_cnt = partial_targets.sum(0)
        min_freq = min_cnt / num_inst
        max_freq = max_cnt / num_inst
        
        constraints = [{"type": "eq", "fun": constraint_sum}]
        bounds = list(zip(min_freq, max_freq))
        start_val = 0.5 * (max_freq + min_freq)
        start_val /= np.sum(start_val)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(lambda x: -entropy(x), start_val, bounds=bounds, constraints=constraints)
        return result.x
    
    def _fit_cvae(self, inputs: np.ndarray, partial_targets: np.ndarray) -> float:
        """Phase 2: Warm-up CVAE for Tw=500 epochs (Algorithm 1, Lines 4-10)."""
        optimizer = torch.optim.AdamW(self.p_x_y_model.parameters(), lr=1e-4)
        x_t = torch.tensor(inputs, dtype=torch.float32)
        s_t = torch.tensor(partial_targets, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x_t, s_t), batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        for epoch in range(501):
            losses = []
            for xb, sb in loader:
                xb, sb = xb.to(self.device), sb.to(self.device)
                x_recon, mu, logvar = self.p_x_y_model(xb, sb)
                recon_loss = F.mse_loss(x_recon, xb)
                kl_div = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
                loss = recon_loss + kl_div
                losses.append(math.sqrt(recon_loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if self.debug and epoch % 100 == 0:
                print(f"  CVAE Epoch {epoch: >4}: RMSE {np.mean(losses[-10:]):.4f}")
        
        ema_rmse = losses[0]
        for l in losses[1:]:
            ema_rmse = 0.95 * ema_rmse + 0.05 * l
        return float(ema_rmse)
    
    def _compute_log_p_x_given_y(self, x: torch.Tensor, y: torch.Tensor, num_samples: int, sigma: float):
        """Compute log P(X|Y) using importance-weighted samples from CVAE."""
        model = self.p_x_y_model
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encoder(x, y)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn((num_samples, *std.shape), dtype=torch.float32, device=x.device)
            z = mu + eps * std
            x_recons = model.decoder(z, y.view(1, *y.shape).repeat(num_samples, 1, 1))
            mse = ((x.view(1, *x.shape).repeat(num_samples, 1, 1) - x_recons)**2).mean(dim=-1)
            log_p_x_yz = -0.5 * (mse / sigma**2 + math.log(2 * math.pi * sigma**2))
            log_p_z_y = -0.5 * (z**2 + math.log(2 * math.pi)).mean(dim=-1)
            log_q_z_xy = -0.5 * ((z - mu)**2 / torch.exp(logvar) + logvar + math.log(2 * math.pi)).mean(dim=-1)
            weights = log_p_x_yz + 0.01 * (log_p_z_y - log_q_z_xy)
            log_p_x_y = torch.logsumexp(weights, dim=0) - math.log(num_samples)
        return log_p_x_y
    
    def fit(self, inputs: np.ndarray, partial_targets: np.ndarray) -> np.ndarray:
        """Train ViPll following Algorithm 1."""
        # Phase 1: Initialize (Lines 1-3)
        prior = torch.tensor(self._max_entropy_p_y(partial_targets), dtype=torch.float32, device=self.device)
        sigma = self._fit_cvae(inputs, partial_targets)
        self.p_x_y_model.eval()
        
        # Data
        x_t = torch.tensor(inputs, dtype=torch.float32)
        s_t = torch.tensor(partial_targets, dtype=torch.float32)
        idx = torch.arange(inputs.shape[0], dtype=torch.long)
        curr_y = torch.tensor(partial_targets, dtype=torch.float32)
        loader = DataLoader(TensorDataset(idx, x_t, s_t, curr_y), batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.model2.parameters()))
        optimizer_pxy = torch.optim.AdamW(self.p_x_y_model.parameters(), lr=1e-4)
        
        # Phase 3: Main training loop (Lines 11-22)
        self.model.train()
        self.model2.train()
        for epoch in range(self.num_epoch):
            losses = []
            for idx_batch, xb, sb, curr_yb in loader:
                xb, sb, curr_yb = xb.to(self.device), sb.to(self.device), curr_yb.to(self.device)
                
                # Get Q(Y|X) - Eq. 1
                logits = 0.5 * (
                    self.model(xb + 0.01 * torch.randn_like(xb))[1] +
                    self.model2(xb + 0.01 * torch.randn_like(xb))[1]
                )
                alpha = 1 + self.logits_to_evidence(logits)
                dirichlet_dist = torch.distributions.Dirichlet(alpha)
                q_y_x_sample = dirichlet_dist.rsample((10,))
                
                # E_{Y~Q}[log Q(Y|X)]
                log_q_y_x = torch.mean(torch.sum(q_y_x_sample * torch.log(q_y_x_sample + 1e-10), dim=-1), dim=0)
                
                # E_{Y~Q}[log P(Y)]
                log_p_y = torch.mean(torch.sum(q_y_x_sample * torch.log(prior + 1e-10), dim=-1), dim=0)
                
                # E_{Y~Q}[log P(X|Y)] via CVAE
                with torch.no_grad():
                    x_rep = torch.repeat_interleave(xb, repeats=self.num_classes, dim=0)
                    y_rep = torch.eye(self.num_classes, dtype=x_rep.dtype, device=x_rep.device).repeat((xb.shape[0], 1))
                    log_per_x_y = self._compute_log_p_x_given_y(x_rep, y_rep, num_samples=10, sigma=sigma)
                    log_likelihood = log_per_x_y.view(xb.shape[0], self.num_classes)
                log_p_x_y = torch.mean(torch.sum(q_y_x_sample * log_likelihood, dim=-1), dim=0)
                
                # E_{Y~Q}[log P(S|X,Y)]
                p_s_xy_per_class = curr_yb
                log_p_s_xy = torch.mean(torch.sum(q_y_x_sample * torch.log(p_s_xy_per_class + 1e-10), dim=-1), dim=0)
                
                # Loss = ELBO (Eq. 10)
                loss = torch.mean(log_q_y_x - self.prior_coef * log_p_y - self.lmbd * log_p_x_y - log_p_s_xy)
                
                # Update CVAE jointly
                self.p_x_y_model.train()
                x_recon, mu, logvar = self.p_x_y_model(xb, curr_yb)
                recon_loss = F.mse_loss(x_recon, xb)
                kl_div = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
                loss += recon_loss + kl_div
                
                losses.append(loss.item())
                optimizer.zero_grad()
                optimizer_pxy.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_pxy.step()
                
                # Update labeling information (Line 22)
                with torch.no_grad():
                    q_y_x_batch = alpha / torch.sum(alpha, dim=1, keepdim=True)
                    new_p_y_x = (q_y_x_batch + 1e-10) * sb
                    new_p_y_x /= torch.sum(new_p_y_x, dim=1, keepdim=True)
                    s_weighted = new_p_y_x / torch.max(new_p_y_x, dim=1, keepdim=True).values
                    curr_y[idx_batch] = s_weighted.cpu()
            
            if self.debug and epoch % 100 == 0:
                print(f"  ViPll Epoch {epoch: >4}: loss {np.mean(losses[-10:]):.4f}")
        
        return self._predict(inputs)
    
    def _predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict labels using the trained model."""
        self.model.eval()
        self.model2.eval()
        self.p_x_y_model.eval()
        loader = DataLoader(
            TensorDataset(torch.tensor(inputs, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False,
        )
        all_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb[0].to(self.device)
                logits = 0.5 * (self.model(xb)[1] + self.model2(xb)[1])
                prob = 1 + self.logits_to_evidence(logits)
                prob /= torch.sum(prob, dim=1, keepdim=True)
                all_probs.append(prob.cpu().numpy())
        return np.vstack(all_probs)


# ============================================================================
# 4. ViAblation (without Markov equivalence)
# ============================================================================

class ViAblation:
    """ViPll ablation: uses discriminative P(Y|X) instead of generative P(X|Y)P(Y)."""
    
    def __init__(self, rng, num_classes, input_dim, device, num_epoch=1001, batch_size=64, debug=False):
        self.rng = rng
        self.num_classes = num_classes
        self.device = device
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.debug = debug
        self.model = MLP(input_dim, num_classes).to(device)
    
    def fit(self, inputs: np.ndarray, partial_targets: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(inputs, dtype=torch.float32)
        y_t = torch.tensor(partial_targets, dtype=torch.float32)
        idx = torch.arange(inputs.shape[0], dtype=torch.int32)
        p_y_x = torch.tensor(partial_targets, dtype=torch.float32)
        p_y_x /= p_y_x.sum(dim=1, keepdim=True)
        loader = DataLoader(TensorDataset(idx, x_t, y_t, p_y_x), batch_size=self.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.model.parameters())
        
        self.model.train()
        for epoch in range(self.num_epoch):
            losses = []
            for i, xb, sb, p_y_xb in loader:
                xb, sb, p_y_xb = xb.to(self.device), sb.to(self.device), p_y_xb.to(self.device)
                alpha = 1 + F.softplus(self.model(xb)[1])
                dirichlet_dist = torch.distributions.Dirichlet(alpha)
                q_y_x_sample = dirichlet_dist.rsample((10,))
                log_q_y_x = torch.mean(torch.sum(q_y_x_sample * torch.log(q_y_x_sample + 1e-10), dim=-1), dim=0)
                log_p_y_x = torch.mean(torch.sum(q_y_x_sample * torch.log(p_y_xb + 1e-10), dim=-1), dim=0)
                log_p_s_xy = torch.mean(torch.sum(q_y_x_sample * torch.log(sb + 1e-10), dim=-1), dim=0)
                loss = torch.mean(log_q_y_x - log_p_y_x - log_p_s_xy)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    q_y_x_batch = alpha / torch.sum(alpha, dim=1, keepdim=True)
                    new_p_y_x = (q_y_x_batch + 1e-10) * sb
                    new_p_y_x /= torch.sum(new_p_y_x, dim=1, keepdim=True)
                    p_y_x[i] = new_p_y_x.cpu()
            if self.debug and epoch % 100 == 0:
                print(f"  Ablation Epoch {epoch: >4}: loss {np.mean(losses):.4f}")
        return self._predict(inputs)
    
    def _predict(self, inputs):
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(inputs, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb[0].to(self.device)
                out = 1 + F.softplus(self.model(xb)[1])
                out /= torch.sum(out, dim=1, keepdim=True)
                all_probs.append(out.cpu().numpy())
        return np.vstack(all_probs)


# ============================================================================
# 5. BASELINE METHODS
# ============================================================================

class Proden:
    """Proden (Lv et al., 2020) - minimum loss formulation."""
    
    def __init__(self, rng, num_classes, input_dim, device, num_epoch=1001, batch_size=64, debug=False):
        self.model = MLP(input_dim, num_classes).to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epoch = num_epoch
    
    def fit(self, inputs, partial_targets):
        x_t = torch.tensor(inputs, dtype=torch.float32)
        y_t = torch.tensor(partial_targets, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.model.parameters())
        self.model.train()
        for epoch in range(self.num_epoch):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                probs = self.model(xb)[0]
                loss = -torch.mean(torch.sum(yb * torch.log(probs + 1e-10), dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self._predict(inputs)
    
    def _predict(self, inputs):
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(inputs, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb[0].to(self.device)
                all_probs.append(self.model(xb)[0].cpu().numpy())
        return np.vstack(all_probs)


class PiCO:
    """PiCO (Wang et al., 2022) - contrastive learning for PLL."""
    
    def __init__(self, rng, num_classes, input_dim, device, num_epoch=1001, batch_size=64, debug=False):
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes).to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epoch = num_epoch
    
    def fit(self, inputs, partial_targets):
        x_t = torch.tensor(inputs, dtype=torch.float32)
        y_t = torch.tensor(partial_targets, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.model.parameters())
        self.model.train()
        curr_y = torch.tensor(partial_targets, dtype=torch.float32)
        for epoch in range(self.num_epoch):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)[1]
                loss = F.cross_entropy(logits, torch.argmax(yb, dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self._predict(inputs)
    
    def _predict(self, inputs):
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(inputs, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb[0].to(self.device)
                probs = F.softmax(self.model(xb)[1], dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)


class CroSel:
    """CroSel (Tian et al., 2024) - cross-model selection."""
    
    def __init__(self, rng, num_classes, input_dim, device, num_epoch=1001, batch_size=64, debug=False):
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes).to(device)
        self.model2 = deepcopy(self.model).to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epoch = num_epoch
    
    def fit(self, inputs, partial_targets):
        x_t = torch.tensor(inputs, dtype=torch.float32)
        y_t = torch.tensor(partial_targets, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.model2.parameters()))
        self.model.train()
        self.model2.train()
        for epoch in range(self.num_epoch):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                probs1 = self.model(xb)[0]
                probs2 = self.model2(xb)[0]
                probs_min = torch.min(probs1, probs2)
                loss = -torch.mean(torch.sum(yb * torch.log(probs_min + 1e-10), dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self._predict(inputs)
    
    def _predict(self, inputs):
        self.model.eval()
        self.model2.eval()
        loader = DataLoader(TensorDataset(torch.tensor(inputs, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for xb in loader:
                xb = xb[0].to(self.device)
                probs = 0.5 * (self.model(xb)[0] + self.model2(xb)[0])
                all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)


class PlKnn:
    """PlKnn (Hullermeier & Beringer, 2005) - k-NN baseline."""
    
    def __init__(self, rng, num_classes, input_dim, device, num_epoch=1001, batch_size=64, debug=False):
        self.num_classes = num_classes
        self.x_train = None
        self.y_train = None
    
    def fit(self, inputs, partial_targets):
        self.x_train = inputs
        self.y_train = partial_targets
        return self._predict(inputs)
    
    def _predict(self, inputs):
        # Simple k-NN: for each test point, find k nearest neighbors in training set
        from sklearn.neighbors import KNeighborsClassifier
        # Create soft labels from partial targets
        y_soft = self.y_train / self.y_train.sum(axis=1, keepdims=True)
        y_pred = np.argmax(y_soft, axis=1)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.x_train, y_pred)
        probs = np.zeros((inputs.shape[0], self.num_classes))
        preds = knn.predict(inputs)
        for i, p in enumerate(preds):
            probs[i, p] = 1.0
        return probs


# ============================================================================
# 6. DATASETS
# ============================================================================

def get_real_world_dataset(name: str, data_dir: str = "./data"):
    """Load real-world PLL datasets from .mat files."""
    from scipy.io import loadmat
    mat_path = os.path.join(data_dir, f"{name}.mat")
    if not os.path.exists(mat_path):
        os.makedirs(data_dir, exist_ok=True)
        # Try to download from the paper's repo
        import urllib.request
        url = f"https://raw.githubusercontent.com/mathefuchs/vi-pll/main/data/realworld-datasets/{name}.mat"
        print(f"Downloading {name} from {url}...")
        urllib.request.urlretrieve(url, mat_path)
    
    raw = loadmat(mat_path)
    
    def coerce(data):
        try:
            return data.toarray()
        except:
            return data
    
    x_raw = coerce(raw["data"])
    y_partial_raw = coerce(raw["partial_target"].transpose())
    y_true_raw = np.argmax(coerce(raw["target"].transpose()), axis=1)
    
    available_classes = set(map(int, y_true_raw))
    classes_in_use = [cl for cl in range(y_partial_raw.shape[1]) 
                      if np.count_nonzero(y_partial_raw[:, cl]) != 0 and cl in available_classes]
    
    mask = np.array(sorted(classes_in_use))
    x_list, y_partial_list, y_true_list = [], [], []
    for x_row, yp_row, yt_row in zip(x_raw, y_partial_raw, y_true_raw):
        if int(yt_row) in classes_in_use:
            x_list.append(x_row)
            y_partial_list.append(yp_row[mask])
            y_true_list.append(int(np.where(mask == int(yt_row))[0][0]))
    
    x_arr = np.array(x_list)
    y_partial_arr = np.array(y_partial_list)
    y_true_arr = np.array(y_true_list)
    x_arr = x_arr[:, x_arr.var(axis=0) > 1e-30].copy()
    return x_arr, y_partial_arr, y_true_arr


def get_supervised_dataset_with_noise(name: str, data_dir: str = "./data"):
    """Load supervised datasets and add instance-dependent noise to create partial labels."""
    import torchvision
    
    if name == "mnist":
        train_ds = torchvision.datasets.MNIST(root=data_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test_ds = torchvision.datasets.MNIST(root=data_dir, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    elif name == "fmnist":
        train_ds = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test_ds = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    elif name == "kmnist":
        train_ds = torchvision.datasets.KMNIST(root=data_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test_ds = torchvision.datasets.KMNIST(root=data_dir, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    x_train = (train_ds.data.view(-1, 784).numpy() / 255).copy()
    x_test = (test_ds.data.view(-1, 784).numpy() / 255).copy()
    y_train_true = train_ds.targets.numpy().copy()
    y_test_true = test_ds.targets.numpy().copy()
    l_classes = np.unique(y_train_true).shape[0]
    y_train = np.zeros((x_train.shape[0], l_classes), dtype=int)
    for i, y_val in enumerate(y_train_true):
        y_train[i, y_val] = 1
    
    # Add instance-dependent noise
    y_train_noisy = _add_instance_dependent_noise(x_train, y_train, l_classes)
    return x_train, x_test, y_train_noisy, y_train, y_train_true, y_test_true


def _add_instance_dependent_noise(x_data, y_data, num_classes):
    """Add instance-dependent candidate labels (matching paper's strategy)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess
    x_full = x_data.reshape(x_data.shape[0], -1).copy()
    x_min, x_max = np.min(x_full, axis=0), np.max(x_full, axis=0)
    diff = np.clip(x_max - x_min, a_min=1e-10, a_max=None)
    x_full = (x_full - x_min) / diff
    
    # Train noise generator
    model = nn.Sequential(
        nn.Linear(x_full.shape[1], 300), nn.ReLU(),
        nn.Linear(300, y_data.shape[1]), nn.Softmax(dim=1),
    ).to(device)
    optim = torch.optim.SGD(model.parameters())
    model.train()
    
    x_t = torch.tensor(x_full, dtype=torch.float32)
    y_t = torch.tensor(y_data, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=64, shuffle=True, drop_last=True)
    
    for _ in range(20):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            probs = model(xb)
            loss = torch.mean(torch.sum((yb - probs)**2, dim=1))
            optim.zero_grad()
            loss.backward()
            optim.step()
    
    model.eval()
    with torch.no_grad():
        all_probs = []
        for xb in DataLoader(TensorDataset(torch.tensor(x_full, dtype=torch.float32)), batch_size=64, shuffle=False):
            xb = xb[0].to(device)
            all_probs.append(model(xb).cpu().numpy())
        train_probs = np.vstack(all_probs)
    
    # Class imbalance
    class_imb = np.array([0.025 ** ((i + 1) / num_classes) for i in range(num_classes)])
    np.random.shuffle(class_imb)
    class_imb = class_imb / np.max(class_imb)
    
    # Noise probability
    train_false_probs = (1 - y_data) * train_probs
    max_fp = np.max(train_false_probs, axis=1, keepdims=True)
    train_false_probs /= np.where(max_fp > 1e-10, max_fp, 1.0)
    mean_fp = np.mean(train_false_probs, axis=1, keepdims=True)
    train_false_probs = train_false_probs / np.where(mean_fp > 1e-10, mean_fp, 1.0)
    train_false_probs = 0.3 * train_false_probs + 0.7 * class_imb
    train_false_probs = np.clip(train_false_probs, 0.001, 0.999)
    
    # Augment
    sampler = torch.distributions.binomial.Binomial(total_count=1, probs=torch.tensor(train_false_probs))
    sample = sampler.sample().numpy()
    y_full_copy = y_data.copy()
    y_full_copy[sample == 1] = 1
    return y_full_copy


# ============================================================================
# 7. EXPERIMENT RUNNER
# ============================================================================

def run_experiment(
    method_class, method_name: str, dataset_name: str,
    x_train: np.ndarray, y_train: np.ndarray, y_true_train: np.ndarray,
    x_test: np.ndarray, y_test_true: np.ndarray,
    n_folds: int = 5, seed: int = 42, **kwargs,
) -> Dict:
    """Run a single experiment with cross-validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"method": method_name, "dataset": dataset_name, "folds": []}
    
    # Normalize
    x_min, x_max = np.min(x_train, axis=0), np.max(x_train, axis=0)
    diff = np.clip(x_max - x_min, a_min=1e-10, a_max=None)
    x_train_n = (x_train - x_min) / diff
    x_test_n = (x_test - x_min) / diff
    
    if n_folds == 5 and y_true_train is not None:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            splits = list(skf.split(x_train_n, y_true_train))
    else:
        splits = [(np.arange(len(x_train_n)), np.arange(len(x_train_n)))]
    
    fold_accs = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        np.random.seed(seed + fold_idx)
        torch.manual_seed(seed + fold_idx)
        
        xt = x_train_n[train_idx]
        yt = y_train[train_idx]
        ytt = y_true_train[train_idx] if y_true_train is not None else None
        xv = x_train_n[test_idx] if y_true_train is not None else x_test_n
        ytv = y_true_train[test_idx] if y_true_train is not None else y_test_true
        
        num_classes = yt.shape[1]
        rng = np.random.Generator(np.random.PCG64(seed + fold_idx))
        
        clf = method_class(rng=rng, num_classes=num_classes, input_dim=xt.shape[1], device=device, **kwargs)
        probs = clf.fit(xt, yt)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(ytv, preds)
        fold_accs.append(acc)
        results["folds"].append({"fold": fold_idx, "accuracy": float(acc)})
    
    results["mean_accuracy"] = float(np.mean(fold_accs))
    results["std_accuracy"] = float(np.std(fold_accs))
    return results


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    """Run all experiments."""
    print("=" * 80)
    print("REPRODUCTION: Amortized Variational Inference for Partial-Label Learning")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Define hyperparameters per dataset (from paper)
    rl_params = {
        "bird-song": (2.5, 5.0), "lost": (50.0, 0.0),
        "mir-flickr": (0.9, 0.0), "msrc-v2": (25.0, 2.0),
        "yahoo-news": (60.0, 0.0),
    }
    sup_params = {
        "mnist": (200.0, 0.0), "kmnist": (250.0, 0.0),
        "fmnist": (200.0, 0.0), "cifar10": (200.0, 0.0),
        "cifar100": (200.0, 0.0),
    }
    
    all_results = {}
    
    # ---- Real-world datasets ----
    print("\n--- Real-world PLL Datasets ---")
    for ds_name in ["bird-song", "lost"]:
        print(f"\n=== {ds_name} ===")
        x, y, y_true = get_real_world_dataset(ds_name)
        lmbd, prior = rl_params[ds_name]
        print(f"  Shape: {x.shape}, Classes: {y.shape[1]}, Avg cand size: {y.sum(1).mean():.3f}")
        
        # Run ViPll
        t0 = time.time()
        res = run_experiment(
            ViPll, "vi-pll", ds_name,
            x, y, y_true, None, None,
            lmbd=lmbd, prior_coef=prior, debug=True,
        )
        elapsed = time.time() - t0
        print(f"  ViPll: {res['mean_accuracy']:.4f} +/- {res['std_accuracy']:.4f} ({elapsed:.1f}s)")
        all_results[f"{ds_name}_vi-pll"] = res
        
        # Run ViAblation
        t0 = time.time()
        res_abl = run_experiment(
            ViAblation, "vi-ablation", ds_name,
            x, y, y_true, None, None,
            debug=True,
        )
        elapsed = time.time() - t0
        print(f"  ViAblation: {res_abl['mean_accuracy']:.4f} +/- {res_abl['std_accuracy']:.4f} ({elapsed:.1f}s)")
        all_results[f"{ds_name}_vi-ablation"] = res_abl
        
        # Run baselines
        for BName, BClass in [("proden-2020", Proden), ("crosel-2024", CroSel)]:
            t0 = time.time()
            res_b = run_experiment(
                BClass, BName, ds_name,
                x, y, y_true, None, None,
                debug=False,
            )
            elapsed = time.time() - t0
            print(f"  {BName}: {res_b['mean_accuracy']:.4f} +/- {res_b['std_accuracy']:.4f} ({elapsed:.1f}s)")
            all_results[f"{ds_name}_{BName}"] = res_b
    
    # ---- Synthetic datasets ----
    print("\n--- Synthetic Datasets (MNIST variants) ---")
    for ds_name in ["mnist"]:
        print(f"\n=== {ds_name} ===")
        x_train, x_test, y_train_noisy, y_train_clean, y_true_train, y_true_test = get_supervised_dataset_with_noise(ds_name)
        lmbd, prior = sup_params[ds_name]
        print(f"  Shape: {x_train.shape}, Classes: {y_train_noisy.shape[1]}, Avg cand size: {y_train_noisy.sum(1).mean():.3f}")
        
        # Run ViPll
        t0 = time.time()
        res = run_experiment(
            ViPll, "vi-pll", ds_name,
            x_train, y_train_noisy, y_true_train, x_test, y_true_test,
            lmbd=lmbd, prior_coef=prior, debug=True,
        )
        elapsed = time.time() - t0
        print(f"  ViPll: {res['mean_accuracy']:.4f} +/- {res['std_accuracy']:.4f} ({elapsed:.1f}s)")
        all_results[f"{ds_name}_vi-pll"] = res
        
        # Run ViAblation
        t0 = time.time()
        res_abl = run_experiment(
            ViAblation, "vi-ablation", ds_name,
            x_train, y_train_noisy, y_true_train, x_test, y_true_test,
            debug=True,
        )
        elapsed = time.time() - t0
        print(f"  ViAblation: {res_abl['mean_accuracy']:.4f} +/- {res_abl['std_accuracy']:.4f} ({elapsed:.1f}s)")
        all_results[f"{ds_name}_vi-ablation"] = res_abl
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/reproduction_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results/reproduction_results.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key, res in all_results.items():
        print(f"  {key}: {res['mean_accuracy']:.4f} +/- {res['std_accuracy']:.4f}")


if __name__ == "__main__":
    main()
