"""
HF Jobs experiment runner for ViPll reproduction.
Usage: hf jobs run --flavor t4-small --timeout 60m python:3.12 uv run repro_vipll_hf.py
"""

# /// script
# dependencies = [
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
#   "numpy",
#   "scipy",
#   "scikit-learn",
#   "huggingface_hub",
# ]
# ///

import math
import warnings
import os
import json
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy

# ============================================================================
# MODELS
# ============================================================================

class MLP(nn.Module):
    def __init__(self, m_features, l_classes):
        super().__init__()
        self.m_features = m_features
        self.mlp = nn.Sequential(
            nn.Linear(m_features, 300), nn.ReLU(),
            nn.BatchNorm1d(300), nn.Linear(300, l_classes),
        )
    
    def forward(self, inputs):
        inputs = inputs.view(-1, self.m_features)
        logits = self.mlp(inputs)
        return F.softmax(logits, dim=1), logits, torch.tensor(0.0)


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
# CIFAR-10 / CIFAR-100 with synthetic partial labels
# ============================================================================

def add_instance_dependent_noise(x_data, y_data, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_full = x_data.reshape(x_data.shape[0], -1).copy()
    x_min, x_max = np.min(x_full, axis=0), np.max(x_full, axis=0)
    diff = np.clip(x_max - x_min, a_min=1e-10, a_max=None)
    x_full = (x_full - x_min) / diff
    
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
    
    class_imb = np.array([0.025 ** ((i + 1) / num_classes) for i in range(num_classes)])
    np.random.shuffle(class_imb)
    class_imb = class_imb / np.max(class_imb)
    
    train_false_probs = (1 - y_data) * train_probs
    max_fp = np.max(train_false_probs, axis=1, keepdims=True)
    train_false_probs /= np.where(max_fp > 1e-10, max_fp, 1.0)
    mean_fp = np.mean(train_false_probs, axis=1, keepdims=True)
    train_false_probs = train_false_probs / np.where(mean_fp > 1e-10, mean_fp, 1.0)
    train_false_probs = 0.3 * train_false_probs + 0.7 * class_imb
    train_false_probs = np.clip(train_false_probs, 0.001, 0.999)
    
    sampler = torch.distributions.binomial.Binomial(total_count=1, probs=torch.tensor(train_false_probs))
    sample = sampler.sample().numpy()
    y_full_copy = y_data.copy()
    y_full_copy[sample == 1] = 1
    return y_full_copy


def get_mnist_with_noise(name):
    import torchvision
    if name == "mnist":
        train_ds = torchvision.datasets.MNIST(root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test_ds = torchvision.datasets.MNIST(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    elif name == "fmnist":
        train_ds = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test_ds = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    else:
        raise ValueError(name)
    
    x_train = (train_ds.data.view(-1, 784).numpy() / 255).copy()
    x_test = (test_ds.data.view(-1, 784).numpy() / 255).copy()
    y_train_true = train_ds.targets.numpy().copy()
    y_test_true = test_ds.targets.numpy().copy()
    l_classes = np.unique(y_train_true).shape[0]
    y_train = np.zeros((x_train.shape[0], l_classes), dtype=int)
    for i, y_val in enumerate(y_train_true):
        y_train[i, y_val] = 1
    y_train_noisy = add_instance_dependent_noise(x_train, y_train, l_classes)
    return x_train, x_test, y_train_noisy, y_train_true, y_test_true


# ============================================================================
# ViPll
# ============================================================================

class ViPll:
    def __init__(self, rng, num_classes, input_dim, device, lmbd=10.0, prior_coef=0.0, num_epoch=1001, batch_size=64, debug=False):
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
        
        self.model = MLP(input_dim, num_classes).to(device)
        self.model2 = deepcopy(self.model).to(device)
        self.logits_to_evidence = F.softplus
        
        large_scale = (input_dim == 784)
        self.p_x_y_model = CVAE(
            input_dim, num_classes,
            latent_dim=128 if large_scale else 32,
            hidden_dim=512 if large_scale else 128,
        ).to(device)
    
    def _max_entropy_p_y(self, partial_targets):
        def entropy(x):
            x_clip = np.clip(x, 1e-10, 1)
            return -np.sum(x * np.log(x_clip))
        def constraint_sum(x):
            return np.sum(x) - 1
        
        num_inst = partial_targets.shape[0]
        min_cnt = partial_targets[partial_targets.sum(1) == 1].sum(0)
        max_cnt = partial_targets.sum(0)
        bounds = list(zip(min_cnt / num_inst, max_cnt / num_inst))
        start_val = 0.5 * (max_cnt + min_cnt) / num_inst
        start_val /= np.sum(start_val)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(lambda x: -entropy(x), start_val, bounds=bounds, constraints=[{"type": "eq", "fun": constraint_sum}])
        return result.x
    
    def _fit_cvae(self, inputs, partial_targets):
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
                print(f"  CVAE Epoch {epoch}: RMSE {np.mean(losses[-10:]):.4f}", flush=True)
        
        ema_rmse = losses[0]
        for l in losses[1:]:
            ema_rmse = 0.95 * ema_rmse + 0.05 * l
        return float(ema_rmse)
    
    def _compute_log_p_x_given_y(self, x, y, num_samples, sigma):
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
    
    def fit(self, inputs, partial_targets):
        prior = torch.tensor(self._max_entropy_p_y(partial_targets), dtype=torch.float32, device=self.device)
        sigma = self._fit_cvae(inputs, partial_targets)
        self.p_x_y_model.eval()
        
        x_t = torch.tensor(inputs, dtype=torch.float32)
        s_t = torch.tensor(partial_targets, dtype=torch.float32)
        idx = torch.arange(inputs.shape[0], dtype=torch.long)
        curr_y = torch.tensor(partial_targets, dtype=torch.float32)
        loader = DataLoader(TensorDataset(idx, x_t, s_t, curr_y), batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.model2.parameters()))
        optimizer_pxy = torch.optim.AdamW(self.p_x_y_model.parameters(), lr=1e-4)
        
        self.model.train()
        self.model2.train()
        for epoch in range(self.num_epoch):
            losses = []
            for idx_batch, xb, sb, curr_yb in loader:
                xb, sb, curr_yb = xb.to(self.device), sb.to(self.device), curr_yb.to(self.device)
                
                logits = 0.5 * (
                    self.model(xb + 0.01 * torch.randn_like(xb))[1] +
                    self.model2(xb + 0.01 * torch.randn_like(xb))[1]
                )
                alpha = 1 + self.logits_to_evidence(logits)
                dirichlet_dist = torch.distributions.Dirichlet(alpha)
                q_y_x_sample = dirichlet_dist.rsample((10,))
                
                log_q_y_x = torch.mean(torch.sum(q_y_x_sample * torch.log(q_y_x_sample + 1e-10), dim=-1), dim=0)
                log_p_y = torch.mean(torch.sum(q_y_x_sample * torch.log(prior + 1e-10), dim=-1), dim=0)
                
                with torch.no_grad():
                    x_rep = torch.repeat_interleave(xb, repeats=self.num_classes, dim=0)
                    y_rep = torch.eye(self.num_classes, dtype=x_rep.dtype, device=x_rep.device).repeat((xb.shape[0], 1))
                    log_per_x_y = self._compute_log_p_x_given_y(x_rep, y_rep, num_samples=10, sigma=sigma)
                    log_likelihood = log_per_x_y.view(xb.shape[0], self.num_classes)
                log_p_x_y = torch.mean(torch.sum(q_y_x_sample * log_likelihood, dim=-1), dim=0)
                
                log_p_s_xy = torch.mean(torch.sum(q_y_x_sample * torch.log(sb + 1e-10), dim=-1), dim=0)
                
                loss = torch.mean(log_q_y_x - self.prior_coef * log_p_y - self.lmbd * log_p_x_y - log_p_s_xy)
                
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
                
                with torch.no_grad():
                    q_y_x_batch = alpha / torch.sum(alpha, dim=1, keepdim=True)
                    new_p_y_x = (q_y_x_batch + 1e-10) * sb
                    new_p_y_x /= torch.sum(new_p_y_x, dim=1, keepdim=True)
                    s_weighted = new_p_y_x / torch.max(new_p_y_x, dim=1, keepdim=True).values
                    curr_y[idx_batch] = s_weighted.cpu()
            
            if self.debug and epoch % 200 == 0:
                print(f"  Epoch {epoch}: loss {np.mean(losses[-10:]):.4f}", flush=True)
        
        return self._predict(inputs)
    
    def _predict(self, inputs):
        self.model.eval()
        self.model2.eval()
        loader = DataLoader(TensorDataset(torch.tensor(inputs, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)
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
# Vi Ablation
# ============================================================================

class ViAblation:
    def __init__(self, rng, num_classes, input_dim, device, num_epoch=1001, batch_size=64, debug=False):
        self.rng = rng
        self.num_classes = num_classes
        self.device = device
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.debug = debug
        self.model = MLP(input_dim, num_classes).to(device)
    
    def fit(self, inputs, partial_targets):
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
            if self.debug and epoch % 200 == 0:
                print(f"  Ablation Epoch {epoch}: loss {np.mean(losses):.4f}", flush=True)
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
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("VIPLL REPRODUCTION - HF JOB RUNNER")
    print("=" * 80, flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    method_name = sys.argv[2] if len(sys.argv) > 2 else "vipll"
    
    results = {"dataset": dataset_name, "method": method_name, "folds": []}
    
    if dataset_name in ["bird-song", "lost", "mir-flickr", "msrc-v2", "yahoo-news"]:
        # Real-world dataset
        print(f"\n--- Loading real-world dataset: {dataset_name} ---", flush=True)
        import urllib.request
        from scipy.io import loadmat
        mat_path = f"{dataset_name}.mat"
        url = f"https://raw.githubusercontent.com/mathefuchs/vi-pll/main/data/realworld-datasets/{dataset_name}.mat"
        print(f"Downloading {url}...", flush=True)
        urllib.request.urlretrieve(url, mat_path)
        
        raw = loadmat(mat_path)
        def coerce(data):
            try: return data.toarray()
            except: return data
        
        x_raw = coerce(raw["data"])
        y_partial_raw = coerce(raw["partial_target"].transpose())
        y_true_raw = np.argmax(coerce(raw["target"].transpose()), axis=1)
        
        available_classes = set(map(int, y_true_raw))
        classes_in_use = [cl for cl in range(y_partial_raw.shape[1]) if np.count_nonzero(y_partial_raw[:, cl]) != 0 and cl in available_classes]
        mask = np.array(sorted(classes_in_use))
        x_list, yp_list, yt_list = [], [], []
        for xr, yp, yt in zip(x_raw, y_partial_raw, y_true_raw):
            if int(yt) in classes_in_use:
                x_list.append(xr)
                yp_list.append(yp[mask])
                yt_list.append(int(np.where(mask == int(yt))[0][0]))
        x_arr = np.array(x_list)
        y_partial_arr = np.array(yp_list)
        y_true_arr = np.array(yt_list)
        x_arr = x_arr[:, x_arr.var(axis=0) > 1e-30].copy()
        
        params = {"bird-song": (2.5, 5.0), "lost": (50.0, 0.0), "mir-flickr": (0.9, 0.0), "msrc-v2": (25.0, 2.0), "yahoo-news": (60.0, 0.0)}
        lmbd, prior = params[dataset_name]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            splits = list(enumerate(skf.split(x_arr, y_true_arr)))
        
        for fold_idx, (train_idx, test_idx) in splits:
            x_train, x_test = x_arr[train_idx], x_arr[test_idx]
            y_train, y_test_true = y_partial_arr[train_idx], y_true_arr[test_idx]
            
            x_min, x_max = np.min(x_train, axis=0), np.max(x_train, axis=0)
            diff = np.clip(x_max - x_min, a_min=1e-10, a_max=None)
            x_train = (x_train - x_min) / diff
            x_test = (x_test - x_min) / diff
            
            rng = np.random.Generator(np.random.PCG64(42 + fold_idx))
            np.random.seed(42 + fold_idx)
            torch.manual_seed(42 + fold_idx)
            
            num_classes = y_train.shape[1]
            
            if method_name == "vipll":
                clf = ViPll(rng=rng, num_classes=num_classes, input_dim=x_train.shape[1], device=device, lmbd=lmbd, prior_coef=prior, debug=True)
            elif method_name == "vi-ablation":
                clf = ViAblation(rng=rng, num_classes=num_classes, input_dim=x_train.shape[1], device=device, debug=True)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            print(f"\nFold {fold_idx}/5 - Training {method_name}...", flush=True)
            t0 = time.time()
            probs = clf.fit(x_train, y_train)
            preds = clf._predict(x_test) if hasattr(clf, '_predict') else None
            if preds is None:
                preds = clf.predict(x_test)
            pred_labels = np.argmax(preds, axis=1)
            acc = accuracy_score(y_test_true, pred_labels)
            elapsed = time.time() - t0
            print(f"Fold {fold_idx}: acc={acc:.4f}, time={elapsed:.1f}s", flush=True)
            results["folds"].append({"fold": fold_idx, "accuracy": float(acc), "time_s": float(elapsed)})
    
    else:
        # Synthetic dataset (MNIST/FashionMNIST)
        print(f"\n--- Loading synthetic dataset: {dataset_name} ---", flush=True)
        x_train, x_test, y_train_noisy, y_train_true, y_test_true = get_mnist_with_noise(dataset_name)
        
        params = {"mnist": (200.0, 0.0), "fmnist": (200.0, 0.0)}
        lmbd, prior = params.get(dataset_name, (200.0, 0.0))
        
        num_classes = y_train_noisy.shape[1]
        
        x_min, x_max = np.min(x_train, axis=0), np.max(x_train, axis=0)
        diff = np.clip(x_max - x_min, a_min=1e-10, a_max=None)
        x_train = (x_train - x_min) / diff
        x_test = (x_test - x_min) / diff
        
        for fold_idx in range(1):  # Single fold for synthetic (no CV - uses train/test split)
            np.random.seed(fold_idx)
            torch.manual_seed(fold_idx)
            rng = np.random.Generator(np.random.PCG64(fold_idx))
            
            if method_name == "vipll":
                clf = ViPll(rng=rng, num_classes=num_classes, input_dim=x_train.shape[1], device=device, lmbd=lmbd, prior_coef=prior, debug=True)
            elif method_name == "vi-ablation":
                clf = ViAblation(rng=rng, num_classes=num_classes, input_dim=x_train.shape[1], device=device, debug=True)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            print(f"\nTraining {method_name} on {dataset_name}...", flush=True)
            t0 = time.time()
            probs = clf.fit(x_train, y_train_noisy)
            if hasattr(clf, '_predict'):
                preds = clf._predict(x_test)
            else:
                preds = clf.predict(x_test)
            pred_labels = np.argmax(preds, axis=1)
            acc = accuracy_score(y_test_true, pred_labels)
            elapsed = time.time() - t0
            print(f"Test acc: {acc:.4f}, time={elapsed:.1f}s", flush=True)
            results["folds"].append({"fold": fold_idx, "accuracy": float(acc), "time_s": float(elapsed)})
    
    # Summary
    accs = [f["accuracy"] for f in results["folds"]]
    times = [f["time_s"] for f in results["folds"]]
    print(f"\n{'='*50}", flush=True)
    print(f"RESULTS: {dataset_name} - {method_name}", flush=True)
    print(f"Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}", flush=True)
    print(f"Mean time per fold: {np.mean(times):.1f}s", flush=True)
    print(f"{'='*50}", flush=True)
    
    # Save results
    results["mean_accuracy"] = float(np.mean(accs))
    results["std_accuracy"] = float(np.std(accs))
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results.json", flush=True)


if __name__ == "__main__":
    main()