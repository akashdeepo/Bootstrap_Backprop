"""
Training: pinball (quantile) loss against single root draws.

The pinball loss averaged over levels is a discretized CRPS -- a proper
scoring rule. Its population minimizer at each level tau is the true
tau-quantile of the conditional law of t given the input, so the network
is pushed toward the posterior-predictive root distribution without ever
constructing a Monte Carlo target.

Model selection is on a held-out validation split (disjoint parameter
draws), fixing v1's selection-on-training-loss flaw.

Targets are trained in doubly standardized space:
    t_std = t / s                (per-dataset scale, see datagen.featurize)
    t_train = t_std / target_scale   (one global constant, computed from the
                                      training split, for numerical conditioning)
target_scale is data-driven -- no knowledge of convergence rates is baked in.
"""

import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .model import QuantileNet


def pinball_loss(q: torch.Tensor, t: torch.Tensor,
                 levels: torch.Tensor) -> torch.Tensor:
    """
    q      (B, L) predicted quantiles
    t      (B, 1) single observed draws
    levels (L,)   quantile levels
    """
    diff = t - q
    return torch.mean(torch.maximum(levels * diff, (levels - 1.0) * diff))


def train_quantile_net(z_train, aux_train, t_std_train,
                       z_val, aux_val, t_std_val,
                       levels: np.ndarray,
                       n_epochs: int = 30, batch_size: int = 512,
                       lr: float = 1e-3, hidden: int = 256, depth: int = 3,
                       device: str = 'cuda', torch_seed: int = 0,
                       verbose: bool = True) -> dict:
    """
    Returns dict with 'model', 'target_scale', 'train_losses', 'val_losses',
    'best_val'. Predictions from the returned model are in t_std units after
    multiplying by target_scale.
    """
    torch.manual_seed(torch_seed)

    # Global conditioning constant from the TRAINING split only
    target_scale = float(np.std(t_std_train))
    if target_scale <= 0:
        target_scale = 1.0

    def to_tensors(z, aux, t_std):
        return (torch.tensor(z), torch.tensor(aux),
                torch.tensor(t_std / target_scale, dtype=torch.float32))

    zt, auxt, tt = to_tensors(z_train, aux_train, t_std_train)
    zv, auxv, tv = to_tensors(z_val, aux_val, t_std_val)

    loader = DataLoader(TensorDataset(zt, auxt, tt), batch_size=batch_size,
                        shuffle=True, drop_last=True)

    levels_t = torch.tensor(levels, dtype=torch.float32, device=device)
    model = QuantileNet(n_input=z_train.shape[1], n_aux=aux_train.shape[1],
                        n_levels=len(levels), hidden=hidden,
                        depth=depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    zv_d, auxv_d, tv_d = zv.to(device), auxv.to(device), tv.to(device)

    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        epoch_loss, n_batches = 0.0, 0
        for zb, auxb, tb in loader:
            zb, auxb, tb = zb.to(device), auxb.to(device), tb.to(device)
            q = model(zb, auxb)
            loss = pinball_loss(q, tb.unsqueeze(1), levels_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_losses.append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i in range(0, len(tv_d), 8192):
                q = model(zv_d[i:i+8192], auxv_d[i:i+8192])
                val_loss += pinball_loss(
                    q, tv_d[i:i+8192].unsqueeze(1), levels_t
                ).item() * min(8192, len(tv_d) - i)
            val_loss /= len(tv_d)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:3d}/{n_epochs}: "
                  f"train={train_losses[-1]:.5f} val={val_loss:.5f} "
                  f"best={best_val:.5f} ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'model': model,
        'target_scale': target_scale,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val': best_val,
    }


def predict_root_quantiles(model, z, aux, s, target_scale: float,
                           device: str = 'cuda',
                           batch: int = 8192) -> np.ndarray:
    """
    Predict de-standardized root quantiles for datasets given featurization.

    Returns (N, L) array: q_root = q_net * target_scale * s.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(z), batch):
            zb = torch.tensor(z[i:i+batch], device=device)
            auxb = torch.tensor(aux[i:i+batch], device=device)
            out.append(model(zb, auxb).cpu().numpy())
    q_std = np.concatenate(out, axis=0) * target_scale
    return q_std * s[:, None]
