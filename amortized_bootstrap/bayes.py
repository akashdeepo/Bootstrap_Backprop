"""
Bayes-optimal oracle for the uniform-max family.

Under theta ~ LogUniform(a, b) and X_1..X_n ~ U(0, theta):
  - posterior density: p(theta | X) propto theta^-(n+1) on [L, b],
    L = max(X_(n), a)
  - change of variables V = (theta / L)^-n gives V | X ~ Uniform(v_min, 1)
    with v_min = (L / b)^n  -- the posterior is EXACTLY a uniform in V,
    so posterior expectations are plain averages over a uniform grid.
  - posterior predictive CDF of the root t = max(X_indep) - theta:
        F(t | X) = E_post[ max(0, 1 + t/theta)^n ],  t <= 0

The predictive depends on the data only through L (sufficiency). We compute
predictive quantiles on a grid of L values and interpolate per dataset.

This is the provably optimal data-conditional answer under the prior; the
trained network's distance to it ("regret to Bayes") measures how much of
the achievable performance the network captures.
"""

import numpy as np


def _predictive_quantiles_for_grid(L_grid: np.ndarray, levels: np.ndarray,
                                   n: int, b: float,
                                   n_nodes: int = 512,
                                   n_t: int = 600) -> np.ndarray:
    """Predictive root quantiles for each L in L_grid. Returns (G, n_levels)."""
    G = len(L_grid)
    out = np.empty((G, len(levels)))

    # Relative t magnitudes (as a fraction of L), geometric toward 0.
    # Root scale is ~L/n; cover from far below the 0.005 quantile to
    # far above the 0.995 quantile.
    rel_mag = np.geomspace(3e-7, 0.5, n_t)[::-1]  # descending magnitude
    # t ascending: -0.5L ... -3e-7 L, then append 0 (F=1)

    chunk = 50
    j_nodes = (np.arange(n_nodes) + 0.5) / n_nodes

    for start in range(0, G, chunk):
        Ls = L_grid[start:start + chunk]                     # (C,)
        C = len(Ls)
        # v_min = (L/b)^n, computed in log space (underflows to 0 harmlessly)
        v_min = np.exp(n * (np.log(Ls) - np.log(b)))         # (C,)
        v = v_min[:, None] + j_nodes[None, :] * (1.0 - v_min[:, None])  # (C,K)
        theta = Ls[:, None] * np.exp(-np.log(v) / n)         # (C,K)

        t = -(Ls[:, None] * rel_mag[None, :])                # (C,T) ascending
        # F(t) = mean_k max(0, 1 + t/theta_k)^n
        ratio = 1.0 + t[:, None, :] / theta[:, :, None]      # (C,K,T)
        np.clip(ratio, 0.0, None, out=ratio)
        F = np.mean(np.exp(n * np.log(np.maximum(ratio, 1e-300))) * (ratio > 0),
                    axis=1)                                  # (C,T)

        # Append t=0 with F=1 for interpolation
        t_full = np.concatenate([t, np.zeros((C, 1))], axis=1)
        F_full = np.concatenate([F, np.ones((C, 1))], axis=1)

        for c in range(C):
            out[start + c] = np.interp(levels, F_full[c], t_full[c])

    return out


def bayes_root_quantiles(x_max: np.ndarray, levels: np.ndarray, n: int,
                         a: float, b: float,
                         grid_size: int = 2000) -> np.ndarray:
    """
    Posterior-predictive root quantiles for each dataset, given its maximum.

    Args:
        x_max  (N,) observed maxima
        levels (L,) quantile levels
    Returns:
        (N, L) predictive quantiles of t = T_n(X_indep) - theta given X
    """
    L_vals = np.maximum(x_max, a)
    lo = L_vals.min() * 0.999
    L_grid = np.geomspace(lo, b, grid_size)
    Q_grid = _predictive_quantiles_for_grid(L_grid, levels, n, b)  # (G, L)

    out = np.empty((len(L_vals), len(levels)))
    for j in range(len(levels)):
        out[:, j] = np.interp(L_vals, L_grid, Q_grid[:, j])
    return out
