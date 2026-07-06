"""
Classical baselines for the MEAN statistic (Monte Carlo resampling; unlike
the max, the bootstrap distribution of the mean has no closed form).

  - normal_interval_quantiles: the CLT/t-interval a practitioner would use
    by default; provably wrong rate/shape for alpha < 2
  - standard_bootstrap_mean_quantiles: n-out-of-n bootstrap (Athreya 1987:
    inconsistent for infinite variance -- converges to a random limit)
  - m_out_of_n_mean_quantiles: m-out-of-n bootstrap root quantiles WITHOUT
    rate correction; the caller applies (m/n)^(1 - 1/alpha) with alpha from
    {2 (naive CLT rate), alpha_true (infeasible oracle), alpha_hat
    (McCulloch, the feasible classical method)}. The rate depending on the
    unknown alpha is exactly the classical difficulty of this problem.
"""

import numpy as np
from scipy.stats import norm as normal_dist


def normal_interval_quantiles(X: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Root quantiles implied by the CLT interval: sd/sqrt(n) * z_tau."""
    n = X.shape[1]
    sd = np.std(X, axis=1, ddof=1)
    z = normal_dist.ppf(levels)
    return (sd / np.sqrt(n))[:, None] * z[None, :]


def standard_bootstrap_mean_quantiles(X: np.ndarray, levels: np.ndarray,
                                      B: int, rng,
                                      chunk: int = 50) -> np.ndarray:
    """n-out-of-n bootstrap root quantiles (T*_b - T_n) by resampling."""
    N, n = X.shape
    T_n = np.mean(X, axis=1)
    out = np.empty((N, len(levels)))
    for i0 in range(0, N, chunk):
        Xc = X[i0:i0 + chunk]
        C = len(Xc)
        idx = rng.integers(0, n, size=(C, B, n), dtype=np.int32)
        means = np.take_along_axis(Xc[:, None, :], idx, axis=2).mean(axis=2)
        roots = means - T_n[i0:i0 + chunk, None]
        out[i0:i0 + chunk] = np.quantile(roots, levels, axis=1).T
    return out


def m_out_of_n_mean_quantiles(X: np.ndarray, levels: np.ndarray,
                              m: int, B: int, rng,
                              chunk: int = 250) -> np.ndarray:
    """
    m-out-of-n bootstrap root quantiles (T*_m,b - T_n), UNSCALED.
    Callers multiply by (m/n)^(1 - 1/alpha) per dataset to approximate the
    quantiles of the full-sample root.
    """
    N, n = X.shape
    T_n = np.mean(X, axis=1)
    out = np.empty((N, len(levels)))
    for i0 in range(0, N, chunk):
        Xc = X[i0:i0 + chunk]
        C = len(Xc)
        idx = rng.integers(0, n, size=(C, B, m), dtype=np.int32)
        means = np.take_along_axis(Xc[:, None, :], idx, axis=2).mean(axis=2)
        roots = means - T_n[i0:i0 + chunk, None]
        out[i0:i0 + chunk] = np.quantile(roots, levels, axis=1).T
    return out


def rate_corrected(q_unscaled: np.ndarray, m: int, n: int,
                   alpha: np.ndarray) -> np.ndarray:
    """Apply the (m/n)^(1 - 1/alpha) rate correction per dataset."""
    factor = (m / n) ** (1.0 - 1.0 / alpha)
    return q_unscaled * factor[:, None]
