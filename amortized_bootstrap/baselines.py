"""
Classical baselines for the max statistic, computed EXACTLY.

For the sample maximum, the resampling distributions have closed forms in
terms of the order statistics, so no Monte Carlo resampling is needed:

  - n-out-of-n bootstrap:  P(T* <= X_(i)) = (i/n)^n
  - m-out-of-n subsampling without replacement (Politis-Romano):
        P(T*_m <= X_(i)) = C(i, m) / C(n, m),  i >= m
    with the rate correction r* = (m/n) * (T*_m - T_n) as proxy draws of
    the root (the max converges at rate n)
  - parametric bootstrap (theta_hat = X_(n), the MLE):
        root* quantiles are theta_hat * (tau^(1/n) - 1) exactly
  - exact pivot (oracle): X_(n)/theta is pivotal with CDF u^n, giving the
    exact equal-tailed CI [T_n / (1-alpha/2)^(1/n), T_n / (alpha/2)^(1/n)]

All functions return root quantiles q(tau) at the requested levels, shaped
(N, n_levels), from which CIs are built as
    [T_n - q(1 - alpha/2), T_n - q(alpha/2)].
"""

import numpy as np
from scipy.special import gammaln


def _quantile_index_table(cdf: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Smallest index i with cdf[i] >= tau, for each level."""
    return np.searchsorted(cdf, levels, side='left')


def standard_bootstrap_quantiles(X_sorted: np.ndarray,
                                 levels: np.ndarray) -> np.ndarray:
    """Exact n-out-of-n bootstrap root quantiles. X_sorted: (N, n)."""
    n = X_sorted.shape[1]
    i = np.arange(1, n + 1)
    cdf = (i / n) ** n
    idx = _quantile_index_table(cdf, levels)          # values in [0, n-1]
    T_n = X_sorted[:, -1]
    return X_sorted[:, idx] - T_n[:, None]


def subsampling_quantiles(X_sorted: np.ndarray, levels: np.ndarray,
                          m: int) -> np.ndarray:
    """
    Exact m-out-of-n (without replacement) subsampling root quantiles with
    rate correction (m/n). X_sorted: (N, n).
    """
    n = X_sorted.shape[1]
    i = np.arange(m, n + 1)
    log_cdf = (gammaln(i + 1) - gammaln(i - m + 1)
               - gammaln(n + 1) + gammaln(n - m + 1))
    cdf = np.exp(log_cdf)
    idx_local = _quantile_index_table(cdf, levels)
    idx = (m - 1) + idx_local                         # index into full order stats
    idx = np.minimum(idx, n - 1)
    T_n = X_sorted[:, -1]
    return (m / n) * (X_sorted[:, idx] - T_n[:, None])


def parametric_bootstrap_quantiles(X_sorted: np.ndarray,
                                   levels: np.ndarray) -> np.ndarray:
    """Exact parametric bootstrap root quantiles with theta_hat = X_(n)."""
    n = X_sorted.shape[1]
    theta_hat = X_sorted[:, -1]
    return theta_hat[:, None] * (levels[None, :] ** (1.0 / n) - 1.0)


def exact_pivot_ci(T_n: np.ndarray, n: int, alpha: float):
    """
    Exact equal-tailed CI for theta from the pivot (X_(n)/theta) ~ CDF u^n.
    Returns (lo, hi), each (N,).
    """
    lo = T_n / (1.0 - alpha / 2.0) ** (1.0 / n)
    hi = T_n / (alpha / 2.0) ** (1.0 / n)
    return lo, hi
