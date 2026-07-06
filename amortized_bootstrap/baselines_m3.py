"""
Classical baselines for the M3 statistics (VaR quantile and Hill).

VaR_0.99 at n=200 (np=198, only ~2 expected tail points beyond):
  - standard bootstrap of the quantile (MC resampling)
  - m-out-of-n bootstrap with sqrt(m/n) rate correction (the quantile root
    is sqrt(n)-regular in theory, but n(1-p)=2 makes the normal regime a
    fiction at this n -- the honest classical toolkit)
  - exact order-statistic (binomial) CI: the distribution-free gold
    standard [X_(r), X_(s)]. At p=0.99, n=200 the achievable two-sided
    coverage is CAPPED at P(Bin(200, 0.99) <= 199) = 1 - 0.99^200 = 0.866,
    below 95% -- the classical method cannot reach nominal at all.

Hill with k top order statistics:
  - standard bootstrap (recompute Hill on resamples; np.partition, not
    full sort, since Hill needs only the top k+1 set and threshold)
  - m-out-of-n with k_m = floor(m^(2/3)) and sqrt(k_m/k_n) rate correction
  - parametric MLE: alpha_hat = n / sum log(x_i / x_(1)), root quantiles
    analytic via the exact Gamma representation
"""

import numpy as np
from scipy.stats import binom
from scipy.stats import gamma as gamma_dist


# ----------------------------------------------------------------------
# Quantile (VaR) baselines
# ----------------------------------------------------------------------

def bootstrap_quantile_roots(X: np.ndarray, levels: np.ndarray,
                             var_level: float, B: int, rng,
                             m: int = None, chunk: int = 50) -> np.ndarray:
    """
    Bootstrap root quantiles for the empirical quantile statistic.
    m = resample size (None -> n, the standard bootstrap). No rate
    correction applied here; callers scale m-out-of-n results.
    """
    N, n = X.shape
    m = m or n
    T_n = np.quantile(X, var_level, axis=1)
    out = np.empty((N, len(levels)))
    for i0 in range(0, N, chunk):
        Xc = X[i0:i0 + chunk]
        C = len(Xc)
        idx = rng.integers(0, n, size=(C, B, m), dtype=np.int32)
        stats = np.quantile(np.take_along_axis(Xc[:, None, :], idx, axis=2),
                            var_level, axis=2)
        roots = stats - T_n[i0:i0 + chunk, None]
        out[i0:i0 + chunk] = np.quantile(roots, levels, axis=1).T
    return out


def binomial_exact_ci(X_sorted: np.ndarray, var_level: float, alpha: float):
    """
    Distribution-free order-statistic CI for the var_level quantile.
    Returns (lo, hi, exact_coverage). Equal-tailed construction; the upper
    index clamps at X_(n) when the binomial upper tail is unreachable.
    """
    N, n = X_sorted.shape
    r = int(binom.ppf(alpha / 2.0, n, var_level))        # lower index, 1-based
    r = max(r, 1)
    s = int(binom.ppf(1.0 - alpha / 2.0, n, var_level)) + 1
    s = min(s, n)
    exact_cov = float(binom.cdf(s - 1, n, var_level)
                      - binom.cdf(r - 1, n, var_level))
    return X_sorted[:, r - 1], X_sorted[:, s - 1], exact_cov


# ----------------------------------------------------------------------
# Hill baselines
# ----------------------------------------------------------------------

def _hill_from_batch(x: np.ndarray, k: int) -> np.ndarray:
    """Hill estimator over the last axis via partition (top k+1 only)."""
    n = x.shape[-1]
    part = np.partition(x, n - k - 1, axis=-1)
    threshold = part[..., n - k - 1]
    top = part[..., n - k:]
    return np.mean(np.log(top), axis=-1) - np.log(threshold)


def bootstrap_hill_roots(X: np.ndarray, levels: np.ndarray, k: int,
                         B: int, rng, m: int = None,
                         k_sub: int = None, chunk: int = 50) -> np.ndarray:
    """
    Bootstrap root quantiles for the Hill estimator. For m-out-of-n pass
    m and k_sub (Hill computed with k_sub on subsamples); no rate
    correction applied here.
    """
    N, n = X.shape
    m = m or n
    k_sub = k_sub or k
    T_n = _hill_from_batch(X, k)
    out = np.empty((N, len(levels)))
    for i0 in range(0, N, chunk):
        Xc = X[i0:i0 + chunk]
        C = len(Xc)
        idx = rng.integers(0, n, size=(C, B, m), dtype=np.int32)
        stats = _hill_from_batch(
            np.take_along_axis(Xc[:, None, :], idx, axis=2), k_sub)
        roots = stats - T_n[i0:i0 + chunk, None]
        out[i0:i0 + chunk] = np.quantile(roots, levels, axis=1).T
    return out


def parametric_hill_quantiles(X: np.ndarray, k: int,
                              levels: np.ndarray) -> np.ndarray:
    """MLE plug-in + exact Gamma root: gamma_hat * (GammaPPF(tau;k)/k - 1)."""
    n = X.shape[1]
    x_min = np.min(X, axis=1)
    gamma_hat = (np.sum(np.log(X), axis=1) - n * np.log(x_min)) / n
    gk = gamma_dist.ppf(levels, a=k) / k
    return gamma_hat[:, None] * (gk[None, :] - 1.0)
