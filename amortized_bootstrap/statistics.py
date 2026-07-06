"""
Statistic functions for all 5 (T, F) pairs.

Each function takes a 1D sample (or 2D batch) and returns the statistic value(s).
When input is 2D with shape (n_samples, sample_size), statistics are computed
along axis=1, returning a 1D array of length n_samples.
"""

import numpy as np


def sample_mean(x: np.ndarray) -> np.ndarray:
    """Sample mean. Works for 1D or 2D (batch) input."""
    if x.ndim == 1:
        return np.mean(x)
    return np.mean(x, axis=1)


def sample_max(x: np.ndarray) -> np.ndarray:
    """Sample maximum. Works for 1D or 2D (batch) input."""
    if x.ndim == 1:
        return np.max(x)
    return np.max(x, axis=1)


def var_quantile(x: np.ndarray, level: float = 0.99) -> np.ndarray:
    """
    Value-at-Risk at given probability level.
    VaR_p = F^{-1}(p) = quantile(x, p).

    For 1D input returns scalar; for 2D returns array of length n_samples.
    """
    if x.ndim == 1:
        return np.quantile(x, level)
    return np.quantile(x, level, axis=1)


def hill_estimator(x: np.ndarray, k: int = 34) -> np.ndarray:
    """
    Hill tail index estimator using top k order statistics.

    H_k = (1/k) * sum_{j=0}^{k-1} [log(x_{(n-j)}) - log(x_{(n-k)})]

    where x_{(1)} <= ... <= x_{(n)} are order statistics.
    Equivalently: sort descending, H_k = mean(log(x[:k]) - log(x[k])).

    Estimates gamma = 1/alpha (the extreme value index).

    Requires all values to be positive (Pareto data).

    For 1D input returns scalar; for 2D returns array of length n_samples.
    """
    if x.ndim == 1:
        x_sorted = np.sort(x)[::-1]  # descending
        log_top_k = np.log(x_sorted[:k])
        log_threshold = np.log(x_sorted[k])
        return np.mean(log_top_k - log_threshold)

    # Batch: sort each row descending
    x_sorted = np.sort(x, axis=1)[:, ::-1]  # (n_samples, sample_size), descending
    log_top_k = np.log(x_sorted[:, :k])           # (n_samples, k)
    log_threshold = np.log(x_sorted[:, k:k+1])    # (n_samples, 1)
    return np.mean(log_top_k - log_threshold, axis=1)


# Registry mapping pair names to statistic functions
STAT_FUNCTIONS = {
    'normal_mean': sample_mean,
    'stable_mean': sample_mean,
    'uniform_max': sample_max,
    'nts_var99': lambda x: var_quantile(x, level=0.99),
    'pareto_hill': lambda x: hill_estimator(x, k=34),
}
