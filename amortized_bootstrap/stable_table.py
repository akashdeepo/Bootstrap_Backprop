"""
Standard symmetric alpha-stable quantile table, built by Monte Carlo from
the same CMS sampler that generates the data (sampler biases cancel), and
the McCulloch (1986) quantile estimator of (alpha, scale) built on top.

The table stores Q_std(alpha, tau) = quantiles of SaS(alpha, scale=1) on a
dense alpha grid; everything downstream interpolates:

  - evaluation truth:  q_true(tau) = sigma * n^(1/alpha-1) * Q_std(alpha, tau)
  - McCulloch alpha:   nu(alpha) = (Q95 - Q05) / (Q75 - Q25) is strictly
    decreasing in alpha; invert the empirical nu on the tabulated curve
  - McCulloch scale:   c_hat = (x75 - x25) / (Q75(a_hat) - Q25(a_hat))
  - parametric-stable bootstrap (analytic, via the stability property):
    q(tau) = c_hat * n^(1/alpha_hat - 1) * Q_std(alpha_hat, tau)

The table is symmetrized (Q <- (Q - reverse(Q))/2), which both enforces the
exact symmetry of SaS and halves the MC variance. Cached under data/.
"""

import time
import numpy as np

from . import config as cfg
from .families import sample_sas_vectorized

ALPHA_GRID = np.round(np.arange(1.02, 1.9951, 0.01), 4)  # 98 points


def build_or_load_table(n_draws: int = 20_000_000,
                        verbose: bool = True) -> dict:
    """Load the cached table for this draw count, or build it."""
    path = cfg.DATA_DIR / f"stable_qtable_{n_draws}.npz"
    if path.exists():
        d = np.load(path)
        return {'alphas': d['alphas'], 'levels': d['levels'], 'Q': d['Q']}

    levels = cfg.QUANTILE_LEVELS
    Q = np.empty((len(ALPHA_GRID), len(levels)))
    rng = cfg.RNG_TABLE
    t0 = time.time()
    if verbose:
        print(f"Building stable quantile table: {len(ALPHA_GRID)} alphas x "
              f"{n_draws/1e6:.0f}M draws (one-time, cached)")

    chunk = 5_000_000
    for i, alpha in enumerate(ALPHA_GRID):
        draws = np.concatenate([
            sample_sas_vectorized(np.float64(alpha),
                                  (min(chunk, n_draws - j),), rng)
            for j in range(0, n_draws, chunk)
        ])
        Q[i] = np.quantile(draws, levels)
        if verbose and (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ALPHA_GRID)} alphas ({time.time()-t0:.0f}s)")

    # Enforce exact symmetry of SaS; halves MC variance
    Q = 0.5 * (Q - Q[:, ::-1])

    np.savez_compressed(path, alphas=ALPHA_GRID, levels=levels, Q=Q)
    if verbose:
        print(f"  Saved {path.name} ({time.time()-t0:.0f}s)")
    return {'alphas': ALPHA_GRID, 'levels': levels, 'Q': Q}


def q_std_interp(table: dict, alphas_query: np.ndarray) -> np.ndarray:
    """Q_std quantiles at each queried alpha via linear interpolation.
    Returns (N, n_levels)."""
    A, Q = table['alphas'], table['Q']
    out = np.empty((len(alphas_query), Q.shape[1]))
    for j in range(Q.shape[1]):
        out[:, j] = np.interp(alphas_query, A, Q[:, j])
    return out


def _level_idx(levels: np.ndarray, tau: float) -> int:
    return int(np.argmin(np.abs(levels - tau)))


def mcculloch_estimate(X: np.ndarray, table: dict):
    """
    McCulloch (1986) quantile estimators for symmetric stable data.

    Returns (alpha_hat, c_hat), each (N,). alpha_hat is clipped to the
    tabulated grid range; datasets that look lighter-tailed than
    alpha=1.99 (e.g. nu below the curve minimum) clip to 1.99.
    """
    levels = table['levels']
    i05 = _level_idx(levels, 0.05)
    i25 = _level_idx(levels, 0.25)
    i75 = _level_idx(levels, 0.75)
    i95 = _level_idx(levels, 0.95)

    x05, x25, x75, x95 = np.percentile(X, [5.0, 25.0, 75.0, 95.0], axis=1)
    iqr = np.maximum(x75 - x25, 1e-12)
    nu_emp = (x95 - x05) / iqr

    A, Q = table['alphas'], table['Q']
    nu_curve = (Q[:, i95] - Q[:, i05]) / (Q[:, i75] - Q[:, i25])
    # nu is decreasing in alpha -> reverse for np.interp
    alpha_hat = np.interp(nu_emp, nu_curve[::-1], A[::-1])
    alpha_hat = np.clip(alpha_hat, A[0], A[-1])

    q75_at = np.interp(alpha_hat, A, Q[:, i75])
    q25_at = np.interp(alpha_hat, A, Q[:, i25])
    c_hat = iqr / np.maximum(q75_at - q25_at, 1e-12)
    return alpha_hat, c_hat


def parametric_stable_quantiles(X: np.ndarray, table: dict,
                                n: int) -> np.ndarray:
    """
    Parametric-stable bootstrap root quantiles, computed analytically:
    estimate (alpha, c) by McCulloch, then use the stability property.
    This is the strongest feasible classical method WHEN the family is
    known to be stable -- our reference oracle for Milestone 2.
    """
    alpha_hat, c_hat = mcculloch_estimate(X, table)
    scale_n = c_hat * n ** (1.0 / alpha_hat - 1.0)
    return scale_n[:, None] * q_std_interp(table, alpha_hat)
