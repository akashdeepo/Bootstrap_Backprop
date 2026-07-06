"""
NTS ground-truth machinery.

1. VaR_std grid: symmetric NTS is location-scale in (mu, sigma), so the
   0.99-quantile of the STANDARD family depends only on (alpha, theta).
   A one-time MC grid + bilinear interpolation in (alpha, log theta)
   supplies T(F) for training-target centering at any of the ~10^6 prior
   draws. Centering error is ~1% of the root scale at n=200 (quantile SE
   ratio sqrt(n / n_pool)), negligible for training and calibration.

2. Test truth: for each held-out (alpha, theta), draw a 2M pool of
   standard NTS values, take T(F)_std = VaR of the pool, and simulate the
   ROOT distribution of the n=200 estimator by resampling datasets from
   the pool (index resampling from a 2M pool is statistically
   indistinguishable from fresh draws at this precision, and uses the
   same np.quantile interpolation as the statistic itself). Location-scale
   then gives every test dataset's truth as sigma * q_root_std.

Both caches are deterministic (RNG_TABLE stream) and small; they are
stored under results/ and committed because they are expensive to
regenerate (unlike data/ caches).
"""

import time
import numpy as np

from . import config as cfg
from .families import sample_nts_vectorized

VAR_LEVEL = 0.99
ALPHA_GRID = np.round(np.linspace(1.05, 1.95, 19), 4)
THETA_GRID = np.round(np.geomspace(0.25, 3.5, 15), 4)


def build_or_load_var_grid(n_draws: int = 2_000_000,
                           verbose: bool = True) -> dict:
    path = cfg.RESULTS_DIR / f"nts_var_grid_{n_draws}.npz"
    if path.exists():
        d = np.load(path)
        return {'alphas': d['alphas'], 'thetas': d['thetas'], 'V': d['V']}

    V = np.empty((len(ALPHA_GRID), len(THETA_GRID)))
    rng = cfg.RNG_TABLE
    t0 = time.time()
    if verbose:
        print(f"Building NTS VaR_std grid: {len(ALPHA_GRID)}x"
              f"{len(THETA_GRID)} cells x {n_draws/1e6:.1f}M draws "
              f"(one-time, cached)")
    for i, a in enumerate(ALPHA_GRID):
        for j, th in enumerate(THETA_GRID):
            draws = sample_nts_vectorized(
                np.full((1, 1), a), np.full((1, 1), th),
                (1, n_draws), rng).ravel()
            V[i, j] = np.quantile(draws, VAR_LEVEL)
        if verbose:
            print(f"  alpha={a} done ({time.time()-t0:.0f}s)")

    np.savez_compressed(path, alphas=ALPHA_GRID, thetas=THETA_GRID, V=V)
    if verbose:
        print(f"  Saved {path.name} ({time.time()-t0:.0f}s)")
    return {'alphas': ALPHA_GRID, 'thetas': THETA_GRID, 'V': V}


def make_var_std_fn(grid: dict):
    """Bilinear interpolator in (alpha, log theta)."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (grid['alphas'], np.log(grid['thetas'])), grid['V'],
        bounds_error=False, fill_value=None)  # linear extrapolation

    def var_std(alpha: np.ndarray, theta: np.ndarray) -> np.ndarray:
        pts = np.stack([alpha, np.log(theta)], axis=1)
        return interp(pts)

    return var_std


def build_or_load_test_truth(params_std: np.ndarray, n: int,
                             levels: np.ndarray, name: str,
                             pool_size: int = 2_000_000,
                             n_replicates: int = 100_000,
                             verbose: bool = True) -> dict:
    """
    Exact-grade truth for held-out standard-family parameters.

    Args:
        params_std: (P, 2) unique (alpha, theta) pairs
    Returns dict:
        't_std':      (P,)   VaR_std(alpha, theta)
        'q_root_std': (P, L) root quantiles of the n-sample estimator
    """
    path = cfg.RESULTS_DIR / f"{name}.npz"
    if path.exists():
        d = np.load(path)
        if d['params_std'].shape == params_std.shape and np.allclose(
                d['params_std'], params_std):
            return {'t_std': d['t_std'], 'q_root_std': d['q_root_std']}

    rng = cfg.RNG_TABLE
    P = len(params_std)
    t_std = np.empty(P)
    q_root_std = np.empty((P, len(levels)))
    t0 = time.time()
    if verbose:
        print(f"Building NTS test truth: {P} params x "
              f"{pool_size/1e6:.0f}M pool + {n_replicates/1e3:.0f}K "
              f"replicates (one-time, cached)")

    for p in range(P):
        a, th = params_std[p]
        pool = sample_nts_vectorized(
            np.full((1, 1), a), np.full((1, 1), th),
            (1, pool_size), rng).ravel()
        t_std[p] = np.quantile(pool, VAR_LEVEL)

        idx = rng.integers(0, pool_size, size=(n_replicates, n))
        reps = np.quantile(pool[idx], VAR_LEVEL, axis=1)
        q_root_std[p] = np.quantile(reps - t_std[p], levels)
        if verbose and (p + 1) % 20 == 0:
            print(f"  {p+1}/{P} params ({time.time()-t0:.0f}s)")

    np.savez_compressed(path, params_std=params_std, t_std=t_std,
                        q_root_std=q_root_std)
    if verbose:
        print(f"  Saved {path.name} ({time.time()-t0:.0f}s)")
    return {'t_std': t_std, 'q_root_std': q_root_std}
