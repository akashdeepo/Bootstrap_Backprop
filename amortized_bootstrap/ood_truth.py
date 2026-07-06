"""
Generic Monte Carlo ground truth and a generic bootstrap baseline, for any
family exposing sample_data / statistic / true_param.

Truth: per unique parameter, draw a large pool from F, then simulate the
n-sample estimator's root distribution by index-resampling datasets from
the pool (statistically indistinguishable from fresh draws at pool sizes
used here, and it applies the SAME statistic implementation as the
evaluation). T(F) comes from the family analytically.

Cached under results/ (small files, minutes to regenerate).
"""

import time
import numpy as np

from . import config as cfg


def mc_root_truth(family, params_unique: np.ndarray, n: int,
                  levels: np.ndarray, name: str, rng,
                  pool_size: int = 2_000_000,
                  n_replicates: int = 100_000,
                  verbose: bool = True) -> np.ndarray:
    """Root quantiles of the n-sample statistic per unique parameter row.
    Returns (P, n_levels)."""
    path = cfg.RESULTS_DIR / f"{name}.npz"
    if path.exists():
        d = np.load(path)
        if d['params'].shape == params_unique.shape and np.allclose(
                d['params'], params_unique):
            return d['q_root']

    P = len(params_unique)
    t_true = family.true_param(params_unique)
    q_root = np.empty((P, len(levels)))
    t0 = time.time()
    if verbose:
        print(f"  MC truth for {family.name}: {P} params x "
              f"{pool_size/1e6:.0f}M pool + {n_replicates/1e3:.0f}K reps")

    for p in range(P):
        pool = family.sample_data(params_unique[p:p+1], pool_size,
                                  rng).ravel()
        idx = rng.integers(0, pool_size, size=(n_replicates, n))
        reps = family.statistic(pool[idx])
        q_root[p] = np.quantile(reps - t_true[p], levels)
        if verbose and (p + 1) % 20 == 0:
            print(f"    {p+1}/{P} ({time.time()-t0:.0f}s)")

    np.savez_compressed(path, params=params_unique, q_root=q_root)
    if verbose:
        print(f"    cached {path.name} ({time.time()-t0:.0f}s)")
    return q_root


def bootstrap_roots_generic(X: np.ndarray, statistic_fn, levels: np.ndarray,
                            B: int, rng, chunk: int = 50) -> np.ndarray:
    """Standard n-out-of-n bootstrap root quantiles for any statistic."""
    N, n = X.shape
    T_n = statistic_fn(X)
    out = np.empty((N, len(levels)))
    for i0 in range(0, N, chunk):
        Xc = X[i0:i0 + chunk]
        C = len(Xc)
        idx = rng.integers(0, n, size=(C, B, n), dtype=np.int32)
        res = np.take_along_axis(Xc[:, None, :], idx, axis=2)
        stats = statistic_fn(res.reshape(C * B, n)).reshape(C, B)
        roots = stats - T_n[i0:i0 + chunk, None]
        out[i0:i0 + chunk] = np.quantile(roots, levels, axis=1).T
    return out
