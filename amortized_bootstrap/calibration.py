"""
Post-hoc quantile recalibration (Kuleshov et al. 2018 style), fit on the
validation split -- never on test data.

If the model were perfectly calibrated, the empirical frequency
    h(tau) = mean_i 1[ t_i <= q_i(tau) ]
over validation examples would equal tau for every level. When it does not,
we replace the level used for nominal tau by tau_adj = h^{-1}(tau), i.e. the
grid level whose EMPIRICAL exceedance frequency matches the nominal one.
This preserves each dataset's predicted shape (a per-level monotone
relabeling) while enforcing marginal calibration over the prior.

PIT values are invariant to monotone transforms of the target, so the
curve can be fit in asinh space and applied to root-space quantiles
unchanged.
"""

import numpy as np


def empirical_coverage_curve(q_pred: np.ndarray, t: np.ndarray,
                             levels: np.ndarray) -> np.ndarray:
    """h[j] = fraction of validation examples with t_i <= q_pred[i, j]."""
    h = (t[:, None] <= q_pred).mean(axis=0)
    # Enforce monotonicity (guards tiny MC non-monotonicities)
    return np.maximum.accumulate(h)


def adjusted_levels(levels: np.ndarray, h: np.ndarray) -> np.ndarray:
    """tau_adj for each nominal tau: the grid level where h equals tau.
    Clamps to the grid ends where the empirical curve does not reach."""
    return np.interp(levels, h, levels)


def apply_recalibration(q_pred: np.ndarray, levels: np.ndarray,
                        tau_adj: np.ndarray) -> np.ndarray:
    """
    Evaluate each dataset's predicted quantile function at the adjusted
    levels: q_recal[:, j] = Q_i(tau_adj[j]), by linear interpolation on the
    level grid. q_pred rows must be monotone (guaranteed by the model head).
    """
    N, L = q_pred.shape
    out = np.empty_like(q_pred)
    k = np.clip(np.searchsorted(levels, tau_adj, side='right') - 1, 0, L - 2)
    denom = levels[k + 1] - levels[k]
    w = np.clip((tau_adj - levels[k]) / np.maximum(denom, 1e-12), 0.0, 1.0)
    out = (1.0 - w)[None, :] * q_pred[:, k] + w[None, :] * q_pred[:, k + 1]
    return out
