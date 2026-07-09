"""
Pretrained root networks: confidence intervals in three lines.

    from amortized_bootstrap import pretrained
    lo, hi = pretrained.interval(x, statistic="var99", level=0.95)

Each statistic maps to a trained specialist checkpoint from this
repository (weights and recalibration frozen; the same models that
produced the paper's results):

    "max"    endpoint of a bounded distribution; T_n = sample maximum.
             Trained on Beta-tailed endpoints with UNKNOWN contact order
             b in (0.4, 2.6), so the convergence rate itself is learned
             from the data.
    "mean"   population mean under possibly infinite variance; trained
             on symmetric alpha-stable data, alpha in (1.1, 1.95).
    "hill"   tail index gamma = 1/alpha via the Hill estimator (k = 34);
             trained on Pareto tails, alpha in (1.5, 4).
    "var99"  99% quantile (value-at-risk); trained on normal tempered
             stable data.

Scope, stated plainly: models are trained at n = 200 observations under
the priors above. Coverage is calibrated marginally over those priors;
performance on data far outside them degrades (see the paper's
out-of-family analysis). Inputs whose interquartile range falls outside
the training scale range trigger a warning rather than silent
extrapolation.
"""

import warnings

import numpy as np
import torch

from . import config as cfg
from .datagen import featurize
from .model import QuantileNet
from .training import predict_root_quantiles
from .statistics import hill_estimator
from .calibration import apply_recalibration

# statistic -> (checkpoint stem, compress inputs, asinh targets (c0),
#               root scales with data scale, log(-root) targets)
_SPECS = {
    "max": ("m4c_beta_max", False, False, True, True),
    "mean": ("m2_stable_mean", True, True, True, False),
    "hill": ("m3_pareto_hill", True, True, False, False),
    "var99": ("m3_nts_var", True, True, True, False),
}

_STAT_FN = {
    "max": lambda x: np.max(x, axis=1),
    "mean": lambda x: np.mean(x, axis=1),
    "hill": lambda x: hill_estimator(x, k=34),
    "var99": lambda x: np.quantile(x, 0.99, axis=1),
}

_N_TRAIN = 200          # models are trained at this sample size
_IQR_RANGE = (0.05, 100.0)   # warn outside this per-dataset scale range

_cache = {}


def _load(statistic: str):
    if statistic in _cache:
        return _cache[statistic]
    if statistic not in _SPECS:
        raise ValueError(f"unknown statistic {statistic!r}; "
                         f"choose from {sorted(_SPECS)}")
    stem, compress, use_c0, scale_root, log_neg = _SPECS[statistic]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(cfg.RESULTS_DIR / f"{stem}_model.pt",
                      map_location=device, weights_only=False)
    model = QuantileNet(n_input=_N_TRAIN, n_aux=2,
                        n_levels=len(cfg.QUANTILE_LEVELS),
                        hidden=384, depth=3).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    entry = {
        'model': model,
        'device': device,
        'target_scale': float(ckpt['target_scale']),
        'c0': float(ckpt['c0']) if use_c0 else None,
        'tau_adj': np.load(cfg.RESULTS_DIR / f"{stem}.npz")['tau_adj'],
        'compress': compress,
        'scale_root': scale_root,
        'log_neg': log_neg,
    }
    _cache[statistic] = entry
    return entry


def _validate(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("x must be a 1-D dataset or a 2-D batch")
    if x.shape[1] != _N_TRAIN:
        raise ValueError(
            f"pretrained root networks require exactly n = {_N_TRAIN} "
            f"observations per dataset (got {x.shape[1]}); varying-n "
            f"models are future work")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    return x, squeeze


def root_quantiles(x, statistic: str):
    """
    Recalibrated quantiles of the root T_n - T(F) given the dataset(s).

    Args:
        x: array of shape (n,) or (batch, n), n = 200.
        statistic: "max", "mean", "hill", or "var99".
    Returns:
        (levels, q): levels of shape (199,), q of shape (batch, 199)
        (or (199,) for a single dataset).
    """
    x, squeeze = _validate(x)
    e = _load(statistic)
    levels = cfg.QUANTILE_LEVELS

    z, aux, s = featurize(x, compress=e['compress'])
    s_root = s if e['scale_root'] else np.ones(len(x))
    iqr_bad = (s < _IQR_RANGE[0]) | (s > _IQR_RANGE[1])
    if np.any(iqr_bad):
        warnings.warn(
            f"{int(iqr_bad.sum())} dataset(s) have IQR outside "
            f"{_IQR_RANGE}; far outside the training scale range the "
            f"intervals are extrapolations (see the paper's "
            f"out-of-family analysis)")

    q_y = predict_root_quantiles(e['model'], z, aux, np.ones(len(z)),
                                 e['target_scale'], device=e['device'])
    if e['log_neg']:
        q = -np.exp(np.clip(q_y[:, ::-1], -700.0, 50.0)) * s_root[:, None]
    elif e['c0'] is not None:
        q = np.sinh(np.clip(q_y, -20.0, 20.0)) * e['c0'] * s_root[:, None]
    else:
        q = q_y * s_root[:, None]
    q = apply_recalibration(q, levels, e['tau_adj'])
    return (levels, q[0]) if squeeze else (levels, q)


def interval(x, statistic: str, level: float = 0.95):
    """
    Equal-tailed confidence interval(s) for T(F).

    Args:
        x: array of shape (n,) or (batch, n), n = 200.
        statistic: "max", "mean", "hill", or "var99".
        level: confidence level in (0.5, 0.99]; quantiles between grid
            levels are linearly interpolated.
    Returns:
        (lo, hi): floats for a single dataset, arrays for a batch.
    """
    if not 0.5 < level <= 0.99:
        raise ValueError("level must be in (0.5, 0.99]")
    x, squeeze = _validate(x)
    levels, q = root_quantiles(x, statistic)
    if q.ndim == 1:
        q = q[None, :]
    T_n = _STAT_FN[statistic](x)

    alpha = 1.0 - level
    q_hi = np.array([np.interp(1.0 - alpha / 2.0, levels, row)
                     for row in q])
    q_lo = np.array([np.interp(alpha / 2.0, levels, row) for row in q])
    lo, hi = T_n - q_hi, T_n - q_lo
    return (float(lo[0]), float(hi[0])) if squeeze else (lo, hi)


def width_tracking_diagnostic(x, statistic: str, rng=None):
    """
    The width-tracking diagnostic for a batch of datasets: returns the
    predicted 95% interval widths and the relative response of those
    widths to pure-noise inputs. An input-blind (memorizing) model shows
    near-zero noise response; see the paper, Section on diagnostics.
    """
    x, _ = _validate(x)
    rng = rng or np.random.default_rng(0)
    levels, q = root_quantiles(x, statistic)
    if q.ndim == 1:
        q = q[None, :]
    i_lo = int(np.argmin(np.abs(levels - 0.025)))
    i_hi = int(np.argmin(np.abs(levels - 0.975)))
    w = q[:, i_hi] - q[:, i_lo]

    x_noise = rng.standard_normal(x.shape)
    _, qn = root_quantiles(np.abs(x_noise) + 1.0, statistic)
    if qn.ndim == 1:
        qn = qn[None, :]
    wn = qn[:, i_hi] - qn[:, i_lo]
    response = float(np.mean(np.abs(wn - w)) / (np.mean(np.abs(w)) + 1e-12))
    return {'widths': w, 'noise_response': response}
