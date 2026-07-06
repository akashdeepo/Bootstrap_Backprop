"""
Evaluation: CI coverage/length, W1 distance to the true root distribution,
regret to the Bayes oracle, and the input-sensitivity diagnostic.

All quantities are computed from root quantiles q(tau) at the shared level
grid. CIs use the standard root construction:
    CI_(1-alpha) for T(F) = [T_n - q(1-alpha/2), T_n - q(alpha/2)]

Normalization: root scales are theta/n for the max, so W1 and CI length are
reported in natural units, multiplied by n/theta, to make them comparable
across the prior range.

The input-sensitivity diagnostic is the permanent guard against the v1
memorization artifact: a model whose output does not track its input fails
Milestone 1 regardless of its other metrics.
"""

import numpy as np


def level_index(levels: np.ndarray, tau: float) -> int:
    idx = int(np.argmin(np.abs(levels - tau)))
    if abs(levels[idx] - tau) > 1e-9:
        raise ValueError(f"Level {tau} not on the level grid")
    return idx


def ci_from_root_quantiles(q_root: np.ndarray, T_n: np.ndarray,
                           levels: np.ndarray, alpha: float):
    """Returns (lo, hi) arrays of shape (N,)."""
    i_lo = level_index(levels, 1.0 - alpha / 2.0)
    i_hi = level_index(levels, alpha / 2.0)
    lo = T_n - q_root[:, i_lo]
    hi = T_n - q_root[:, i_hi]
    return lo, hi


def coverage_and_length(q_root: np.ndarray, T_n: np.ndarray,
                        true_value: np.ndarray, levels: np.ndarray,
                        alpha: float, norm: np.ndarray):
    """
    Returns (coverage, mean normalized length). norm is the per-dataset
    normalization factor (n / theta for the max).
    """
    lo, hi = ci_from_root_quantiles(q_root, T_n, levels, alpha)
    covered = (true_value >= lo) & (true_value <= hi)
    length = (hi - lo) * norm
    return float(np.mean(covered)), float(np.mean(length))


def w1_normalized(q_a: np.ndarray, q_b: np.ndarray,
                  norm: np.ndarray) -> float:
    """
    Mean absolute difference between two quantile functions over the level
    grid (an approximation of W1 on [0.005, 0.995]), normalized per dataset.
    """
    return float(np.mean(np.mean(np.abs(q_a - q_b), axis=1) * norm))


def evaluate_method(name: str, q_root: np.ndarray, T_n: np.ndarray,
                    true_value: np.ndarray, q_true: np.ndarray,
                    levels: np.ndarray, norm: np.ndarray,
                    q_ref: np.ndarray = None) -> dict:
    """q_ref: optional reference oracle (exact Bayes when computable, else
    the strongest classical oracle); adds a 'w1_ref' regret column."""
    cov95, len95 = coverage_and_length(q_root, T_n, true_value, levels,
                                       0.05, norm)
    cov90, len90 = coverage_and_length(q_root, T_n, true_value, levels,
                                       0.10, norm)
    row = {
        'method': name,
        'cov95': cov95,
        'cov90': cov90,
        'len95': len95,
        'w1_truth': w1_normalized(q_root, q_true, norm),
    }
    if q_ref is not None:
        row['w1_ref'] = w1_normalized(q_root, q_ref, norm)
    return row


def evaluate_direct_ci(name: str, lo95, hi95, lo90, hi90,
                       true_value: np.ndarray, norm: np.ndarray) -> dict:
    """Row for methods that produce CIs directly (e.g. the exact
    order-statistic CI), without a root quantile function. W1 is NaN."""
    cov95 = float(np.mean((true_value >= lo95) & (true_value <= hi95)))
    cov90 = float(np.mean((true_value >= lo90) & (true_value <= hi90)))
    return {
        'method': name,
        'cov95': cov95,
        'cov90': cov90,
        'len95': float(np.mean((hi95 - lo95) * norm)),
        'w1_truth': float('nan'),
    }


def print_results_table(rows: list, n_datasets: int, unit: str = 'nrm'):
    se95 = np.sqrt(0.05 * 0.95 / n_datasets)
    print("\n" + "=" * 88)
    print(f"{'Method':<26} {'Cov95':>7} {'Cov90':>7} {'Len95*' + unit:>11} "
          f"{'W1truth*' + unit:>13} {'W1ref*' + unit:>12}")
    print("-" * 88)
    for r in rows:
        wb = f"{r['w1_ref']:12.4f}" if 'w1_ref' in r else f"{'--':>12}"
        print(f"{r['method']:<26} {r['cov95']:7.3f} {r['cov90']:7.3f} "
              f"{r['len95']:11.3f} {r['w1_truth']:13.4f} {wb}")
    print("-" * 88)
    print(f"Nominal: 0.950 / 0.900. Coverage MC standard error: "
          f"~{se95:.4f} (95%). Lengths/W1 normalized per dataset ({unit}).")
    print("=" * 88)


def input_sensitivity_diagnostic(predict_fn, z, aux,
                                 q_model: np.ndarray, q_ref: np.ndarray,
                                 levels: np.ndarray, rng) -> dict:
    """
    The anti-memorization guard. Two checks:

    1. width_tracking_corr: correlation across test datasets between the
       model's log 95%-interval width and the reference's (truth or oracle).
       Constant output (v1 artifact) gives ~0; a real conditional method
       gives ~1. Width is used rather than the median because roots of
       symmetric statistics have median ~0 for every dataset.
    2. noise_response: relative change in predicted widths when the input
       is replaced by Gaussian noise. v1 gave ~0 (output unchanged); any
       genuinely conditional model responds strongly.

    predict_fn(z, aux) must return de-standardized root quantiles (N, L),
    i.e. the experiment's full prediction pipeline as a closure.
    """
    i_lo = level_index(levels, 0.025)
    i_hi = level_index(levels, 0.975)

    w_model = q_model[:, i_hi] - q_model[:, i_lo]
    w_ref = q_ref[:, i_hi] - q_ref[:, i_lo]
    w_model = np.maximum(w_model, 1e-12)
    w_ref = np.maximum(w_ref, 1e-12)
    corr = float(np.corrcoef(np.log(w_model), np.log(w_ref))[0, 1])

    z_noise = rng.standard_normal(z.shape).astype(np.float32)
    aux_noise = rng.standard_normal(aux.shape).astype(np.float32)
    q_noise = predict_fn(z_noise, aux_noise)
    w_noise = q_noise[:, i_hi] - q_noise[:, i_lo]
    denom = np.mean(np.abs(w_model)) + 1e-12
    noise_response = float(np.mean(np.abs(w_noise - w_model)) / denom)

    return {
        'width_tracking_corr': corr,
        'noise_response': noise_response,
    }
