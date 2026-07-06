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
                    q_bayes: np.ndarray = None) -> dict:
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
    if q_bayes is not None:
        row['w1_bayes'] = w1_normalized(q_root, q_bayes, norm)
    return row


def print_results_table(rows: list, n_datasets: int):
    se95 = np.sqrt(0.05 * 0.95 / n_datasets)
    print("\n" + "=" * 86)
    print(f"{'Method':<24} {'Cov95':>7} {'Cov90':>7} {'Len95*n/th':>11} "
          f"{'W1truth*n/th':>13} {'W1bayes*n/th':>13}")
    print("-" * 86)
    for r in rows:
        wb = f"{r['w1_bayes']:13.4f}" if 'w1_bayes' in r else f"{'--':>13}"
        print(f"{r['method']:<24} {r['cov95']:7.3f} {r['cov90']:7.3f} "
              f"{r['len95']:11.3f} {r['w1_truth']:13.4f} {wb}")
    print("-" * 86)
    print(f"Nominal: 0.950 / 0.900. Coverage MC standard error: "
          f"~{se95:.4f} (95%). W1 in units of theta/n.")
    print("=" * 86)


def input_sensitivity_diagnostic(model, z, aux, s, target_scale: float,
                                 q_model: np.ndarray, q_bayes: np.ndarray,
                                 x_max: np.ndarray, levels: np.ndarray,
                                 rng, device: str = 'cuda') -> dict:
    """
    The anti-memorization guard. Three checks:

    1. tracking_corr_bayes: correlation across test datasets between the
       model's median root and the Bayes oracle's median root. Constant
       output (v1 artifact) gives ~0; a real conditional method gives ~1.
    2. tracking_corr_xmax: correlation between the model's median root and
       the observed maximum (the sufficient statistic).
    3. noise_response: relative change in predicted medians when the input
       is replaced by Gaussian noise. v1 gave ~0 (output unchanged); any
       genuinely conditional model responds strongly.
    """
    from .training import predict_root_quantiles

    i_med = level_index(levels, 0.5)
    med_model = q_model[:, i_med]
    med_bayes = q_bayes[:, i_med]

    corr_bayes = float(np.corrcoef(med_model, med_bayes)[0, 1])
    corr_xmax = float(np.corrcoef(med_model, x_max)[0, 1])

    z_noise = rng.standard_normal(z.shape).astype(np.float32)
    aux_noise = rng.standard_normal(aux.shape).astype(np.float32)
    q_noise = predict_root_quantiles(model, z_noise, aux_noise, s,
                                     target_scale, device=device)
    med_noise = q_noise[:, i_med]
    denom = np.mean(np.abs(med_model)) + 1e-12
    noise_response = float(np.mean(np.abs(med_noise - med_model)) / denom)

    return {
        'tracking_corr_bayes': corr_bayes,
        'tracking_corr_xmax': corr_xmax,
        'noise_response': noise_response,
    }
