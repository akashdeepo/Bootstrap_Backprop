"""
Milestone 3a: Hill estimator on the Pareto family.

X ~ c * Pareto(alpha), alpha ~ U(1.5, 4), c ~ LogU(0.5, 5), T_n = Hill
with k = 34, T(F) = gamma = 1/alpha, n = 200. The Hill root is
scale-invariant and, for exact Pareto, analytically Gamma-distributed
(Renyi), so truth is exact and a Bayes oracle is computable -- the second
regret-to-Bayes anchor after uniform max.

Usage:
    python -m amortized_bootstrap.experiments.m3_pareto_hill [--epochs 40]
"""

import argparse
import time
import numpy as np
import torch

from .. import config as cfg
from ..families import ParetoHillFamily
from ..datagen import generate_examples, featurize
from ..training import train_quantile_net, predict_root_quantiles
from ..model import count_parameters
from ..hill_oracle import hill_bayes_root_quantiles
from ..baselines_m3 import (bootstrap_hill_roots, parametric_hill_quantiles)
from ..evaluation import (evaluate_method, print_results_table,
                          input_sensitivity_diagnostic)
from ..calibration import (empirical_coverage_curve, adjusted_levels,
                           apply_recalibration)
from ..export import export_table

N_VAL = 20_000
N_TEST_PARAMS = 200
N_TEST_DATASETS_PER_PARAM = 100
SUBSAMPLE_M = 34            # m-out-of-n resample size (n^(2/3) convention)
K_SUB = 10                  # Hill k on subsamples: floor(34^(2/3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--n-train', type=int, default=800_000)
    parser.add_argument('--hidden', type=int, default=384)
    parser.add_argument('--b-boot', type=int, default=2000)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = args.device

    n = cfg.N
    levels = cfg.QUANTILE_LEVELS
    family = ParetoHillFamily()
    k = family.k

    print("=" * 70)
    print("Milestone 3a: Pareto Hill, alpha ~ U(1.5, 4), k = 34, n = 200")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    t0 = time.time()
    X_tr, t_tr, _ = generate_examples(family, args.n_train, n, cfg.RNG_TRAIN)
    X_va, t_va, p_va = generate_examples(family, N_VAL, n, cfg.RNG_VAL)

    p_test_unique = family.sample_params(N_TEST_PARAMS, cfg.RNG_TEST)
    params_test = np.repeat(p_test_unique, N_TEST_DATASETS_PER_PARAM, axis=0)
    X_te = family.sample_data(params_test, n, cfg.RNG_TEST)
    gamma_te = 1.0 / params_test[:, 0]
    print(f"[1] Data: train={args.n_train}, val={N_VAL}, "
          f"test={len(params_test)} ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Featurize. The Hill root is SCALE-INVARIANT: no de-standardization
    #    by the data scale (s_root = 1); only the global c0 conditioning.
    # ------------------------------------------------------------------
    t0 = time.time()
    z_tr, aux_tr, _ = featurize(X_tr, compress=True)
    z_va, aux_va, _ = featurize(X_va, compress=True)
    z_te, aux_te, _ = featurize(X_te, compress=True)

    q25, q75 = np.percentile(t_tr, [25.0, 75.0])
    c0 = (q75 - q25) / 1.349
    y_tr = np.arcsinh(t_tr / c0)
    y_va = np.arcsinh(t_va / c0)
    print(f"[2] Featurized; c0={c0:.5f} ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print(f"[3] Training on {device} ...")
    result = train_quantile_net(
        z_tr, aux_tr, y_tr, z_va, aux_va, y_va, levels,
        n_epochs=args.epochs, batch_size=512, lr=1e-3, hidden=args.hidden,
        device=device, torch_seed=cfg.TORCH_SEED)
    model = result['model']
    target_scale = result['target_scale']
    print(f"    params={count_parameters(model):,}, "
          f"best val pinball={result['best_val']:.5f}")

    def predict_q(z_in, aux_in):
        q_y = predict_root_quantiles(model, z_in, aux_in,
                                     np.ones(len(z_in)), target_scale,
                                     device=device)
        return np.sinh(np.clip(q_y, -20.0, 20.0)) * c0

    # ------------------------------------------------------------------
    # 4. Predict, recalibrate (own roots), oracles, baselines
    # ------------------------------------------------------------------
    t0 = time.time()
    q_model = predict_q(z_te, aux_te)

    q_val = predict_q(z_va, aux_va)
    own_va = family.statistic(X_va) - family.true_param(p_va)
    h_curve = empirical_coverage_curve(q_val, own_va, levels)
    tau_adj = adjusted_levels(levels, h_curve)
    q_model_recal = apply_recalibration(q_model, levels, tau_adj)

    T_n = family.statistic(X_te)
    q_true = family.true_root_quantiles(params_test, levels)
    norm = np.sqrt(k) / gamma_te   # root scale is gamma / sqrt(k)

    print("    Bayes oracle (posterior mixture)...")
    q_bayes = hill_bayes_root_quantiles(
        X_te, k, levels, family.alpha_min, family.alpha_max,
        family.c_min, family.c_max)

    print("    bootstrap baselines...")
    q_sboot = bootstrap_hill_roots(X_te, levels, k, B=args.b_boot,
                                   rng=cfg.RNG_BASELINE)
    q_moon_raw = bootstrap_hill_roots(X_te, levels, k, B=args.b_boot,
                                      rng=cfg.RNG_BASELINE,
                                      m=SUBSAMPLE_M, k_sub=K_SUB)
    q_moon = q_moon_raw * np.sqrt(K_SUB / k)
    q_param = parametric_hill_quantiles(X_te, k, levels)
    print(f"[4] Predictions + baselines ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    rows = [
        evaluate_method('standard_bootstrap', q_sboot, T_n, gamma_te,
                        q_true, levels, norm),
        evaluate_method(f'm_of_n_m{SUBSAMPLE_M}_k{K_SUB}', q_moon, T_n,
                        gamma_te, q_true, levels, norm),
        evaluate_method('parametric_MLE', q_param, T_n, gamma_te,
                        q_true, levels, norm),
        evaluate_method('bayes_(oracle)', q_bayes, T_n, gamma_te,
                        q_true, levels, norm),
        evaluate_method('learned_raw', q_model, T_n, gamma_te,
                        q_true, levels, norm, q_ref=q_bayes),
        evaluate_method('learned_recal_(ours)', q_model_recal, T_n,
                        gamma_te, q_true, levels, norm, q_ref=q_bayes),
    ]
    print_results_table(rows, len(params_test), unit='sqk/g')
    export_table(rows, 'm3_pareto_hill',
                 'Hill estimator, Pareto family, alpha ~ U(1.5, 4), '
                 'k = 34, n = 200')

    # ------------------------------------------------------------------
    # 6. Diagnostic + gates
    # ------------------------------------------------------------------
    diag = input_sensitivity_diagnostic(
        predict_q, z_te, aux_te, q_model, q_true, levels, cfg.RNG_DIAG)
    print("\nInput-sensitivity diagnostic:")
    for kk, v in diag.items():
        print(f"  {kk}: {v:.4f}")

    learned = rows[-1]
    sb, moon = rows[0], rows[1]
    best_np = min(sb['w1_truth'], moon['w1_truth'])
    gates = {
        'G1_uses_input': (diag['width_tracking_corr'] > 0.95
                          and diag['noise_response'] > 0.5),
        'G2_calibrated': abs(learned['cov95'] - 0.95) < 0.01,
        'G3_beats_nonparametric': all(
            learned['w1_truth'] < r['w1_truth']
            and abs(learned['cov95'] - 0.95) <= abs(r['cov95'] - 0.95) + 0.002
            for r in (sb, moon)),
        'G4_near_bayes': learned['w1_ref'] < 0.25 * best_np,
    }
    print("Gates:")
    for kk, v in gates.items():
        print(f"  {kk}: {'PASS' if v else 'FAIL'}")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    own_te = T_n - gamma_te
    i_lo = int(np.argmin(np.abs(levels - 0.025)))
    i_hi = int(np.argmin(np.abs(levels - 0.975)))
    out = cfg.RESULTS_DIR / 'm3_pareto_hill.npz'
    save = {
        'levels': levels,
        'params_test': params_test,
        'diag_width_tracking_corr': diag['width_tracking_corr'],
        'diag_noise_response': diag['noise_response'],
        'train_losses': np.array(result['train_losses']),
        'val_losses': np.array(result['val_losses']),
        'tau_adj': tau_adj,
        'test_cov_raw': (own_te[:, None] <= q_model).mean(axis=0),
        'test_cov_recal': (own_te[:, None] <= q_model_recal).mean(axis=0),
        'w_model': (q_model_recal[:, i_hi] - q_model_recal[:, i_lo]).astype(np.float32),
        'w_true': (q_true[:, i_hi] - q_true[:, i_lo]).astype(np.float32),
        'w_bayes': (q_bayes[:, i_hi] - q_bayes[:, i_lo]).astype(np.float32),
        'q_model_recal_head': q_model_recal[:200],
        'q_bayes_head': q_bayes[:200],
        'q_true_head': q_true[:200],
        'q_sboot_head': q_sboot[:200],
    }
    for r in rows:
        key = r['method'].replace('(', '').replace(')', '').replace('=', '')
        for metric in ('cov95', 'cov90', 'len95', 'w1_truth'):
            save[f"{key}_{metric}"] = r[metric]
    save['learned_recal_ours_w1_ref'] = learned['w1_ref']
    np.savez_compressed(out, **save)

    torch.save({'state_dict': model.state_dict(),
                'target_scale': target_scale, 'c0': c0,
                'n_input': n, 'n_levels': len(levels)},
               cfg.RESULTS_DIR / 'm3_pareto_hill_model.pt')
    print(f"\nSaved: {out.name}, m3_pareto_hill_model.pt")


if __name__ == '__main__':
    main()
