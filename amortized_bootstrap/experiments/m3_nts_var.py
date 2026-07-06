"""
Milestone 3b: VaR_0.99 on the NTS family -- the "no comfortable parametric
bootstrap" case.

X ~ mu + sigma * NTS_std(alpha, theta), alpha ~ U(1.1, 1.9),
theta ~ LogU(0.3, 3), sigma ~ LogU(0.5, 5), mu ~ U(-2, 2), T_n = empirical
VaR_0.99, n = 200 (only ~2 expected observations beyond the quantile).

No closed-form T(F): training-target centering uses the precomputed
VaR_std(alpha, theta) MC grid (location-scale reduction); test truth uses
fresh 2M-draw pools per held-out parameter. The classical showstopper
here: the distribution-free order-statistic CI CANNOT reach 95% coverage
at p=0.99, n=200 (achievable max = 1 - 0.99^200 = 0.866).

Usage:
    python -m amortized_bootstrap.experiments.m3_nts_var [--epochs 40]
"""

import argparse
import time
import numpy as np
import torch

from .. import config as cfg
from ..families import NTSVaRFamily
from ..datagen import generate_examples, featurize
from ..training import train_quantile_net, predict_root_quantiles
from ..model import count_parameters
from ..nts_truth import (build_or_load_var_grid, make_var_std_fn,
                         build_or_load_test_truth)
from ..baselines_m3 import bootstrap_quantile_roots, binomial_exact_ci
from ..evaluation import (evaluate_method, evaluate_direct_ci,
                          print_results_table,
                          input_sensitivity_diagnostic)
from ..calibration import (empirical_coverage_curve, adjusted_levels,
                           apply_recalibration)
from ..export import export_table

N_VAL = 20_000
N_TEST_PARAMS = 100
N_TEST_DATASETS_PER_PARAM = 100
SUBSAMPLE_M = 100   # m = n/2; smaller m degenerates for p = 0.99


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--n-train', type=int, default=800_000)
    parser.add_argument('--hidden', type=int, default=384)
    parser.add_argument('--b-boot', type=int, default=2000)
    parser.add_argument('--grid-draws', type=int, default=2_000_000)
    parser.add_argument('--truth-pool', type=int, default=2_000_000)
    parser.add_argument('--truth-reps', type=int, default=100_000)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = args.device

    n = cfg.N
    levels = cfg.QUANTILE_LEVELS

    print("=" * 70)
    print("Milestone 3b: NTS VaR_0.99, alpha ~ U(1.1, 1.9), "
          "theta ~ LogU(0.3, 3), n = 200")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 0. VaR_std grid (training-target centering) + family
    # ------------------------------------------------------------------
    grid = build_or_load_var_grid(n_draws=args.grid_draws)
    var_std_fn = make_var_std_fn(grid)
    family = NTSVaRFamily(var_std_fn=var_std_fn)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    t0 = time.time()
    X_tr, t_tr, _ = generate_examples(family, args.n_train, n, cfg.RNG_TRAIN)
    X_va, t_va, p_va = generate_examples(family, N_VAL, n, cfg.RNG_VAL)

    p_test_unique = family.sample_params(N_TEST_PARAMS, cfg.RNG_TEST)
    params_test = np.repeat(p_test_unique, N_TEST_DATASETS_PER_PARAM, axis=0)
    X_te = family.sample_data(params_test, n, cfg.RNG_TEST)
    sigma_te = params_test[:, 2]
    print(f"[1] Data: train={args.n_train}, val={N_VAL}, "
          f"test={len(params_test)} ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Test truth: fresh pools per held-out (alpha, theta), then
    #    location-scale to every test dataset
    # ------------------------------------------------------------------
    truth = build_or_load_test_truth(
        p_test_unique[:, :2], n, levels,
        name=f"m3_nts_truth_{args.truth_pool}_{args.truth_reps}",
        pool_size=args.truth_pool, n_replicates=args.truth_reps)
    rep = np.repeat(np.arange(N_TEST_PARAMS), N_TEST_DATASETS_PER_PARAM)
    true_value = (params_test[:, 3]
                  + sigma_te * truth['t_std'][rep])
    q_true = sigma_te[:, None] * truth['q_root_std'][rep]

    # ------------------------------------------------------------------
    # 3. Featurize + train (asinh protocol, root scales with sigma -> s)
    # ------------------------------------------------------------------
    t0 = time.time()
    z_tr, aux_tr, s_tr = featurize(X_tr, compress=True)
    z_va, aux_va, s_va = featurize(X_va, compress=True)
    z_te, aux_te, s_te = featurize(X_te, compress=True)

    t_std_tr = t_tr / s_tr
    t_std_va = t_va / s_va
    q25, q75 = np.percentile(t_std_tr, [25.0, 75.0])
    c0 = (q75 - q25) / 1.349
    y_tr = np.arcsinh(t_std_tr / c0)
    y_va = np.arcsinh(t_std_va / c0)
    print(f"[2] Featurized; c0={c0:.5f} ({time.time()-t0:.1f}s)")

    print(f"[3] Training on {device} ...")
    result = train_quantile_net(
        z_tr, aux_tr, y_tr, z_va, aux_va, y_va, levels,
        n_epochs=args.epochs, batch_size=512, lr=1e-3, hidden=args.hidden,
        device=device, torch_seed=cfg.TORCH_SEED)
    model = result['model']
    target_scale = result['target_scale']
    print(f"    params={count_parameters(model):,}, "
          f"best val pinball={result['best_val']:.5f}")

    def predict_q(z_in, aux_in, s_in):
        q_y = predict_root_quantiles(model, z_in, aux_in,
                                     np.ones(len(z_in)), target_scale,
                                     device=device)
        return np.sinh(np.clip(q_y, -20.0, 20.0)) * c0 * s_in[:, None]

    # ------------------------------------------------------------------
    # 4. Predict, recalibrate (own roots), baselines
    # ------------------------------------------------------------------
    t0 = time.time()
    q_model = predict_q(z_te, aux_te, s_te)

    q_val = predict_q(z_va, aux_va, s_va)
    own_va = family.statistic(X_va) - family.true_param(p_va)
    h_curve = empirical_coverage_curve(q_val, own_va, levels)
    tau_adj = adjusted_levels(levels, h_curve)
    q_model_recal = apply_recalibration(q_model, levels, tau_adj)

    T_n = family.statistic(X_te)
    w_true = q_true[:, -1] - q_true[:, 0]
    norm = 3.92 / (q_true[:, int(np.argmin(np.abs(levels - 0.975)))]
                   - q_true[:, int(np.argmin(np.abs(levels - 0.025)))])

    print("    bootstrap baselines...")
    q_sboot = bootstrap_quantile_roots(X_te, levels, family.var_level,
                                       B=args.b_boot, rng=cfg.RNG_BASELINE)
    q_moon_raw = bootstrap_quantile_roots(X_te, levels, family.var_level,
                                          B=args.b_boot,
                                          rng=cfg.RNG_BASELINE,
                                          m=SUBSAMPLE_M)
    q_moon = q_moon_raw * np.sqrt(SUBSAMPLE_M / n)
    X_te_sorted = np.sort(X_te, axis=1)
    lo95, hi95, cov_exact95 = binomial_exact_ci(X_te_sorted,
                                                family.var_level, 0.05)
    lo90, hi90, cov_exact90 = binomial_exact_ci(X_te_sorted,
                                                family.var_level, 0.10)
    print(f"    binomial exact CI: max achievable two-sided coverage "
          f"{cov_exact95:.3f} (target 0.95)")
    print(f"[4] Predictions + baselines ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    rows = [
        evaluate_method('standard_bootstrap', q_sboot, T_n, true_value,
                        q_true, levels, norm),
        evaluate_method(f'm_of_n_m{SUBSAMPLE_M}', q_moon, T_n, true_value,
                        q_true, levels, norm),
        evaluate_direct_ci('binomial_exact_orderstat', lo95, hi95,
                           lo90, hi90, true_value, norm),
        evaluate_method('learned_raw', q_model, T_n, true_value,
                        q_true, levels, norm),
        evaluate_method('learned_recal_(ours)', q_model_recal, T_n,
                        true_value, q_true, levels, norm),
    ]
    print_results_table(rows, len(params_test), unit='1/w')
    export_table(rows, 'm3_nts_var',
                 'VaR 0.99, NTS family, alpha ~ U(1.1, 1.9), n = 200')

    # ------------------------------------------------------------------
    # 6. Diagnostic + gates
    # ------------------------------------------------------------------
    diag = input_sensitivity_diagnostic(
        lambda z_in, aux_in: predict_q(z_in, aux_in, s_te),
        z_te, aux_te, q_model, q_true, levels, cfg.RNG_DIAG)
    print("\nInput-sensitivity diagnostic:")
    for kk, v in diag.items():
        print(f"  {kk}: {v:.4f}")

    learned = rows[-1]
    sb, moon, binom_row = rows[0], rows[1], rows[2]
    best_np_w1 = min(sb['w1_truth'], moon['w1_truth'])
    classical_devs = [abs(r['cov95'] - 0.95) for r in (sb, moon, binom_row)]
    gates = {
        'G1_uses_input': (diag['width_tracking_corr'] > 0.95
                          and diag['noise_response'] > 0.5),
        'G2_calibrated': abs(learned['cov95'] - 0.95) < 0.01,
        'G3_beats_nonparametric': all(
            learned['w1_truth'] < r['w1_truth']
            and abs(learned['cov95'] - 0.95) <= abs(r['cov95'] - 0.95) + 0.002
            for r in (sb, moon)),
        'G4_dominates_classical': (
            abs(learned['cov95'] - 0.95) < min(classical_devs)
            and learned['w1_truth'] < 0.5 * best_np_w1),
    }
    print("Gates:")
    for kk, v in gates.items():
        print(f"  {kk}: {'PASS' if v else 'FAIL'}")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    own_te = T_n - true_value
    i_lo = int(np.argmin(np.abs(levels - 0.025)))
    i_hi = int(np.argmin(np.abs(levels - 0.975)))
    out = cfg.RESULTS_DIR / 'm3_nts_var.npz'
    save = {
        'levels': levels,
        'params_test': params_test,
        'true_value': true_value,
        'diag_width_tracking_corr': diag['width_tracking_corr'],
        'diag_noise_response': diag['noise_response'],
        'train_losses': np.array(result['train_losses']),
        'val_losses': np.array(result['val_losses']),
        'tau_adj': tau_adj,
        'binomial_max_cov95': cov_exact95,
        'test_cov_raw': (own_te[:, None] <= q_model).mean(axis=0),
        'test_cov_recal': (own_te[:, None] <= q_model_recal).mean(axis=0),
        'w_model': (q_model_recal[:, i_hi] - q_model_recal[:, i_lo]).astype(np.float32),
        'w_true': (q_true[:, i_hi] - q_true[:, i_lo]).astype(np.float32),
        'q_model_recal_head': q_model_recal[:200],
        'q_true_head': q_true[:200],
        'q_sboot_head': q_sboot[:200],
    }
    for r in rows:
        key = r['method'].replace('(', '').replace(')', '').replace('=', '')
        for metric in ('cov95', 'cov90', 'len95', 'w1_truth'):
            save[f"{key}_{metric}"] = r[metric]
    np.savez_compressed(out, **save)

    torch.save({'state_dict': model.state_dict(),
                'target_scale': target_scale, 'c0': c0,
                'n_input': n, 'n_levels': len(levels)},
               cfg.RESULTS_DIR / 'm3_nts_var_model.pt')
    print(f"\nSaved: {out.name}, m3_nts_var_model.pt")


if __name__ == '__main__':
    main()
