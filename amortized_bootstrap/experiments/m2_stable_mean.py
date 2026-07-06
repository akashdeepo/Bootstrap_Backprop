"""
Milestone 2: stable mean with unknown alpha -- the Athreya case.

X ~ mu + sigma * SaS(alpha), alpha ~ U(1.1, 1.95), sigma ~ LogUniform(0.5, 5),
mu ~ U(-2, 2), T_n = mean, n = 200. The standard bootstrap is inconsistent
(Athreya 1987); the root rate n^(1-1/alpha) depends on the unknown alpha, so
rate-corrected subsampling requires estimating alpha -- the classical pain
point this milestone targets.

Heavy-tail handling: inputs are asinh-compressed order statistics; targets
are trained as y = asinh(t_std / c0) (quantiles commute with monotone maps,
so predicted y-quantiles invert exactly to root quantiles).

Gates:
  G1  model output tracks its input (width correlation vs truth, noise test)
  G2  CI coverage calibrated at 95%
  G3  beats every feasible nonparametric method (standard bootstrap,
      m-out-of-n with naive and with McCulloch-estimated rates) on W1,
      with coverage at least as close to nominal
  G4  within 1.5x of the parametric-stable oracle (McCulloch + stability),
      the strongest feasible classical method WHEN the family is known

Usage:
    python -m amortized_bootstrap.experiments.m2_stable_mean [--epochs 40]
"""

import argparse
import time
import numpy as np
import torch

from .. import config as cfg
from ..families import StableMeanFamily
from ..datagen import generate_examples, featurize
from ..training import train_quantile_net, predict_root_quantiles
from ..model import count_parameters
from ..stable_table import (build_or_load_table, q_std_interp,
                            mcculloch_estimate, parametric_stable_quantiles)
from ..baselines_mean import (normal_interval_quantiles,
                              standard_bootstrap_mean_quantiles,
                              m_out_of_n_mean_quantiles, rate_corrected)
from ..evaluation import (evaluate_method, print_results_table,
                          input_sensitivity_diagnostic)
from ..calibration import (empirical_coverage_curve, adjusted_levels,
                           apply_recalibration)

N_VAL = 20_000
N_TEST_PARAMS = 200
N_TEST_DATASETS_PER_PARAM = 100
SUBSAMPLE_M = 34


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--n-train', type=int, default=800_000)
    parser.add_argument('--hidden', type=int, default=384)
    parser.add_argument('--table-draws', type=int, default=20_000_000)
    parser.add_argument('--b-boot', type=int, default=2000)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = args.device

    n = cfg.N
    levels = cfg.QUANTILE_LEVELS
    family = StableMeanFamily()

    print("=" * 70)
    print("Milestone 2: stable mean, alpha ~ U(1.1, 1.95), "
          "sigma ~ LogU(0.5, 5), n = 200")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 0. Ground-truth quantile table (one-time, cached)
    # ------------------------------------------------------------------
    table = build_or_load_table(n_draws=args.table_draws)

    # ------------------------------------------------------------------
    # 1. Data generation (disjoint seed streams)
    # ------------------------------------------------------------------
    t0 = time.time()
    X_tr, t_tr, _ = generate_examples(family, args.n_train, n, cfg.RNG_TRAIN)
    X_va, t_va, p_va = generate_examples(family, N_VAL, n, cfg.RNG_VAL)

    p_test_unique = family.sample_params(N_TEST_PARAMS, cfg.RNG_TEST)
    params_test = np.repeat(p_test_unique, N_TEST_DATASETS_PER_PARAM, axis=0)
    X_te = family.sample_data(params_test, n, cfg.RNG_TEST)
    alpha_te = params_test[:, 0]
    mu_te = params_test[:, 2]
    print(f"[1] Data: train={args.n_train}, val={N_VAL}, "
          f"test={len(params_test)} datasets from {N_TEST_PARAMS} held-out "
          f"(alpha, sigma, mu) ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Featurize (asinh-compressed) + target transform
    # ------------------------------------------------------------------
    t0 = time.time()
    z_tr, aux_tr, s_tr = featurize(X_tr, compress=True)
    z_va, aux_va, s_va = featurize(X_va, compress=True)
    z_te, aux_te, s_te = featurize(X_te, compress=True)

    t_std_tr = t_tr / s_tr
    t_std_va = t_va / s_va
    q25, q75 = np.percentile(t_std_tr, [25.0, 75.0])
    c0 = (q75 - q25) / 1.349  # robust scale of the standardized root
    y_tr = np.arcsinh(t_std_tr / c0)
    y_va = np.arcsinh(t_std_va / c0)
    print(f"[2] Featurized; target c0={c0:.5f} ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Train (targets in asinh space)
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

    ones = np.ones(len(z_te))

    def invert_target(q_y, s):
        # asinh-space quantiles -> root quantiles. |q_y| ~ <= 8 for any
        # trained model; the clip only guards untrained/garbage inputs
        # (e.g. the noise diagnostic) against sinh overflow.
        return np.sinh(np.clip(q_y, -20.0, 20.0)) * c0 * s[:, None]

    def predict_fn(z_in, aux_in):
        q_y = predict_root_quantiles(model, z_in, aux_in,
                                     np.ones(len(z_in)), target_scale,
                                     device=device)
        return invert_target(q_y, s_te[:len(z_in)])

    # ------------------------------------------------------------------
    # 4. Predictions, truth, baselines
    # ------------------------------------------------------------------
    t0 = time.time()
    q_y_te = predict_root_quantiles(model, z_te, aux_te, ones,
                                    target_scale, device=device)
    q_model = invert_target(q_y_te, s_te)

    # Post-hoc recalibration, fit on the VALIDATION split (PIT is invariant
    # to the monotone asinh transform, so fit directly in y space).
    #
    # IMPORTANT: fit against each validation dataset's OWN root
    # (T_n(X) - mu), not the independent replicate t used for training.
    # CI coverage is a statement about the own root, which is dependent on
    # the prediction through X; marginal PIT calibration against
    # independent replicates does not target the coverage functional
    # (Loh 1987-style simulation calibration does).
    q_y_val = predict_root_quantiles(model, z_va, aux_va,
                                     np.ones(len(z_va)), target_scale,
                                     device=device)
    t_own_va = family.statistic(X_va) - family.true_param(p_va)
    y_own_va = np.arcsinh((t_own_va / s_va) / c0)
    h_curve = empirical_coverage_curve(q_y_val, y_own_va, levels)
    tau_adj = adjusted_levels(levels, h_curve)
    q_model_recal = apply_recalibration(q_model, levels, tau_adj)
    print(f"    recalibration: tau_adj(0.025)={np.interp(0.025, levels, tau_adj):.4f}, "
          f"tau_adj(0.975)={np.interp(0.975, levels, tau_adj):.4f}")

    root_scale_true = family.root_scale(params_test, n)
    q_true = root_scale_true[:, None] * q_std_interp(table, alpha_te)
    norm = 1.0 / root_scale_true
    T_n = np.mean(X_te, axis=1)

    print("    computing bootstrap baselines (MC resampling)...")
    q_normal = normal_interval_quantiles(X_te, levels)
    q_sboot = standard_bootstrap_mean_quantiles(X_te, levels,
                                                B=args.b_boot,
                                                rng=cfg.RNG_BASELINE)
    q_moon_raw = m_out_of_n_mean_quantiles(X_te, levels, m=SUBSAMPLE_M,
                                           B=args.b_boot,
                                           rng=cfg.RNG_BASELINE)
    alpha_hat, _ = mcculloch_estimate(X_te, table)
    q_moon_naive = rate_corrected(q_moon_raw, SUBSAMPLE_M, n,
                                  np.full(len(X_te), 2.0))
    q_moon_oracle = rate_corrected(q_moon_raw, SUBSAMPLE_M, n, alpha_te)
    q_moon_mcc = rate_corrected(q_moon_raw, SUBSAMPLE_M, n, alpha_hat)
    q_param = parametric_stable_quantiles(X_te, table, n)
    print(f"[4] Predictions + baselines ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    rows = [
        evaluate_method('normal_CLT_interval', q_normal, T_n, mu_te,
                        q_true, levels, norm),
        evaluate_method('standard_bootstrap', q_sboot, T_n, mu_te,
                        q_true, levels, norm),
        evaluate_method(f'm_of_n_naive_rate_m{SUBSAMPLE_M}', q_moon_naive,
                        T_n, mu_te, q_true, levels, norm),
        evaluate_method(f'm_of_n_mcculloch_m{SUBSAMPLE_M}', q_moon_mcc,
                        T_n, mu_te, q_true, levels, norm),
        evaluate_method(f'm_of_n_oracle_rate_m{SUBSAMPLE_M}', q_moon_oracle,
                        T_n, mu_te, q_true, levels, norm),
        evaluate_method('parametric_stable_(ref)', q_param, T_n, mu_te,
                        q_true, levels, norm),
        evaluate_method('learned_raw', q_model, T_n, mu_te,
                        q_true, levels, norm, q_ref=q_param),
        evaluate_method('learned_recal_(ours)', q_model_recal, T_n, mu_te,
                        q_true, levels, norm, q_ref=q_param),
    ]
    print_results_table(rows, len(params_test), unit='1/sc')

    # ------------------------------------------------------------------
    # 6. Input-sensitivity diagnostic
    # ------------------------------------------------------------------
    diag = input_sensitivity_diagnostic(
        predict_fn, z_te, aux_te, q_model, q_true, levels, cfg.RNG_DIAG)
    print("\nInput-sensitivity diagnostic (v1 artifact gives ~0 on all):")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}")

    # alpha-recovery probe: does the model's predicted width track alpha?
    i_lo = int(np.argmin(np.abs(levels - 0.025)))
    i_hi = int(np.argmin(np.abs(levels - 0.975)))
    w_model = np.log(q_model[:, i_hi] - q_model[:, i_lo])
    w_true = np.log(q_true[:, i_hi] - q_true[:, i_lo])
    print(f"  log-width corr with truth (per-dataset): "
          f"{np.corrcoef(w_model, w_true)[0, 1]:.4f}")

    # ------------------------------------------------------------------
    # 7. Gates
    # ------------------------------------------------------------------
    learned = rows[-1]  # recalibrated
    feasible_np = [rows[1], rows[2], rows[3]]  # standard, naive, mcculloch
    param_ref = rows[5]
    gates = {
        'G1_uses_input': (diag['width_tracking_corr'] > 0.95
                          and diag['noise_response'] > 0.5),
        'G2_calibrated': abs(learned['cov95'] - 0.95) < 0.01,
        'G3_beats_nonparametric': all(
            learned['w1_truth'] < r['w1_truth']
            and abs(learned['cov95'] - 0.95) <= abs(r['cov95'] - 0.95) + 0.002
            for r in feasible_np),
        'G4_matches_parametric_oracle':
            learned['w1_truth'] < 1.5 * param_ref['w1_truth'],
    }
    print("\nGates:")
    for k, v in gates.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    out = cfg.RESULTS_DIR / 'm2_stable_mean.npz'
    save = {
        'levels': levels,
        'params_test': params_test,
        'alpha_hat_mcculloch': alpha_hat,
        'diag_width_tracking_corr': diag['width_tracking_corr'],
        'diag_noise_response': diag['noise_response'],
        'train_losses': np.array(result['train_losses']),
        'val_losses': np.array(result['val_losses']),
        'tau_adj': tau_adj,
        'q_model_head': q_model[:200],
        'q_model_recal_head': q_model_recal[:200],
        'q_true_head': q_true[:200],
        'q_param_head': q_param[:200],
        'q_sboot_head': q_sboot[:200],
    }
    for r in rows:
        key = (r['method'].replace('(', '').replace(')', '')
               .replace('=', ''))
        for metric in ('cov95', 'cov90', 'len95', 'w1_truth'):
            save[f"{key}_{metric}"] = r[metric]
    save['learned_recal_ours_w1_ref'] = learned['w1_ref']
    np.savez_compressed(out, **save)

    ckpt = cfg.RESULTS_DIR / 'm2_stable_mean_model.pt'
    torch.save({'state_dict': model.state_dict(),
                'target_scale': target_scale, 'c0': c0,
                'n_input': n, 'n_levels': len(levels)}, ckpt)
    print(f"\nSaved: {out.name}, {ckpt.name}")


if __name__ == '__main__':
    main()
