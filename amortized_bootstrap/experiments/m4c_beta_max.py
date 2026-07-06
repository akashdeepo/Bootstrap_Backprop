"""
Milestone 4c: can the network learn a data-conditional RATE?

Training family: X = theta * V, F_V(v) = 1 - (1-v)^b, with BOTH the
endpoint theta AND the contact order b unknown (theta ~ LogU(0.5, 5),
b ~ U(0.4, 2.6)). The max converges at rate n^(-1/b), so the root scale
varies by orders of magnitude across the prior -- the uniform-trained
specialist collapsed to 0.05-0.10 coverage on b != 1 (M4a regime 3).
This experiment tests the prescribed fix: widen the prior over b.

Target transform: the max root is strictly negative and spans ~6 decades,
so targets are trained as y = log(-t_std). Quantiles commute with
monotone maps; the map is DECREASING, so predicted y-quantiles invert
with a level reversal: q_t(tau_j) = -exp(q_y(tau_{L-1-j})).

Gates:
  G1  input sensitivity (log-width tracking across 6 decades)
  G2  coverage calibrated overall
  G3  beats standard bootstrap and subsampling on W1 with coverage
      at least as close to nominal
  G4  RATE LEARNED: coverage within 3pp of nominal on each b slice
      (b in [0.4,0.7], [0.9,1.1], [2.0,2.6]) -- the M4a regime-3 cases
      become within-prior successes

Usage:
    python -m amortized_bootstrap.experiments.m4c_beta_max [--epochs 40]
"""

import argparse
import time
import numpy as np
import torch

from .. import config as cfg
from ..families import BetaMaxPriorFamily
from ..datagen import generate_examples, featurize
from ..training import train_quantile_net, predict_root_quantiles
from ..model import count_parameters
from ..baselines import standard_bootstrap_quantiles, subsampling_quantiles
from ..evaluation import (evaluate_method, print_results_table,
                          input_sensitivity_diagnostic, level_index)
from ..calibration import (empirical_coverage_curve, adjusted_levels,
                           apply_recalibration)
from ..export import export_table

N_VAL = 20_000
N_TEST_PARAMS = 200
N_TEST_DATASETS_PER_PARAM = 100
SUBSAMPLE_M = 34
B_SLICES = [(0.4, 0.7), (0.9, 1.1), (2.0, 2.6)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--n-train', type=int, default=800_000)
    parser.add_argument('--hidden', type=int, default=384)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = args.device

    n = cfg.N
    levels = cfg.QUANTILE_LEVELS
    family = BetaMaxPriorFamily()

    print("=" * 70)
    print("Milestone 4c: beta max with UNKNOWN contact order, "
          "theta ~ LogU(0.5, 5), b ~ U(0.4, 2.6), n = 200")
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
    b_te = params_test[:, 1]
    true_value = family.true_param(params_test)
    print(f"[1] Data: train={args.n_train}, val={N_VAL}, "
          f"test={len(params_test)} ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Featurize; log(-root) targets (root spans ~6 decades over b)
    # ------------------------------------------------------------------
    t0 = time.time()
    z_tr, aux_tr, s_tr = featurize(X_tr)
    z_va, aux_va, s_va = featurize(X_va)
    z_te, aux_te, s_te = featurize(X_te)

    y_tr = np.log(np.maximum(-(t_tr / s_tr), 1e-300))
    y_va = np.log(np.maximum(-(t_va / s_va), 1e-300))
    print(f"[2] Featurized; y range [{y_tr.min():.1f}, {y_tr.max():.1f}] "
          f"({time.time()-t0:.1f}s)")

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

    def predict_q(z_in, aux_in, s_in):
        q_y = predict_root_quantiles(model, z_in, aux_in,
                                     np.ones(len(z_in)), target_scale,
                                     device=device)
        # decreasing map t = -exp(y): reverse levels, then negate-exp
        q_t_std = -np.exp(np.clip(q_y[:, ::-1], -700.0, 50.0))
        return q_t_std * s_in[:, None]

    # ------------------------------------------------------------------
    # 4. Predict, recalibrate (own roots), truth, baselines
    # ------------------------------------------------------------------
    t0 = time.time()
    q_model = predict_q(z_te, aux_te, s_te)

    q_val = predict_q(z_va, aux_va, s_va)
    own_va = family.statistic(X_va) - family.true_param(p_va)
    h_curve = empirical_coverage_curve(q_val, own_va, levels)
    tau_adj = adjusted_levels(levels, h_curve)
    q_model_recal = apply_recalibration(q_model, levels, tau_adj)

    T_n = family.statistic(X_te)
    q_true = family.true_root_quantiles(params_test, levels, n)
    i_lo = level_index(levels, 0.025)
    i_hi = level_index(levels, 0.975)
    norm = 3.92 / np.maximum(q_true[:, i_hi] - q_true[:, i_lo], 1e-300)

    X_te_sorted = np.sort(X_te, axis=1)
    q_sboot = standard_bootstrap_quantiles(X_te_sorted, levels)
    q_subsamp = subsampling_quantiles(X_te_sorted, levels, m=SUBSAMPLE_M)
    print(f"[4] Predictions + baselines ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Evaluate (overall + b slices)
    # ------------------------------------------------------------------
    rows = [
        evaluate_method('standard_bootstrap', q_sboot, T_n, true_value,
                        q_true, levels, norm),
        evaluate_method(f'subsampling_m={SUBSAMPLE_M}', q_subsamp, T_n,
                        true_value, q_true, levels, norm),
        evaluate_method('learned_raw', q_model, T_n, true_value,
                        q_true, levels, norm),
        evaluate_method('learned_recal_(ours)', q_model_recal, T_n,
                        true_value, q_true, levels, norm),
    ]
    print_results_table(rows, len(params_test), unit='1/w')
    export_table(rows, 'm4c_beta_max' + cfg.VTAG,
                 'Bounded max with unknown endpoint contact order, '
                 'b ~ U(0.4, 2.6), n = 200')

    print("\nCoverage by contact-order slice (learned_recal vs baselines):")
    slice_cov = {}
    for lo_b, hi_b in B_SLICES:
        m = (b_te >= lo_b) & (b_te <= hi_b)
        rows_slice = {}
        for nm, q in [('learned_recal', q_model_recal),
                      ('standard_boot', q_sboot),
                      ('subsampling', q_subsamp)]:
            lo_ci = T_n[m] - q[m][:, i_hi]
            hi_ci = T_n[m] - q[m][:, i_lo]
            rows_slice[nm] = float(np.mean((true_value[m] >= lo_ci)
                                           & (true_value[m] <= hi_ci)))
        slice_cov[(lo_b, hi_b)] = rows_slice
        print(f"  b in [{lo_b}, {hi_b}] (N={m.sum()}): "
              f"ours={rows_slice['learned_recal']:.3f}  "
              f"boot={rows_slice['standard_boot']:.3f}  "
              f"subsamp={rows_slice['subsampling']:.3f}")

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
    sb, subsamp = rows[0], rows[1]
    gates = {
        'G1_uses_input': (diag['width_tracking_corr'] > 0.95
                          and diag['noise_response'] > 0.5),
        'G2_calibrated': abs(learned['cov95'] - 0.95) < 0.01,
        'G3_beats_nonparametric': all(
            learned['w1_truth'] < r['w1_truth']
            and abs(learned['cov95'] - 0.95) <= abs(r['cov95'] - 0.95) + 0.002
            for r in (sb, subsamp)),
        'G4_rate_learned': all(
            abs(sc['learned_recal'] - 0.95) < 0.03
            for sc in slice_cov.values()),
    }
    print("Gates:")
    for kk, v in gates.items():
        print(f"  {kk}: {'PASS' if v else 'FAIL'}")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    own_te = T_n - true_value
    out = cfg.RESULTS_DIR / f'm4c_beta_max{cfg.VTAG}.npz'
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
        'q_model_recal_head': q_model_recal[:200],
        'q_true_head': q_true[:200],
        'q_sboot_head': q_sboot[:200],
    }
    for (lo_b, hi_b), sc in slice_cov.items():
        for nm, v in sc.items():
            save[f'slice_{lo_b}_{hi_b}_{nm}'] = v
    for r in rows:
        key = r['method'].replace('(', '').replace(')', '').replace('=', '')
        for metric in ('cov95', 'cov90', 'len95', 'w1_truth'):
            save[f"{key}_{metric}"] = r[metric]
    np.savez_compressed(out, **save)

    torch.save({'state_dict': model.state_dict(),
                'target_scale': target_scale,
                'n_input': n, 'n_levels': len(levels)},
               cfg.RESULTS_DIR / f'm4c_beta_max_model{cfg.VTAG}.pt')
    print(f"\nSaved: {out.name}")


if __name__ == '__main__':
    main()
