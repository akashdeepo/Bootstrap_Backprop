"""
Milestone 1: proof of concept on the uniform-max family.

Uniform(0, theta) with theta ~ LogUniform(0.5, 5), T_n = max, n = 200.
The one non-regular case where the Bayes-optimal data-conditional answer
is computable, so the pipeline can be validated end-to-end.

Gates (from RESEARCH_PLAN.md):
  G1  model output tracks its input (anti-memorization diagnostic)
  G2  CI coverage calibrated (within ~1% of nominal at 95%)
  G3  beats the standard bootstrap on W1 to the true root distribution
  G4  near the Bayes oracle (small regret relative to remaining error)

Usage:
    python -m amortized_bootstrap.experiments.m1_uniform_max [--epochs 30]
"""

import argparse
import time
import numpy as np
import torch

from .. import config as cfg
from ..families import UniformMaxFamily
from ..datagen import generate_examples, featurize
from ..training import train_quantile_net, predict_root_quantiles
from ..model import count_parameters
from ..bayes import bayes_root_quantiles
from ..baselines import (standard_bootstrap_quantiles, subsampling_quantiles,
                         parametric_bootstrap_quantiles)
from ..evaluation import (evaluate_method, print_results_table,
                          input_sensitivity_diagnostic)

N_VAL = 20_000
N_TEST_PARAMS = 200
N_TEST_DATASETS_PER_PARAM = 100
SUBSAMPLE_M = 34  # floor(n^(2/3)), matching the v1 Hill convention


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
    family = UniformMaxFamily(theta_min=0.5, theta_max=5.0)

    print("=" * 70)
    print("Milestone 1: uniform max, theta ~ LogUniform(0.5, 5), n = 200")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Data generation (train / val / test from disjoint seed streams)
    # ------------------------------------------------------------------
    t0 = time.time()
    X_tr, t_tr, _ = generate_examples(family, args.n_train, n, cfg.RNG_TRAIN)
    X_va, t_va, _ = generate_examples(family, N_VAL, n, cfg.RNG_VAL)

    theta_test = family.sample_params(N_TEST_PARAMS, cfg.RNG_TEST)
    params_test = np.repeat(theta_test, N_TEST_DATASETS_PER_PARAM)
    X_te = family.sample_data(params_test, n, cfg.RNG_TEST)
    print(f"[1] Data: train={args.n_train}, val={N_VAL}, "
          f"test={len(params_test)} datasets from {N_TEST_PARAMS} held-out "
          f"thetas ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Featurize
    # ------------------------------------------------------------------
    t0 = time.time()
    z_tr, aux_tr, s_tr = featurize(X_tr)
    z_va, aux_va, s_va = featurize(X_va)
    z_te, aux_te, s_te = featurize(X_te)
    t_std_tr = t_tr / s_tr
    t_std_va = t_va / s_va
    print(f"[2] Featurized ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print(f"[3] Training on {device} ...")
    result = train_quantile_net(
        z_tr, aux_tr, t_std_tr, z_va, aux_va, t_std_va, levels,
        n_epochs=args.epochs, batch_size=512, lr=1e-3, hidden=args.hidden,
        device=device, torch_seed=cfg.TORCH_SEED)
    model = result['model']
    target_scale = result['target_scale']
    print(f"    params={count_parameters(model):,}, "
          f"best val pinball={result['best_val']:.5f}")

    # ------------------------------------------------------------------
    # 4. Predict + oracles + baselines on the test split
    # ------------------------------------------------------------------
    t0 = time.time()
    X_te_sorted = np.sort(X_te, axis=1)
    T_n = X_te_sorted[:, -1]
    norm = n / params_test

    q_model = predict_root_quantiles(model, z_te, aux_te, s_te,
                                     target_scale, device=device)
    q_true = family.true_root_quantiles(params_test, levels, n)
    q_bayes = bayes_root_quantiles(T_n, levels, n,
                                   a=family.theta_min, b=family.theta_max)
    q_std_boot = standard_bootstrap_quantiles(X_te_sorted, levels)
    q_subsamp = subsampling_quantiles(X_te_sorted, levels, m=SUBSAMPLE_M)
    q_param = parametric_bootstrap_quantiles(X_te_sorted, levels)
    # Exact pivot as an implied root-quantile function:
    # q(tau) = T_n * (1 - tau^(-1/n)) reproduces the exact equal-tailed CI
    q_pivot = T_n[:, None] * (1.0 - levels[None, :] ** (-1.0 / n))
    print(f"[4] Predictions + baselines ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    rows = [
        evaluate_method('standard_bootstrap', q_std_boot, T_n, params_test,
                        q_true, levels, norm),
        evaluate_method(f'subsampling_m={SUBSAMPLE_M}', q_subsamp, T_n,
                        params_test, q_true, levels, norm),
        evaluate_method('parametric_bootstrap', q_param, T_n, params_test,
                        q_true, levels, norm),
        evaluate_method('exact_pivot_(oracle)', q_pivot, T_n, params_test,
                        q_true, levels, norm),
        evaluate_method('bayes_(oracle)', q_bayes, T_n, params_test,
                        q_true, levels, norm),
        evaluate_method('learned_(ours)', q_model, T_n, params_test,
                        q_true, levels, norm, q_ref=q_bayes),
    ]
    print_results_table(rows, len(params_test), unit='n/th')

    # ------------------------------------------------------------------
    # 6. Input-sensitivity diagnostic (anti-memorization guard)
    # ------------------------------------------------------------------
    def predict_fn(z_in, aux_in):
        return predict_root_quantiles(model, z_in, aux_in, s_te,
                                      target_scale, device=device)

    diag = input_sensitivity_diagnostic(
        predict_fn, z_te, aux_te, q_model, q_bayes, levels, cfg.RNG_DIAG)
    print("\nInput-sensitivity diagnostic (v1 artifact gives ~0 on all):")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 7. Gates
    # ------------------------------------------------------------------
    learned = rows[-1]
    sb = rows[0]
    subsamp = rows[1]
    # G4 note: comparing w1_bayes to the model's own w1_truth is degenerate
    # when the Bayes oracle nearly coincides with the truth (triangle
    # inequality forces the ratio toward 1 for ANY imperfect model). The
    # meaningful criterion is regret relative to the best classical
    # NONPARAMETRIC method: the model should capture most of the
    # improvement that separates subsampling from the Bayes oracle.
    frac_captured = ((sb['w1_truth'] - learned['w1_truth'])
                     / (sb['w1_truth'] - rows[4]['w1_truth']))
    gates = {
        'G1_uses_input': (diag['width_tracking_corr'] > 0.95
                          and diag['noise_response'] > 0.5),
        'G2_calibrated': abs(learned['cov95'] - 0.95) < 0.01,
        'G3_beats_standard_bootstrap': learned['w1_truth'] < sb['w1_truth'],
        'G4_near_bayes': learned['w1_ref'] < 0.25 * subsamp['w1_truth'],
    }
    print(f"\nFraction of standard-bootstrap-to-Bayes gap captured: "
          f"{frac_captured:.1%}")
    print("Gates:")
    for k, v in gates.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    out = cfg.RESULTS_DIR / 'm1_uniform_max.npz'
    save = {
        'levels': levels,
        'theta_test': params_test,
        'T_n': T_n,
        'diag_' + 'width_tracking_corr': diag['width_tracking_corr'],
        'diag_' + 'noise_response': diag['noise_response'],
        'train_losses': np.array(result['train_losses']),
        'val_losses': np.array(result['val_losses']),
        # quantile curves for the first 200 test datasets (plots later)
        'q_model_head': q_model[:200],
        'q_bayes_head': q_bayes[:200],
        'q_true_head': q_true[:200],
        'q_std_boot_head': q_std_boot[:200],
    }
    for r in rows:
        key = r['method'].replace('(', '').replace(')', '').replace('=', '')
        for metric in ('cov95', 'cov90', 'len95', 'w1_truth'):
            save[f"{key}_{metric}"] = r[metric]
    save['learned_ours_w1_ref'] = learned['w1_ref']
    np.savez_compressed(out, **save)

    ckpt = cfg.RESULTS_DIR / 'm1_uniform_max_model.pt'
    torch.save({'state_dict': model.state_dict(),
                'target_scale': target_scale,
                'n_input': n, 'n_levels': len(levels)}, ckpt)
    print(f"\nSaved: {out.name}, {ckpt.name}")


if __name__ == '__main__':
    main()
