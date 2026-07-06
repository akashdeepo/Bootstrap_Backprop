"""
Milestone 4b: one universal model across all four families.

A single quantile network is trained on a mixture of all four families'
examples, with a 4-dim statistic token appended to the auxiliary inputs
(the deployment scenario knows which statistic it is computing). Each
family keeps its own target normalization c0 and its own own-root
recalibration curve (fit on that family's validation split).

Test sets are regenerated with config.fresh_rng(2) so they are
BIT-IDENTICAL to each specialist's test set -- universal vs specialist is
a paired comparison on the same datasets (and the cached NTS test truth
is reused).

Protocol note: the universal model uses asinh input compression for ALL
families (the m1 specialist used raw inputs; a shared input pipeline is
required for a shared network).

Usage:
    python -m amortized_bootstrap.experiments.m4_universal [--epochs 40]
"""

import argparse
import time
import numpy as np
import torch

from .. import config as cfg
from ..families import (UniformMaxFamily, StableMeanFamily,
                        ParetoHillFamily, NTSVaRFamily)
from ..datagen import generate_examples, featurize
from ..model import count_parameters
from ..training import train_quantile_net, predict_root_quantiles
from ..stable_table import build_or_load_table, q_std_interp
from ..nts_truth import (build_or_load_var_grid, make_var_std_fn,
                         build_or_load_test_truth)
from ..evaluation import evaluate_method, print_results_table
from ..calibration import (empirical_coverage_curve, adjusted_levels,
                           apply_recalibration)
from ..export import export_table

N_TRAIN_PER_FAMILY = 200_000
N_VAL_PER_FAMILY = 10_000


def build_families():
    grid = build_or_load_var_grid()
    var_std_fn = make_var_std_fn(grid)
    return [
        ('m1_uniform_max', UniformMaxFamily(), True),   # scale_root
        ('m2_stable_mean', StableMeanFamily(), True),
        ('m3_pareto_hill', ParetoHillFamily(), False),
        ('m3_nts_var', NTSVaRFamily(var_std_fn=var_std_fn), True),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--n-train', type=int, default=N_TRAIN_PER_FAMILY)
    parser.add_argument('--hidden', type=int, default=384)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = args.device

    n = cfg.N
    levels = cfg.QUANTILE_LEVELS
    fams = build_families()
    F = len(fams)

    print("=" * 78)
    print(f"Milestone 4b: universal model, {F} families x "
          f"{args.n_train} train examples, statistic token")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Mixed training/validation data (per-family c0; shared network)
    # ------------------------------------------------------------------
    t0 = time.time()
    z_tr_all, aux_tr_all, y_tr_all = [], [], []
    z_va_all, aux_va_all, y_va_all = [], [], []
    per_family = {}

    for fi, (stem, fam, scale_root) in enumerate(fams):
        X_tr, t_tr, _ = generate_examples(fam, args.n_train, n,
                                          cfg.RNG_M4_TRAIN)
        X_va, t_va, p_va = generate_examples(fam, N_VAL_PER_FAMILY, n,
                                             cfg.RNG_M4_VAL)
        z_tr, aux_tr, s_tr = featurize(X_tr, compress=True)
        z_va, aux_va, s_va = featurize(X_va, compress=True)

        sr_tr = s_tr if scale_root else np.ones(len(X_tr))
        sr_va = s_va if scale_root else np.ones(len(X_va))
        t_std_tr = t_tr / sr_tr
        q25, q75 = np.percentile(t_std_tr, [25.0, 75.0])
        c0 = (q75 - q25) / 1.349

        token = np.zeros((1, F), dtype=np.float32)
        token[0, fi] = 1.0

        z_tr_all.append(z_tr)
        aux_tr_all.append(np.hstack([aux_tr,
                                     np.repeat(token, len(z_tr), axis=0)]))
        y_tr_all.append(np.arcsinh(t_std_tr / c0))
        z_va_all.append(z_va)
        aux_va_all.append(np.hstack([aux_va,
                                     np.repeat(token, len(z_va), axis=0)]))
        y_va_all.append(np.arcsinh((t_va / sr_va) / c0))

        own_va = fam.statistic(X_va) - fam.true_param(p_va)
        per_family[stem] = dict(fam=fam, fi=fi, c0=c0,
                                scale_root=scale_root,
                                z_va=z_va, aux_va=aux_va_all[-1],
                                sr_va=sr_va, own_va=own_va)
        print(f"  [{stem}] c0={c0:.5f} ({time.time()-t0:.0f}s)")

    z_tr = np.concatenate(z_tr_all)
    aux_tr = np.concatenate(aux_tr_all)
    y_tr = np.concatenate(y_tr_all)
    z_va = np.concatenate(z_va_all)
    aux_va = np.concatenate(aux_va_all)
    y_va = np.concatenate(y_va_all)
    del z_tr_all, aux_tr_all, y_tr_all, z_va_all, aux_va_all, y_va_all
    print(f"[1] Mixed data: {len(z_tr)} train, {len(z_va)} val "
          f"({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    print(f"[2] Training on {device} ...")
    result = train_quantile_net(
        z_tr, aux_tr, y_tr, z_va, aux_va, y_va, levels,
        n_epochs=args.epochs, batch_size=512, lr=1e-3, hidden=args.hidden,
        device=device, torch_seed=cfg.TORCH_SEED)
    model = result['model']
    target_scale = result['target_scale']
    print(f"    params={count_parameters(model):,}, "
          f"best val pinball={result['best_val']:.5f}")

    # ------------------------------------------------------------------
    # 3. Per-family evaluation on the SPECIALIST test sets
    # ------------------------------------------------------------------
    stable_table = build_or_load_table()
    all_rows = []
    out_save = {'levels': levels}

    for stem, fam, scale_root in fams:
        info = per_family[stem]
        c0 = info['c0']
        rng_te = cfg.fresh_rng(2)   # bit-identical to the specialist run

        def predict_q(z_in, aux_in, sr_in):
            q_y = predict_root_quantiles(model, z_in, aux_in,
                                         np.ones(len(z_in)), target_scale,
                                         device=device)
            return np.sinh(np.clip(q_y, -20.0, 20.0)) * c0 * sr_in[:, None]

        # -- regenerate the specialist's test set --
        if stem == 'm1_uniform_max':
            theta = fam.sample_params(200, rng_te)
            params = np.repeat(theta, 100)
            X_te = fam.sample_data(params, n, rng_te)
            q_true = fam.true_root_quantiles(params, levels, n)
            true_value = params
            norm = n / params
        elif stem == 'm2_stable_mean':
            p_u = fam.sample_params(200, rng_te)
            params = np.repeat(p_u, 100, axis=0)
            X_te = fam.sample_data(params, n, rng_te)
            scale_n = fam.root_scale(params, n)
            q_true = scale_n[:, None] * q_std_interp(stable_table,
                                                     params[:, 0])
            true_value = params[:, 2]
            norm = 1.0 / scale_n
        elif stem == 'm3_pareto_hill':
            p_u = fam.sample_params(200, rng_te)
            params = np.repeat(p_u, 100, axis=0)
            X_te = fam.sample_data(params, n, rng_te)
            q_true = fam.true_root_quantiles(params, levels)
            true_value = 1.0 / params[:, 0]
            norm = np.sqrt(fam.k) * params[:, 0]
        else:  # m3_nts_var
            p_u = fam.sample_params(100, rng_te)
            params = np.repeat(p_u, 100, axis=0)
            X_te = fam.sample_data(params, n, rng_te)
            truth = build_or_load_test_truth(
                p_u[:, :2], n, levels,
                name="m3_nts_truth_2000000_100000")
            rep = np.repeat(np.arange(100), 100)
            true_value = params[:, 3] + params[:, 2] * truth['t_std'][rep]
            q_true = params[:, 2:3] * truth['q_root_std'][rep]
            i_lo = int(np.argmin(np.abs(levels - 0.025)))
            i_hi = int(np.argmin(np.abs(levels - 0.975)))
            norm = 3.92 / (q_true[:, i_hi] - q_true[:, i_lo])

        T_n = fam.statistic(X_te)
        z_te, aux2_te, s_te = featurize(X_te, compress=True)
        token = np.zeros((len(z_te), F), dtype=np.float32)
        token[:, info['fi']] = 1.0
        aux_te = np.hstack([aux2_te, token])
        sr_te = s_te if scale_root else np.ones(len(X_te))

        q_model = predict_q(z_te, aux_te, sr_te)
        # per-family own-root recalibration from the validation split
        q_val = predict_q(info['z_va'], info['aux_va'], info['sr_va'])
        h = empirical_coverage_curve(q_val, info['own_va'], levels)
        tau_adj = adjusted_levels(levels, h)
        q_recal = apply_recalibration(q_model, levels, tau_adj)

        row = evaluate_method(f'{stem}|universal', q_recal, T_n,
                              true_value, q_true, levels, norm)
        d_spec = np.load(cfg.RESULTS_DIR / f"{stem}.npz")
        spec_row = {
            'method': f'{stem}|specialist',
            'cov95': float(d_spec['learned_recal_ours_cov95']),
            'cov90': float(d_spec['learned_recal_ours_cov90']),
            'len95': float(d_spec['learned_recal_ours_len95']),
            'w1_truth': float(d_spec['learned_recal_ours_w1_truth']),
        }
        all_rows.extend([spec_row, row])
        out_save[f'{stem}_universal_cov95'] = row['cov95']
        out_save[f'{stem}_universal_w1'] = row['w1_truth']
        print(f"  [{stem}] universal cov95={row['cov95']:.3f} "
              f"W1={row['w1_truth']:.4f} | specialist "
              f"cov95={spec_row['cov95']:.3f} W1={spec_row['w1_truth']:.4f}")

    print_results_table(all_rows, 20000, unit='nrm')
    export_table(all_rows, 'm4_universal',
                 'Universal model (statistic token) vs specialists, '
                 'same test sets')

    np.savez_compressed(cfg.RESULTS_DIR / 'm4_universal.npz', **out_save)
    torch.save({'state_dict': model.state_dict(),
                'target_scale': target_scale,
                'c0_per_family': {k: v['c0']
                                  for k, v in per_family.items()}},
               cfg.RESULTS_DIR / 'm4_universal_model.pt')
    print("\nSaved: m4_universal.npz, m4_universal_model.pt")


if __name__ == '__main__':
    main()
