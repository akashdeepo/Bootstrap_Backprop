"""
Milestone 4a: out-of-family (OOD) stress tests.

Each trained specialist is evaluated, UNCHANGED (including its in-family
recalibration curve), on data from distributions outside its training
prior. This is the honest successor to v1's LOO experiment: it measures
how far the learned corrections transfer, with degradation reported as a
finding either way.

OOD map (see ood_families.py for the rationale):
  m1 uniform_max  -> beta_max b=0.5, b=2      (analytic truth)
  m2 stable_mean  -> t_mean heavy (nu 1.2-1.9), t_mean light (nu 2.5-8)
  m3 pareto_hill  -> burr_hill, frechet_hill
  m3 nts_var99    -> t_var99

Usage:
    python -m amortized_bootstrap.experiments.m4_ood [--b-boot 1000]
"""

import argparse
import time
import numpy as np
import torch

from .. import config as cfg
from ..datagen import featurize
from ..model import QuantileNet
from ..training import predict_root_quantiles
from ..ood_families import (BetaMaxFamily, StudentTMeanFamily,
                            BurrHillFamily, FrechetHillFamily,
                            StudentTVaRFamily)
from ..ood_truth import mc_root_truth, bootstrap_roots_generic
from ..evaluation import evaluate_method, print_results_table
from ..calibration import apply_recalibration
from ..export import export_table

N_OOD_PARAMS = 50
N_DATASETS_PER_PARAM = 100

# Specialist pipelines: (results npz stem, compress inputs, uses asinh
# target transform, root scales with data scale)
SPECIALISTS = {
    'm1_uniform_max': dict(compress=False, asinh=False, scale_root=True),
    'm2_stable_mean': dict(compress=True, asinh=True, scale_root=True),
    'm3_pareto_hill': dict(compress=True, asinh=True, scale_root=False),
    'm3_nts_var': dict(compress=True, asinh=True, scale_root=True),
}

OOD_MAP = [
    ('m1_uniform_max', BetaMaxFamily(b=0.5)),
    ('m1_uniform_max', BetaMaxFamily(b=2.0)),
    ('m2_stable_mean', StudentTMeanFamily(1.2, 1.9, 'heavy')),
    ('m2_stable_mean', StudentTMeanFamily(2.5, 8.0, 'light')),
    ('m3_pareto_hill', BurrHillFamily()),
    ('m3_pareto_hill', FrechetHillFamily()),
    ('m3_nts_var', StudentTVaRFamily()),
]


def load_specialist(stem: str, device: str):
    # our own checkpoints; they contain numpy scalars (c0), so full load
    ckpt = torch.load(cfg.RESULTS_DIR / f"{stem}_model.pt",
                      map_location=device, weights_only=False)
    model = QuantileNet(n_input=cfg.N, n_aux=2,
                        n_levels=len(cfg.QUANTILE_LEVELS),
                        hidden=384, depth=3).to(device)  # chain runs: 384
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    d = np.load(cfg.RESULTS_DIR / f"{stem}.npz")
    return {
        'model': model,
        'target_scale': float(ckpt['target_scale']),
        'c0': float(ckpt['c0']) if 'c0' in ckpt else None,
        'tau_adj': d['tau_adj'],
        'ref_cov95': float(d['learned_recal_ours_cov95']),
        'ref_w1': float(d['learned_recal_ours_w1_truth']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b-boot', type=int, default=1000)
    parser.add_argument('--n-params', type=int, default=N_OOD_PARAMS)
    parser.add_argument('--truth-pool', type=int, default=2_000_000)
    parser.add_argument('--truth-reps', type=int, default=100_000)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = args.device

    n = cfg.N
    levels = cfg.QUANTILE_LEVELS
    all_rows = []

    print("=" * 78)
    print("Milestone 4a: out-of-family stress tests "
          f"({args.n_params} params x {N_DATASETS_PER_PARAM} datasets each)")
    print("=" * 78)

    for stem, fam in OOD_MAP:
        spec_cfg = SPECIALISTS[stem]
        spec = load_specialist(stem, device)
        t0 = time.time()

        params_u = fam.sample_params(args.n_params, cfg.RNG_M4_OOD)
        params = np.repeat(params_u, N_DATASETS_PER_PARAM, axis=0)
        X = fam.sample_data(params, n, cfg.RNG_M4_OOD)
        true_value = fam.true_param(params)
        T_n = fam.statistic(X)

        if hasattr(fam, 'true_root_quantiles'):
            q_true = fam.true_root_quantiles(params, levels, n)
        else:
            q_root_u = mc_root_truth(fam, params_u, n, levels,
                                     name=f"m4_truth_{fam.name}",
                                     rng=cfg.RNG_M4_POOL,
                                     pool_size=args.truth_pool,
                                     n_replicates=args.truth_reps)
            rep = np.repeat(np.arange(args.n_params), N_DATASETS_PER_PARAM)
            q_true = q_root_u[rep]

        z, aux, s = featurize(X, compress=spec_cfg['compress'])
        s_root = s if spec_cfg['scale_root'] else np.ones(len(X))

        q_y = predict_root_quantiles(spec['model'], z, aux,
                                     np.ones(len(z)), spec['target_scale'],
                                     device=device)
        if spec_cfg['asinh']:
            q_model = np.sinh(np.clip(q_y, -20.0, 20.0)) \
                * spec['c0'] * s_root[:, None]
        else:
            q_model = q_y * s_root[:, None]
        q_model_recal = apply_recalibration(q_model, levels,
                                            spec['tau_adj'])

        q_sboot = bootstrap_roots_generic(X, fam.statistic, levels,
                                          B=args.b_boot,
                                          rng=cfg.RNG_BASELINE)

        i_lo = int(np.argmin(np.abs(levels - 0.025)))
        i_hi = int(np.argmin(np.abs(levels - 0.975)))
        norm = 3.92 / np.maximum(q_true[:, i_hi] - q_true[:, i_lo], 1e-12)

        tag = f"{fam.name}"
        rows = [
            evaluate_method(f'{tag}|standard_boot', q_sboot, T_n,
                            true_value, q_true, levels, norm),
            evaluate_method(f'{tag}|learned_raw', q_model, T_n,
                            true_value, q_true, levels, norm),
            evaluate_method(f'{tag}|learned_recal', q_model_recal, T_n,
                            true_value, q_true, levels, norm),
        ]
        all_rows.extend(rows)
        r = rows[-1]
        print(f"\n[{stem} -> {fam.name}] ({time.time()-t0:.0f}s)  "
              f"in-family ref: cov95={spec['ref_cov95']:.3f}, "
              f"W1={spec['ref_w1']:.3f}")
        print(f"  OOD learned_recal: cov95={r['cov95']:.3f} "
              f"cov90={r['cov90']:.3f} W1={r['w1_truth']:.3f} | "
              f"standard boot: cov95={rows[0]['cov95']:.3f} "
              f"W1={rows[0]['w1_truth']:.3f}")

    print_results_table(all_rows, args.n_params * N_DATASETS_PER_PARAM,
                        unit='1/w')
    export_table(all_rows, 'm4_ood',
                 'Out-of-family stress tests: specialists evaluated '
                 'outside their training prior')

    out = cfg.RESULTS_DIR / 'm4_ood.npz'
    save = {'levels': levels}
    for r in all_rows:
        key = (r['method'].replace('|', '_').replace('(', '')
               .replace(')', '').replace('=', ''))
        for metric in ('cov95', 'cov90', 'len95', 'w1_truth'):
            save[f"{key}_{metric}"] = r[metric]
    np.savez_compressed(out, **save)
    print(f"\nSaved: {out.name}")


if __name__ == '__main__':
    main()
