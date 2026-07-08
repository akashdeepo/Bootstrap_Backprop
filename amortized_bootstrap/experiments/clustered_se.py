"""
Clustered (between-parameter) coverage standard errors.

Test sets are P unique parameters x D datasets each, so coverage
indicators cluster within parameter: the honest SE treats the P per-
parameter coverage rates as the units, SE = sd(per-param coverage) /
sqrt(P), rather than the pooled binomial SE over P*D datasets.

Recomputes per-dataset coverage by loading each specialist checkpoint and
regenerating its test set deterministically (fresh_rng(2)); no retraining.

Usage:
    python -m amortized_bootstrap.experiments.clustered_se
"""

import numpy as np
import torch

from .. import config as cfg
from ..datagen import featurize
from ..model import QuantileNet
from ..training import predict_root_quantiles
from ..families import (UniformMaxFamily, StableMeanFamily,
                        ParetoHillFamily, NTSVaRFamily, BetaMaxPriorFamily)
from ..nts_truth import (build_or_load_var_grid, make_var_std_fn,
                         build_or_load_test_truth)
from ..calibration import apply_recalibration
from ..evaluation import level_index
from ..export import TABLES_DIR


def load_spec(stem, device):
    ckpt = torch.load(cfg.RESULTS_DIR / f"{stem}_model.pt",
                      map_location=device, weights_only=False)
    model = QuantileNet(n_input=cfg.N, n_aux=2,
                        n_levels=len(cfg.QUANTILE_LEVELS),
                        hidden=384, depth=3).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    d = np.load(cfg.RESULTS_DIR / f"{stem}.npz")
    return model, ckpt, d['tau_adj']


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = cfg.N
    levels = cfg.QUANTILE_LEVELS
    i_lo = level_index(levels, 0.975)  # CI lower uses upper root quantile
    i_hi = level_index(levels, 0.025)

    grid = build_or_load_var_grid()
    var_std_fn = make_var_std_fn(grid)

    # (stem, family, n_test_params, compress, asinh, scale_root, log_neg)
    SPECS = [
        ('m1_uniform_max', UniformMaxFamily(), 200, False, False, True, False),
        ('m2_stable_mean', StableMeanFamily(), 200, True, True, True, False),
        ('m3_pareto_hill', ParetoHillFamily(), 200, True, True, False, False),
        ('m3_nts_var', NTSVaRFamily(var_std_fn=var_std_fn), 100, True, True,
         True, False),
        ('m4c_beta_max', BetaMaxPriorFamily(), 200, False, False, True, True),
    ]

    print(f"{'family':<18} {'cov95':>7} {'pooledSE':>9} {'clustSE':>8} "
          f"{'P':>4}")
    print("-" * 52)
    rows_tex = []
    for stem, fam, P, compress, asinh, scale_root, log_neg in SPECS:
        model, ckpt, tau_adj = load_spec(stem, device)
        target_scale = float(ckpt['target_scale'])
        c0 = float(ckpt['c0']) if 'c0' in ckpt else None

        rng_te = cfg.fresh_rng(2)
        D = 100
        p_u = fam.sample_params(P, rng_te)
        params = (np.repeat(p_u, D, axis=0) if p_u.ndim == 2
                  else np.repeat(p_u, D))
        X = fam.sample_data(params, n, rng_te)
        true_value = fam.true_param(params)

        T_n = fam.statistic(X)
        z, aux, s = featurize(X, compress=compress)
        s_root = s if scale_root else np.ones(len(X))

        q_y = predict_root_quantiles(model, z, aux, np.ones(len(z)),
                                     target_scale, device=device)
        if log_neg:
            q = -np.exp(np.clip(q_y[:, ::-1], -700.0, 50.0)) * s_root[:, None]
        elif asinh:
            q = np.sinh(np.clip(q_y, -20.0, 20.0)) * c0 * s_root[:, None]
        else:
            q = q_y * s_root[:, None]
        q = apply_recalibration(q, levels, tau_adj)

        lo = T_n - q[:, i_lo]
        hi = T_n - q[:, i_hi]
        cov = ((true_value >= lo) & (true_value <= hi)).astype(float)

        cov_by_param = cov.reshape(P, D).mean(axis=1)
        pooled = cov.mean()
        pooled_se = np.sqrt(pooled * (1 - pooled) / len(cov))
        clust_se = cov_by_param.std(ddof=1) / np.sqrt(P)
        print(f"{stem:<18} {pooled:>7.4f} {pooled_se:>9.4f} "
              f"{clust_se:>8.4f} {P:>4}")
        rows_tex.append((stem, pooled, pooled_se, clust_se, P))

    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{95\\% coverage with pooled and clustered "
             "(between-parameter) standard errors.}",
             "\\label{tab:clustered_se}",
             "\\begin{tabular}{lcccc}", "\\toprule",
             "Family & Cov.\\ 95 & Pooled SE & Clustered SE & Params \\\\",
             "\\midrule"]
    for stem, c, pse, cse, P in rows_tex:
        lines.append(f"{stem.replace('_', chr(92)+'_')} & {c:.3f} & "
                     f"{pse:.4f} & {cse:.4f} & {P} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES_DIR / "clustered_se.tex").write_text("\n".join(lines) + "\n")
    print("exported clustered_se.tex -> paper/tables/")


if __name__ == '__main__':
    main()
