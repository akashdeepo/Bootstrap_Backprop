"""
Aggregate multi-seed replication results.

Reads results/{family}{tag}.npz for tags '', '_v1', '_v2' (base seed plus
AB_VARIANT retrainings; test sets are identical across variants, so the
spread is pure training stochasticity + training-data redraw) and reports
mean and range of the learned-recalibrated metrics per family.

Usage:
    python -m amortized_bootstrap.experiments.replication_summary
"""

import numpy as np

from .. import config as cfg
from ..export import TABLES_DIR

FAMILIES = ['m1_uniform_max', 'm2_stable_mean', 'm3_pareto_hill',
            'm3_nts_var', 'm4c_beta_max']
PRETTY = {'m1_uniform_max': 'Uniform max',
          'm2_stable_mean': 'Stable mean',
          'm3_pareto_hill': 'Pareto Hill',
          'm3_nts_var': 'NTS VaR 0.99',
          'm4c_beta_max': 'Beta max (unknown $b$)'}
TAGS = ['', '_v1', '_v2']
METRICS = ['cov95', 'cov90', 'len95', 'w1_truth']


def main():
    lines_tex = [
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Multi-seed replication: mean [min, max] over three "
        "training seeds; test sets identical across seeds.}",
        "\\label{tab:replication}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Family & Cov.\\ 95 & Cov.\\ 90 & Len.\\ 95 & W1 truth \\\\",
        "\\midrule",
    ]
    print(f"{'Family':<18} {'metric':<9} {'mean':>8} {'min':>8} {'max':>8}")
    print("-" * 56)
    for fam in FAMILIES:
        vals = {m: [] for m in METRICS}
        found = []
        for tag in TAGS:
            path = cfg.RESULTS_DIR / f"{fam}{tag}.npz"
            if not path.exists():
                continue
            d = np.load(path)
            key = 'learned_recal_ours'
            if f'{key}_cov95' not in d:
                continue
            found.append(tag if tag else 'base')
            for m in METRICS:
                vals[m].append(float(d[f'{key}_{m}']))
        if not found:
            print(f"{fam:<18} -- no results found")
            continue
        cells = []
        for m in METRICS:
            v = np.array(vals[m])
            print(f"{fam:<18} {m:<9} {v.mean():>8.4f} {v.min():>8.4f} "
                  f"{v.max():>8.4f}   (seeds: {', '.join(found)})")
            cells.append(f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]")
        lines_tex.append(PRETTY.get(fam, fam.replace('_', '\\_')) + " & "
                         + " & ".join(cells) + " \\\\")
        print()
    lines_tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = TABLES_DIR / "replication_summary.tex"
    out.write_text("\n".join(lines_tex) + "\n")
    print(f"exported {out.name} -> paper/tables/")


if __name__ == '__main__':
    main()
