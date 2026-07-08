"""
Milestone 5: real-data vignette -- 99% value-at-risk on daily market
returns, with MEASURABLE coverage.

Real data has no known truth, so naive real-data "coverage" is
unmeasurable. The semi-synthetic protocol recovers knowable truth: treat
the EMPIRICAL distribution of a long real loss series as the population
F. The true VaR_0.99 is then the 99% quantile of the full series, known
exactly by construction. Draw i.i.d. subsamples of n = 200 from the
series, apply the NTS-trained specialist UNCHANGED (frozen weights and
recalibration), and measure honest coverage against the known population
VaR, alongside the standard bootstrap, the m-out-of-n bootstrap, and the
exact order-statistic interval on identical subsamples.

Scope note (stated in the paper): i.i.d. resampling deliberately removes
serial dependence; the exercise tests distribution-shape transfer (real
heavy-tailed, asymmetric losses vs the NTS training prior), not
time-series forecasting. Real returns are OUT OF PRIOR: the Student-t
OOD results (cov 0.861 vs bootstrap 0.703) set the pre-registered
expectation, and results are reported whatever they are. Asset list was
fixed before any result was computed.

Data: daily closes from Stooq (free CSV endpoint), cached under data/
(gitignored; run once with network access). Losses are negated percent
log returns, so VaR_0.99 of losses is the standard 99% VaR.

Usage:
    python -m amortized_bootstrap.experiments.m5_real_var
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .. import config as cfg
from ..datagen import featurize
from ..model import QuantileNet
from ..training import predict_root_quantiles
from ..baselines_m3 import bootstrap_quantile_roots, binomial_exact_ci
from ..calibration import apply_recalibration
from ..evaluation import level_index
from ..export import export_table

VAR_LEVEL = 0.99
# Asset list fixed BEFORE any result was computed. (Amended once, before
# any results, for data ACCESS only: the original Stooq source sits
# behind a JavaScript wall, so series were re-selected from FRED's open
# CSV endpoint.)
ASSETS = [
    ('nasdaq', 'NASDAQCOM', 'Nasdaq Composite'),
    ('sp500', 'SP500', 'S&P 500'),
    ('eurusd', 'DEXUSEU', 'EUR/USD'),
    ('usdjpy', 'DEXJPUS', 'USD/JPY'),
    # gold PM fix (GOLDPMGBD228NLBM) was retired from FRED; replaced
    # before results were seen for that asset
    ('gbpusd', 'DEXUSUK', 'GBP/USD'),
]
START_DATE = '2000-01-01'
N_SUB = 2000
N = 200


def fetch_closes(name: str, fred_id: str) -> np.ndarray:
    """Daily observations from FRED's open CSV endpoint, cached under
    data/real/. Missing values ('.') are dropped."""
    cache = cfg.DATA_DIR / 'real' / f"{name}.csv"
    cache.parent.mkdir(exist_ok=True)
    if not cache.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
        print(f"  downloading {fred_id} from FRED...")
        # curl: python urllib intermittently stalls on this endpoint
        subprocess.run(['curl', '-sL', '--max-time', '90',
                        url, '-o', str(cache)], check=True)
    text = cache.read_text()
    if not text.startswith('observation_date'):
        cache.unlink()
        raise RuntimeError(f"{name}: FRED returned a non-CSV response "
                           f"(series {fred_id} may be retired)")
    rows = text.strip().splitlines()[1:]
    closes = []
    for row in rows:
        parts = row.split(',')
        if len(parts) < 2 or parts[1] in ('.', ''):
            continue
        if parts[0] >= START_DATE:
            v = float(parts[1])
            if v > 0:
                closes.append(v)
    if len(closes) < 1000:
        raise RuntimeError(f"{name}: only {len(closes)} usable rows")
    return np.array(closes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sub', type=int, default=N_SUB)
    parser.add_argument('--b-boot', type=int, default=1000)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = args.device
    levels = cfg.QUANTILE_LEVELS
    i_lo = level_index(levels, 0.975)
    i_hi = level_index(levels, 0.025)
    i_lo90 = level_index(levels, 0.95)
    i_hi90 = level_index(levels, 0.05)

    # Frozen NTS specialist + its recalibration
    ckpt = torch.load(cfg.RESULTS_DIR / 'm3_nts_var_model.pt',
                      map_location=device, weights_only=False)
    model = QuantileNet(n_input=N, n_aux=2, n_levels=len(levels),
                        hidden=384, depth=3).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    target_scale = float(ckpt['target_scale'])
    c0 = float(ckpt['c0'])
    tau_adj = np.load(cfg.RESULTS_DIR / 'm3_nts_var.npz')['tau_adj']

    print("=" * 74)
    print("Milestone 5: real-data VaR_0.99 vignette "
          f"({args.n_sub} subsamples of n={N} per asset)")
    print("=" * 74)

    rows = []
    save = {'levels': levels}
    for name, symbol, label in ASSETS:
        t0 = time.time()
        closes = fetch_closes(name, symbol)
        losses = -100.0 * np.diff(np.log(closes))
        var_true = np.quantile(losses, VAR_LEVEL)
        print(f"\n[{label}] {len(losses)} daily losses, "
              f"true VaR_0.99 = {var_true:.3f}%")

        idx = cfg.RNG_REAL.integers(0, len(losses),
                                    size=(args.n_sub, N))
        X = losses[idx]
        T_n = np.quantile(X, VAR_LEVEL, axis=1)

        # ours (frozen specialist + frozen recalibration)
        z, aux, s = featurize(X, compress=True)
        q_y = predict_root_quantiles(model, z, aux, np.ones(len(z)),
                                     target_scale, device=device)
        q_model = np.sinh(np.clip(q_y, -20.0, 20.0)) * c0 * s[:, None]
        q_model = apply_recalibration(q_model, levels, tau_adj)

        # classical methods on the same subsamples
        q_sboot = bootstrap_quantile_roots(X, levels, VAR_LEVEL,
                                           B=args.b_boot,
                                           rng=cfg.RNG_BASELINE)
        q_moon = bootstrap_quantile_roots(X, levels, VAR_LEVEL,
                                          B=args.b_boot,
                                          rng=cfg.RNG_BASELINE,
                                          m=100) * np.sqrt(100 / N)
        X_sorted = np.sort(X, axis=1)
        blo95, bhi95, _ = binomial_exact_ci(X_sorted, VAR_LEVEL, 0.05)
        blo90, bhi90, _ = binomial_exact_ci(X_sorted, VAR_LEVEL, 0.10)

        def ci_cov(q_root):
            lo95 = T_n - q_root[:, i_lo]
            hi95 = T_n - q_root[:, i_hi]
            lo90 = T_n - q_root[:, i_lo90]
            hi90 = T_n - q_root[:, i_hi90]
            return (float(np.mean((var_true >= lo95) & (var_true <= hi95))),
                    float(np.mean((var_true >= lo90) & (var_true <= hi90))),
                    float(np.median(hi95 - lo95) / var_true))

        results = {
            'learned_(ours)': ci_cov(q_model),
            'standard_bootstrap': ci_cov(q_sboot),
            'm_of_n_m100': ci_cov(q_moon),
            'binomial_exact': (
                float(np.mean((var_true >= blo95) & (var_true <= bhi95))),
                float(np.mean((var_true >= blo90) & (var_true <= bhi90))),
                float(np.median(bhi95 - blo95) / var_true)),
        }
        for meth, (c95, c90, w) in results.items():
            rows.append({'method': f"{name}|{meth}", 'cov95': c95,
                         'cov90': c90, 'len95': w,
                         'w1_truth': float('nan')})
            save[f"{name}_{meth}_cov95"] = c95
            save[f"{name}_{meth}_cov90"] = c90
            save[f"{name}_{meth}_relwidth"] = w
        save[f"{name}_var_true"] = var_true
        save[f"{name}_n_obs"] = len(losses)

        r = results
        print(f"  cov95: ours={r['learned_(ours)'][0]:.3f}  "
              f"boot={r['standard_bootstrap'][0]:.3f}  "
              f"m-of-n={r['m_of_n_m100'][0]:.3f}  "
              f"binom={r['binomial_exact'][0]:.3f}   "
              f"({time.time()-t0:.0f}s)")

    print()
    export_table(rows, 'm5_real_var',
                 'Real-data VaR 0.99: coverage against the known '
                 'population quantile of each full return series '
                 '(2000 i.i.d. subsamples of n = 200 per asset; '
                 'Len. 95 column is median width / true VaR)')
    np.savez_compressed(cfg.RESULTS_DIR / 'm5_real_var.npz', **save)
    print("Saved: m5_real_var.npz")


if __name__ == '__main__':
    main()
