"""
Paper figures, generated from the saved experiment .npz files.

Design notes (dataviz method): color follows the METHOD identity and is
fixed across every figure -- ours=blue, standard bootstrap=aqua,
m-out-of-n=yellow, parametric=green, Bayes oracle=violet; the truth is a
neutral dashed reference line, not a series. Palette validated (light
surface, worst adjacent CVD dE 24.2); aqua and yellow are sub-3:1 on
white, so their marks always carry direct value labels. One axis per
panel; hairline grid; no rainbow, no dual axes.

Outputs paper/figures/*.png (300 dpi) and *.pdf (vector, for LaTeX).

Usage:
    python -m amortized_bootstrap.experiments.make_figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .. import config as cfg
from ..export import FIGURES_DIR

# ---- method identity colors (validated categorical slots) ----
C_OURS = '#2a78d6'      # blue
C_OURS_RAW = '#86b6ef'  # lighter step of the same hue (raw vs recal)
C_SBOOT = '#1baf7a'     # aqua
C_MOON = '#eda100'      # yellow
C_PARAM = '#008300'     # green
C_BAYES = '#4a3aa7'     # violet
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

plt.rcParams.update({
    'font.size': 8.5,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK2,
    'axes.titlecolor': INK,
    'axes.titlesize': 9,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'axes.axisbelow': True,
    'legend.frameon': False,
    'legend.fontsize': 7.5,
    'figure.dpi': 120,
})

FAMILIES = [
    ('m1_uniform_max', 'Uniform max'),
    ('m2_stable_mean', 'Stable mean'),
    ('m3_pareto_hill', 'Pareto Hill'),
    ('m3_nts_var', 'NTS VaR 0.99'),
]


def _load(name):
    path = cfg.RESULTS_DIR / f"{name}.npz"
    return np.load(path) if path.exists() else None


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}",
                    dpi=300 if ext == 'png' else None,
                    bbox_inches='tight')
    plt.close(fig)
    print(f"  saved paper/figures/{name}.png/.pdf")


# ----------------------------------------------------------------------
# Figure 1: reliability (nominal level vs empirical own-root coverage)
# ----------------------------------------------------------------------

def fig_reliability():
    """Deviation form: empirical minus nominal coverage, in percentage
    points -- makes the recalibration correction visible (the raw and
    recalibrated curves are indistinguishable on a full 0-1 diagonal)."""
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.5), sharey=True)
    for ax, (name, title) in zip(axes, FAMILIES):
        d = _load(name)
        _style_axes(ax)
        ax.axhline(0.0, ls='--', lw=0.9, color=MUTED, zorder=1)
        if d is not None and 'test_cov_raw' in d:
            lv = d['levels']
            ax.plot(lv, 100 * (d['test_cov_raw'] - lv), lw=1.6,
                    color=C_OURS_RAW, zorder=2)
            ax.plot(lv, 100 * (d['test_cov_recal'] - lv), lw=1.6,
                    color=C_OURS, zorder=3)
        ax.set_title(title)
        ax.set_xlabel('nominal level')
        ax.set_xlim(0, 1)
        ax.set_ylim(-4, 4)
        ax.set_xticks([0, 0.5, 1.0])
    axes[0].set_ylabel('coverage deviation (pp)')
    fig.legend(handles=[
        plt.Line2D([], [], color=C_OURS_RAW, lw=1.6, label='learned (raw)'),
        plt.Line2D([], [], color=C_OURS, lw=1.6,
                   label='learned (recalibrated)'),
        plt.Line2D([], [], color=MUTED, lw=0.9, ls='--', label='ideal'),
    ], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.12))
    _save(fig, 'fig_reliability')


# ----------------------------------------------------------------------
# Figure 2: W1 to the true root distribution, by method and family
# ----------------------------------------------------------------------

_METHOD_ROWS = {
    'm1_uniform_max': [
        ('standard_bootstrap_w1_truth', 'Standard bootstrap', C_SBOOT),
        ('subsampling_m34_w1_truth', 'Subsampling m=34', C_MOON),
        ('parametric_bootstrap_w1_truth', 'Parametric', C_PARAM),
        ('bayes_oracle_w1_truth', 'Bayes oracle', C_BAYES),
        ('learned_recal_ours_w1_truth', 'Learned (ours)', C_OURS),
    ],
    'm2_stable_mean': [
        ('standard_bootstrap_w1_truth', 'Standard bootstrap', C_SBOOT),
        ('m_of_n_mcculloch_m34_w1_truth', 'm-out-of-n (est. rate)', C_MOON),
        ('parametric_stable_ref_w1_truth', 'Parametric stable', C_PARAM),
        ('learned_recal_ours_w1_truth', 'Learned (ours)', C_OURS),
    ],
    'm3_pareto_hill': [
        ('standard_bootstrap_w1_truth', 'Standard bootstrap', C_SBOOT),
        ('m_of_n_m34_k10_w1_truth', 'm-out-of-n m=34', C_MOON),
        ('parametric_MLE_w1_truth', 'Parametric MLE', C_PARAM),
        ('bayes_oracle_w1_truth', 'Bayes oracle', C_BAYES),
        ('learned_recal_ours_w1_truth', 'Learned (ours)', C_OURS),
    ],
    'm3_nts_var': [
        ('standard_bootstrap_w1_truth', 'Standard bootstrap', C_SBOOT),
        ('m_of_n_m100_w1_truth', 'm-out-of-n m=100', C_MOON),
        ('learned_recal_ours_w1_truth', 'Learned (ours)', C_OURS),
    ],
}


def fig_w1():
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
    for ax, (name, title) in zip(axes, FAMILIES):
        d = _load(name)
        _style_axes(ax)
        ax.grid(axis='x')
        ax.grid(False, axis='y')
        if d is not None:
            rows = [(lbl, float(d[k]), c) for k, lbl, c in
                    _METHOD_ROWS[name] if k in d]
            rows = rows[::-1]  # ours at top after inversion
            y = np.arange(len(rows))
            vals = [r[1] for r in rows]
            ax.barh(y, vals, height=0.62,
                    color=[r[2] for r in rows], edgecolor='none')
            ax.set_yticks(y)
            ax.set_yticklabels([r[0] for r in rows], fontsize=7)
            ax.set_xscale('log')
            for yi, v in zip(y, vals):
                ax.text(v * 1.15, yi, f"{v:.3f}", va='center',
                        fontsize=6.5, color=INK2)
            ax.set_xlim(right=max(vals) * 4.5)
        ax.set_title(title)
        ax.set_xlabel('W1 to true root dist. (log)')
    fig.tight_layout(w_pad=1.6)
    _save(fig, 'fig_w1_comparison')


# ----------------------------------------------------------------------
# Figure 3: width tracking (the anti-memorization figure)
# ----------------------------------------------------------------------

def fig_width_tracking():
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
    rng = np.random.default_rng(0)
    for ax, (name, title) in zip(axes, FAMILIES):
        d = _load(name)
        _style_axes(ax)
        if d is not None and 'w_model' in d:
            w_m = np.log10(np.maximum(d['w_model'], 1e-12))
            w_t = np.log10(np.maximum(d['w_true'], 1e-12))
            idx = rng.choice(len(w_m), size=min(3000, len(w_m)),
                             replace=False)
            lo = min(w_t.min(), w_m.min())
            hi = max(w_t.max(), w_m.max())
            pad = 0.05 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls='--',
                    lw=0.9, color=MUTED, zorder=1)
            ax.scatter(w_t[idx], w_m[idx], s=3, color=C_OURS, alpha=0.18,
                       linewidths=0, zorder=2)
            r = np.corrcoef(w_t, w_m)[0, 1]
            ax.text(0.05, 0.90, f"corr = {r:.3f}", fontsize=8,
                    color=INK, transform=ax.transAxes)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title)
        ax.set_xlabel('log10 true 95% width')
    axes[0].set_ylabel('log10 predicted width')
    fig.tight_layout()
    _save(fig, 'fig_width_tracking')


# ----------------------------------------------------------------------
# Figure 4: predicted quantile functions vs truth (two exemplars)
# ----------------------------------------------------------------------

def fig_quantile_overlay():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    d1 = _load('m1_uniform_max')
    ax = axes[0]
    _style_axes(ax)
    if d1 is not None and 'q_model_recal_head' in d1:
        lv = d1['levels']
        i = 0
        # distinct line styles for grayscale robustness
        ax.plot(lv, d1['q_std_boot_head'][i], lw=1.4, color=C_SBOOT,
                ls=':', label='standard bootstrap')
        ax.plot(lv, d1['q_bayes_head'][i], lw=1.4, color=C_BAYES,
                ls='-.', label='Bayes oracle')
        ax.plot(lv, d1['q_model_recal_head'][i], lw=1.6, color=C_OURS,
                label='learned (ours)')
        ax.plot(lv, d1['q_true_head'][i], ls='--', lw=1.1, color=INK,
                label='truth')
    ax.set_title('Uniform max: one test dataset')
    ax.set_xlabel('quantile level')
    ax.set_ylabel('root quantile')
    ax.legend(loc='lower right')

    d2 = _load('m2_stable_mean')
    ax = axes[1]
    _style_axes(ax)
    if d2 is not None and 'q_model_recal_head' in d2:
        lv = d2['levels']
        i = 0
        ax.plot(lv, d2['q_sboot_head'][i], lw=1.4, color=C_SBOOT,
                ls=':', label='standard bootstrap')
        ax.plot(lv, d2['q_param_head'][i], lw=1.4, color=C_PARAM,
                ls='-.', label='parametric stable')
        ax.plot(lv, d2['q_model_recal_head'][i], lw=1.6, color=C_OURS,
                label='learned (ours)')
        ax.plot(lv, d2['q_true_head'][i], ls='--', lw=1.1, color=INK,
                label='truth')
    ax.set_title('Stable mean: one test dataset')
    ax.set_xlabel('quantile level')
    ax.legend(loc='upper left')

    fig.tight_layout()
    _save(fig, 'fig_quantile_overlay')


# ----------------------------------------------------------------------
# Figure 5: real-data VaR coverage (dot plot vs the nominal level)
# ----------------------------------------------------------------------

C_BINOM = '#e34948'  # categorical slot 6; binomial exact appears only here

def fig_real_var():
    d = _load('m5_real_var')
    if d is None:
        print('  m5_real_var.npz missing; skipping fig_real_var')
        return
    assets = [('nasdaq', 'Nasdaq Composite'), ('sp500', 'S&P 500'),
              ('eurusd', 'EUR/USD'), ('usdjpy', 'USD/JPY'),
              ('gbpusd', 'GBP/USD')]
    # distinct marker shapes so the figure survives grayscale print
    methods = [('learned_(ours)', 'Learned (ours)', C_OURS, 'o'),
               ('standard_bootstrap', 'Standard bootstrap', C_SBOOT, 's'),
               ('m_of_n_m100', 'm-out-of-n', C_MOON, '^'),
               ('binomial_exact', 'Binomial exact', C_BINOM, 'D')]

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    _style_axes(ax)
    ax.grid(axis='x')
    ax.grid(False, axis='y')
    ax.axvline(0.95, ls='--', lw=1.0, color=MUTED, zorder=1)
    ax.text(0.951, 4.42, 'nominal 0.95', fontsize=7, color=MUTED)

    y = np.arange(len(assets))[::-1]
    for key, label, color, marker in methods:
        covs = [float(d[f'{a}_{key}_cov95']) for a, _ in assets]
        ax.scatter(covs, y, s=32, color=color, zorder=3, marker=marker,
                   edgecolors='#52514e', linewidths=0.5, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl in assets], fontsize=8)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.6, 4.9)
    ax.set_xlabel('95% interval coverage vs known population VaR')
    ax.legend(loc='lower left', ncol=1, fontsize=7, handletextpad=0.1,
              borderaxespad=0.2)
    fig.tight_layout()
    _save(fig, 'fig_real_var')


# ----------------------------------------------------------------------
# Figure 6: method overview schematic (training loop vs deployment)
# ----------------------------------------------------------------------

def _box(ax, x, y, w, h, text, fc='#ffffff', ec=AXIS, fontsize=7.5,
         bold=False, tc=INK):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.6",
        facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color=tc, zorder=3,
            fontweight='bold' if bold else 'normal')


def _arrow(ax, x0, y0, x1, y1, dashed=False, color=INK2, label=None,
           lx=0, ly=0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.1,
                                linestyle='--' if dashed else '-',
                                shrinkA=1, shrinkB=1), zorder=1)
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label,
                fontsize=6.5, color=INK2, ha='center', zorder=3)


def fig_method():
    fig, ax = plt.subplots(figsize=(9.8, 4.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 41)
    ax.axis('off')

    from matplotlib.patches import FancyBboxPatch
    # lane backgrounds
    ax.add_patch(FancyBboxPatch((0.5, 14.5), 99, 26,
                 boxstyle="round,pad=0.2", facecolor='#f6f6f3',
                 edgecolor='none', zorder=0))
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 99, 12.4,
                 boxstyle="round,pad=0.2", facecolor='#eef3fb',
                 edgecolor='none', zorder=0))
    ax.text(2, 38.6, 'TRAINING  (simulation, offline, repeated for '
            r'$8 \times 10^5$ examples)', fontsize=8, color=INK,
            fontweight='bold')
    ax.text(2, 10.9, 'DEPLOYMENT  (one forward pass per dataset)',
            fontsize=8, color=INK, fontweight='bold')

    # ---- training lane ----
    _box(ax, 2, 26, 13, 5.4,
         'Draw parameters\n' + r'$\lambda \sim \pi$' + '\n(prior over family)')
    _box(ax, 21, 30.5, 16, 5.4,
         'Dataset\n' + r'$X \sim F_\lambda$  ($n = 200$)')
    _box(ax, 21, 21.5, 16, 5.4,
         'Independent replicate\n' + r'$\tilde X \sim F_\lambda$')
    _box(ax, 43, 30.5, 15, 5.4,
         'Sort + standardize\n' + r'$(z, \mathrm{aux})$')
    _box(ax, 43, 21.5, 15, 5.4,
         'Single root draw\n' + r'$t = T_n(\tilde X) - T(F_\lambda)$')
    _box(ax, 64, 30.5, 14, 5.4, 'Monotone quantile\nnetwork '
         + r'$q_\theta$', fc='#dcebfb', ec=C_OURS, bold=True)
    _box(ax, 84, 30.5, 14, 5.4, '199 predicted\nquantiles '
         + r'$q(\tau)$')
    _box(ax, 84, 21.5, 14, 5.4, 'Pinball loss\n(proper scoring rule)',
         fc='#fdf3dc', ec=C_MOON)
    _box(ax, 43, 15.6, 35, 4.2,
         'Validation split: own-root recalibration  '
         + r'$\tau \rightarrow h^{-1}(\tau)$', fc='#ece9f7', ec=C_BAYES)

    _arrow(ax, 15, 29.5, 21, 32.2)
    _arrow(ax, 15, 28, 21, 24.8)
    _arrow(ax, 37, 33.2, 43, 33.2)
    _arrow(ax, 37, 24.2, 43, 24.2)
    _arrow(ax, 58, 33.2, 64, 33.2)
    _arrow(ax, 78, 33.2, 84, 33.2)
    _arrow(ax, 91, 30.5, 91, 26.9)                       # quantiles -> loss
    _arrow(ax, 58, 24.2, 84, 24.2)                       # t -> loss
    _arrow(ax, 84, 22.6, 71, 30.4, dashed=True,
           label='gradients', lx=4.5, ly=2.4)
    _arrow(ax, 68, 30.5, 63, 19.8, dashed=True)          # net -> recal fit

    # ---- deployment lane ----
    _box(ax, 2, 3, 13, 5.4, 'Observed\ndataset ' + r'$X$')
    _box(ax, 21, 3, 16, 5.4, 'Sort + standardize\n' + r'$(z, \mathrm{aux})$')
    _box(ax, 43, 3, 15, 5.4, 'Frozen network\n' + r'$q_\theta$',
         fc='#dcebfb', ec=C_OURS, bold=True)
    _box(ax, 64, 3, 14, 5.4, 'Recalibrated\nquantiles')
    _box(ax, 84, 3, 14, 5.4, 'Confidence interval\n'
         + r'$[\,T_n - q_{1-\alpha/2},\; T_n - q_{\alpha/2}\,]$',
         fontsize=6.8)

    _arrow(ax, 15, 5.7, 21, 5.7)
    _arrow(ax, 37, 5.7, 43, 5.7)
    _arrow(ax, 58, 5.7, 64, 5.7)
    _arrow(ax, 78, 5.7, 84, 5.7)
    _arrow(ax, 60.5, 15.4, 69, 8.6, dashed=True,
           label=r'$h^{-1}$', lx=3, ly=0.5)

    fig.tight_layout(pad=0.4)
    _save(fig, 'fig_method')


def main():
    print("Generating paper figures -> paper/figures/")
    fig_reliability()
    fig_w1()
    fig_width_tracking()
    fig_quantile_overlay()
    fig_real_var()
    fig_method()


if __name__ == '__main__':
    main()
