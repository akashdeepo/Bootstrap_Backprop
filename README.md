# amortized-bootstrap

Neural amortized estimation of sampling distributions for statistics where
Efron's bootstrap is inconsistent (bounded-support maxima, infinite-variance
means, extreme quantiles, tail-index estimators).

A quantile network is trained on simulated (dataset, root-draw) pairs from a
prior over a distribution family. At test time, one forward pass maps a
single dataset of n=200 observations to an estimate of the sampling
distribution of the root T_n - T(F), from which confidence intervals are
built.

## Requirements

Python 3.10+, PyTorch (CUDA recommended), numpy, scipy, matplotlib.

## Usage

All experiments run as modules from the repo root:

```
python -m amortized_bootstrap.experiments.m1_uniform_max     # max, Uniform(0, theta)
python -m amortized_bootstrap.experiments.m2_stable_mean     # mean, symmetric alpha-stable
python -m amortized_bootstrap.experiments.m3_pareto_hill     # Hill estimator, Pareto
python -m amortized_bootstrap.experiments.m3_nts_var         # VaR 0.99, normal tempered stable
python -m amortized_bootstrap.experiments.make_figures       # figures from saved results
```

Common flags: `--epochs`, `--n-train`, `--hidden`, `--device cpu|cuda`.
Each experiment prints a method-comparison table (coverage, interval length,
Wasserstein-1 to the true root distribution), runs an input-sensitivity
diagnostic, and saves results to `results/*.npz` plus a model checkpoint.

First runs build one-time Monte Carlo ground-truth caches (a stable quantile
table under `data/`, NTS truth caches under `results/`); subsequent runs
load them.

## Package layout

```
amortized_bootstrap/
  config.py          seeds (SeedSequence streams), paths, quantile levels
  families.py        distribution families with parameter priors
  datagen.py         single-draw training examples + featurization
  model.py           monotone quantile network (cumulative-softplus head)
  training.py        pinball-loss training, validation-based selection
  calibration.py     post-hoc quantile recalibration (fit on validation)
  evaluation.py      coverage/length/W1 metrics + diagnostics
  bayes.py           exact Bayes oracle, uniform max
  hill_oracle.py     exact Bayes oracle, Pareto Hill
  stable_table.py    stable quantile table + McCulloch estimator
  nts_truth.py       NTS VaR ground-truth machinery
  baselines*.py      classical methods: standard bootstrap, m-out-of-n,
                     subsampling, parametric bootstraps, order-statistic CI
  experiments/       milestone runners (see Usage)
```

## Reproducibility

Everything derives from `SeedSequence(42)` via independent, index-stable
child streams (`config.py`); train/validation/test data and test-time ground
truth never share a stream. Results in `results/` were produced by the
commands above with default arguments.
