# amortized-bootstrap

Amortized neural estimation of sampling distributions for statistics where
Efron's bootstrap is provably inconsistent: maxima of bounded-support
distributions, means under infinite variance, extreme quantiles, and
tail-index estimators.

**Status:** research code accompanying a paper in preparation.

## The idea

For a statistic `T_n` computed from an i.i.d. sample of size `n`, classical
inference needs the sampling distribution of the root `T_n - T(F)`. The
bootstrap estimates it by resampling -- and fails in well-documented ways for
non-regular statistics (Bickel-Freedman 1981; Athreya 1987). The classical
remedies (m-out-of-n bootstrap, subsampling) require rate corrections that
depend on unknown parameters and behave poorly at realistic sample sizes.

This project trains a quantile network by simulation instead:

1. Draw distribution parameters from a prior over a family; draw one dataset
   `X` and, independently, ONE draw of the root `t = T_n(X') - T(F)`.
2. Train a monotone quantile head on sorted, standardized order statistics
   of `X` with pinball loss against the single draw `t`. Pinball loss is a
   proper scoring rule, so the population minimizer is the true conditional
   law of the root given the data.
3. Recalibrate quantile levels on a validation split against each dataset's
   OWN root (simulation calibration), then read confidence intervals off the
   predicted quantiles: `[T_n - q(1-a/2), T_n - q(a/2)]`.

At test time a single forward pass maps one dataset of `n = 200` points to
its full sampling-distribution estimate. Every experiment ships with an
input-sensitivity diagnostic that fails loudly if a model ignores its input,
and with exact or Monte Carlo ground truth built from seed streams that never
touch training.

## Installation

Python 3.10+. Install PyTorch first (CUDA build recommended,
[pytorch.org](https://pytorch.org)), then:

```
pip install -e .
```

or just `pip install -r requirements.txt` and run from the repo root.

## Quickstart

```
python -m amortized_bootstrap.experiments.m1_uniform_max --epochs 5 --n-train 50000
```

trains a small model on the uniform-max family and prints a method-comparison
table (CI coverage, interval length, Wasserstein-1 distance to the exact root
distribution) against the standard bootstrap, subsampling, a parametric
bootstrap, and the exact Bayes oracle.

## Reproducing the experiments

All experiments run as modules from the repo root. Default settings match
the committed results in `results/` (RTX 3060, ~15-25 min per experiment;
first runs also build one-time Monte Carlo ground-truth caches):

```
python -m amortized_bootstrap.experiments.m1_uniform_max     # max | Uniform(0, theta)
python -m amortized_bootstrap.experiments.m2_stable_mean     # mean | symmetric alpha-stable (Athreya case)
python -m amortized_bootstrap.experiments.m3_pareto_hill     # Hill | Pareto
python -m amortized_bootstrap.experiments.m3_nts_var         # VaR 0.99 | normal tempered stable
python -m amortized_bootstrap.experiments.m4_ood             # out-of-family stress tests
python -m amortized_bootstrap.experiments.m4_universal       # one model, all statistics
python -m amortized_bootstrap.experiments.make_figures       # figures from saved results
```

Common flags: `--epochs`, `--n-train`, `--hidden`, `--device cpu|cuda`.
Each run prints its comparison table, runs the input-sensitivity diagnostic,
and writes `results/<name>.npz` plus a model checkpoint.

## Repository map

```
amortized_bootstrap/
  config.py          seed streams (SeedSequence(42)), paths, quantile levels
  families.py        training families with parameter priors + samplers
  ood_families.py    out-of-family test distributions (Milestone 4)
  datagen.py         single-draw training examples, featurization
  model.py           monotone quantile network (cumulative-softplus head)
  training.py        pinball-loss training, validation-based selection
  calibration.py     own-root quantile recalibration
  evaluation.py      coverage / length / W1 metrics, diagnostics
  bayes.py           exact Bayes oracle: uniform max
  hill_oracle.py     exact Bayes oracle: Pareto Hill
  stable_table.py    stable quantile table + McCulloch estimator
  nts_truth.py       NTS VaR ground-truth machinery
  ood_truth.py       generic MC truth + generic bootstrap baseline
  baselines*.py      classical methods: standard bootstrap, m-out-of-n,
                     subsampling, parametric bootstraps, order-statistic CI
  experiments/       milestone runners (see above)
tests/               fast CPU test suite (pytest)
results/             .npz eval arrays, model checkpoints, truth caches
```

## Design invariants

- **No shared targets.** Every training example draws fresh parameters from
  the prior; targets are single independent root draws, never Monte Carlo
  target vectors. A constant-output model is structurally suboptimal.
- **Leakage-proof randomness.** Train / validation / test / ground-truth
  data come from independent, index-stable `SeedSequence` streams; test
  parameters and test-time truth never touch training.
- **The input-sensitivity diagnostic must pass.** Model output must track
  the input (interval-width correlation with the truth) and must respond to
  garbage inputs; experiments print this check every run.

## Tests

```
pytest tests/ -q
```

Fast, CPU-only: analytic-vs-Monte-Carlo checks for the exact root
distributions, sampler validation, calibration round-trips, quantile-head
monotonicity, and seed determinism.

## Citation and license

The accompanying paper is in preparation; a citation entry will be added on
publication. Until then the code is provided for review with all rights
reserved (no license file yet -- deliberately).
