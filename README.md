# amortized-bootstrap

**When the bootstrap is provably wrong, learn the sampling distribution instead.**

[![tests](https://github.com/akashdeepo/Bootstrap_Backprop/actions/workflows/ci.yml/badge.svg)](https://github.com/akashdeepo/Bootstrap_Backprop/actions)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![status](https://img.shields.io/badge/paper-in%20preparation-orange)

Efron's bootstrap quietly powers most of applied uncertainty
quantification, and it is *provably inconsistent* for some of the
statistics people care about most: maxima of bounded distributions, means
of infinite-variance data, extreme quantiles, tail-index estimators. The
classical fixes (m-out-of-n bootstrap, subsampling) need rate corrections
that depend on the very parameters you do not know.

This repo trains a small neural network, by simulation alone, to output
the sampling distribution of a statistic directly. One forward pass on a
single dataset of n = 200 observations gives you calibrated confidence
intervals in regimes where resampling cannot.

## The headline numbers

95% interval coverage on four canonical bootstrap-failure problems
(20,000 test datasets per problem, parameters held out from training):

| Problem | Learned (this repo) | Standard bootstrap | Best classical option |
|---|---|---|---|
| Max of bounded support | **0.949** | 0.877 | 0.951, but must *know* the family |
| Mean, infinite variance (stable) | **0.952** | 0.967 with 4x worse W1 | oracle-rate m-out-of-n: 0.991, intervals 52% too long |
| Hill tail index (Pareto) | **0.951** | 0.913 | parametric MLE: 0.947, must know the family |
| 99% VaR (tempered stable) | **0.947** | 0.724 | *none can work*: exact interval capped at 0.851 |

The VaR row is the point: at n = 200 and p = 0.99, **no distribution-free
method can reach 95% coverage even in principle** (the exact
order-statistic interval has a provable ceiling of 0.866). The learned
method gets 94.7%.

On **real daily market returns** (Nasdaq, S&P 500, EUR/USD, USD/JPY,
GBP/USD, evaluated against the known population quantile of each full
series), the same frozen model averages **0.868** coverage against the
bootstrap's 0.730. A single universal network with a statistic token
matches all four specialist models at once.

Where the exact Bayes-optimal answer is computable (two of the four
problems), the network captures **over 97% of the achievable
improvement**. This is not "beats a strawman"; it is near the ceiling of
what any method could do with the same data.

## How it works, in 60 seconds

1. **Pick a prior** over a family of data-generating processes (e.g.
   symmetric stable with unknown tail index, scale, location).
2. **Simulate training pairs**: draw parameters, draw a dataset X, and
   independently draw ONE value of the root T_n - T(F) from a second
   dataset. No Monte Carlo target distributions are ever built.
3. **Train a monotone quantile network** on sorted, standardized order
   statistics of X, scored by the pinball loss against that single draw.
   Pinball is a proper scoring rule, so the population minimizer is
   exactly the conditional law of the root given the data.
4. **Recalibrate** quantile levels on a validation split against each
   dataset's own root (not predictive PIT; the difference is worth a
   full coverage point and is one of the paper's findings).
5. **Deploy**: one forward pass maps any new dataset to 199 recalibrated
   root quantiles; confidence intervals read off directly.

## Why you can trust the numbers

The first version of this project produced spectacular results that were
completely wrong: with shared training targets, the loss-optimal network
was a *constant* that ignored its input, and no marginal metric could
tell. That failure is baked into this codebase as a design rule and a
test:

- **No shared targets, ever.** Every training example draws fresh
  parameters; a constant predictor is structurally suboptimal.
- **A mandatory input-sensitivity diagnostic** runs in every experiment:
  predicted interval widths must track the truth across test datasets
  (correlations 0.95 to 1.00 here) and must respond to garbage inputs.
- **Leakage-proof seeding**: train, validation, test, and ground-truth
  randomness come from disjoint, index-stable streams of a single
  master seed.
- **Oracle-anchored evaluation**: exact Bayes-optimal references where
  computable, exact (non-resampled) classical baselines for the max,
  fresh Monte Carlo ground truth that never touches training.
- **Replication**: every experiment run under three training seeds with
  paired test sets; every gate passed in every replicate, coverage
  spread 0.2 to 0.4 percentage points.

## Quickstart

Python 3.10+; install PyTorch first (CUDA recommended), then:

```
pip install -e .
python -m amortized_bootstrap.experiments.m1_uniform_max --epochs 5 --n-train 50000
```

That trains a small demo model in a couple of minutes and prints a full
method-comparison table (coverage, interval length, Wasserstein distance
to the exact truth) against the bootstrap, subsampling, a parametric
bootstrap, and the exact Bayes oracle.

## Reproducing everything

Defaults reproduce the committed results in `results/` (RTX 3060-class
GPU, roughly 15 to 25 minutes per experiment; first runs also build
one-time Monte Carlo ground-truth caches):

```
python -m amortized_bootstrap.experiments.m1_uniform_max     # max | Uniform(0, theta)
python -m amortized_bootstrap.experiments.m2_stable_mean     # mean | alpha-stable (Athreya case)
python -m amortized_bootstrap.experiments.m3_pareto_hill     # Hill | Pareto
python -m amortized_bootstrap.experiments.m3_nts_var         # VaR 0.99 | tempered stable
python -m amortized_bootstrap.experiments.m4_ood             # out-of-family stress tests
python -m amortized_bootstrap.experiments.m4_universal       # one model, all four statistics
python -m amortized_bootstrap.experiments.m4c_beta_max       # learning an unknown convergence RATE
python -m amortized_bootstrap.experiments.m5_real_var        # real market data (downloads from FRED)
python -m amortized_bootstrap.experiments.replication_summary
python -m amortized_bootstrap.experiments.make_figures
```

Multi-seed replication: prefix any experiment with `AB_VARIANT=1` (or 2)
to re-randomize training while keeping test sets fixed.

The rate-learning experiment (`m4c_beta_max`) deserves a special mention:
the convergence rate of the max varies as n^(-1/b) with the contact
order b unknown, spanning six orders of magnitude across the prior. The
network stays within 3 points of nominal coverage on every slice of b
while the bootstrap swings between 49% and 97%.

## Package layout

```
amortized_bootstrap/
  config.py          seed streams, paths, quantile levels
  families.py        training families with parameter priors + samplers
  ood_families.py    out-of-family test distributions
  datagen.py         single-draw training examples, featurization
  model.py           monotone quantile network (cumulative softplus head)
  training.py        pinball-loss training, validation-based selection
  calibration.py     own-root quantile recalibration
  evaluation.py      coverage / length / W1 metrics, diagnostics
  bayes.py           exact Bayes oracle: uniform max
  hill_oracle.py     exact Bayes oracle: Pareto Hill
  stable_table.py    stable quantile table + McCulloch estimator
  nts_truth.py       tempered-stable VaR ground-truth machinery
  ood_truth.py       generic MC truth + generic bootstrap baseline
  baselines*.py      classical methods, exact where closed forms exist
  experiments/       the runners listed above
tests/               fast CPU test suite (pytest tests/ -q)
results/             .npz eval arrays, model checkpoints, truth caches
```

## Tests

```
pytest tests/ -q
```

Ten fast, CPU-only tests guard the mathematical claims the results rely
on: analytic-vs-Monte-Carlo agreement for the exact root distributions,
the stable sampler's Gaussian limit, calibration round-trips,
quantile-head monotonicity, seed determinism, and featurization
equivariance. The same suite runs in CI on every push.

## Status, citation, license

The accompanying paper is in preparation; a citation entry and an open
license will be added on publication. Until then the code is available
for review with all rights reserved. Star or watch the repo if you want
the update.
