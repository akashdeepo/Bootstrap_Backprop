# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

v2 of the bootstrap-correction research (Akash Deep, Texas Tech): **amortized inference for sampling distributions of non-regular statistics**. A network is trained on datasets drawn from a prior over distribution families and learns to output the data-conditional distribution of the root T_n - T(F), targeting exactly the cases where Efron's bootstrap is provably inconsistent (heavy tails, extremes, tail quantiles, tail index).

This repo supersedes the v1 project (`../Bootstrap BackProp/`, frozen at tag `v1-baseline`), whose headline results were found to be a memorization artifact: the v1 training target was a single fixed ground-truth distribution per problem, so the loss-optimal model ignored its input. `RESEARCH_PLAN.md` is the canonical design document — read it before making methodological changes.

## Commands

Scripts run as modules from the repo root:

```
python -m amortized_bootstrap.experiments.m1_uniform_max            # Milestone 1
python -m amortized_bootstrap.experiments.m1_uniform_max --epochs 40 --n-train 800000 --hidden 384
```

No test suite yet. Stack: Python, PyTorch (CUDA), numpy, scipy.

## Hard Constraints

- **NO Unicode characters in any code or output** — Windows cp1252 console. ASCII only (`->`, `alpha`, `+-`).
- **The anti-memorization invariants are non-negotiable** (they are the reason v2 exists):
  1. Every training example draws fresh parameters from the prior — no target is ever shared across examples.
  2. Training targets are SINGLE independent root draws scored with a proper scoring rule (pinball/CRPS) — never Monte Carlo target vectors.
  3. Train/val/test use disjoint `SeedSequence` streams (`config.py`); test parameters and test ground truth never touch training.
  4. Every experiment must run the input-sensitivity diagnostic (`evaluation.input_sensitivity_diagnostic`) and it must PASS (output must track the input; a constant-output model must fail loudly).
- Seeds: everything descends from `SeedSequence(42)` in `config.py`. Never create ad-hoc RNGs.

## Architecture

The pipeline (per family): `families.py` (prior + sampler + statistic + T(F)) -> `datagen.py` (single-draw examples, sorted/standardized featurization with scale equivariance) -> `model.py` (MLP on sorted order statistics -> monotone quantile head via cumulative softplus) -> `training.py` (pinball loss, validation-based model selection) -> `evaluation.py` (CI coverage/length, W1 to truth, regret to Bayes, input-sensitivity diagnostic).

Key modeling conventions:
- The model predicts quantiles of the STANDARDIZED root t/s (s = per-dataset IQR from `featurize`), scaled by a global `target_scale` constant computed from the training split. De-standardize with `training.predict_root_quantiles`.
- CIs are built from root quantiles as [T_n - q(1-alpha/2), T_n - q(alpha/2)].
- W1 and CI lengths are reported in natural units (multiplied by n/theta) so results aggregate across the prior.

Problem-specific oracle/baseline modules (currently uniform max only):
- `bayes.py` — exact posterior predictive via the V = (theta/L)^-n uniformization; depends on data only through the max. Regret to this oracle is the strongest evaluation metric.
- `baselines.py` — standard bootstrap, m-out-of-n subsampling, and parametric bootstrap computed EXACTLY from order statistics (closed forms exist for the max; no resampling noise).
- `distributions.py`, `statistics.py` — copied from v1 (custom CMS stable sampler, Kanter subordinator-rejection NTS sampler; both correct and needed for later milestones).

Gate G4 note: never compare model-to-Bayes W1 against the model's own truth-W1 (degenerate when Bayes ~= truth by the triangle inequality); compare regret against the best classical nonparametric baseline instead.

## Where Results Go

`results/` — .npz metrics and .pt checkpoints per milestone, committed to git (small). Large regenerable caches must be gitignored; currently all data is generated on the fly (single-draw training makes this cheap), so there is no data directory.
