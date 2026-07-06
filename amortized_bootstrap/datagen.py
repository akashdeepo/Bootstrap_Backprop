"""
Meta-dataset generation and featurization.

Each training example is a triple (X, t, params):
  - params ~ prior (fresh draw, never shared between examples)
  - X      ~ F(params), the observed dataset (n points)
  - t      = T_n(X_indep) - T(F(params)), a SINGLE draw of the root from an
             INDEPENDENT dataset X_indep ~ F(params)

Training a quantile head with pinball loss against single draws t is a
proper scoring rule: the minimizer is the true conditional law of t | X,
i.e. the posterior-predictive root distribution. No Monte Carlo target
vectors are ever built, and no target is shared across examples.

Featurization (built-in invariances):
  - sort the dataset (order statistics are sufficient for exchangeable data)
  - standardize by median/IQR; the root for location-scale-equivariant
    statistics satisfies root(c*X) = c*root(X), so the model predicts the
    root in units of s = IQR and predictions are de-standardized by s
  - aux inputs [log s, med/s] let the model use absolute scale/location,
    which matters because the prior is not scale-invariant
"""

import numpy as np
from numpy.random import Generator


def generate_examples(family, n_examples: int, n: int, rng: Generator):
    """
    Generate (X, t, params) for one split.

    Returns:
        X      (n_examples, n)  raw datasets
        t      (n_examples,)    single independent root draw per example
        params (n_examples,)    the true parameters (kept for evaluation only;
                                never fed to the model)
    """
    params = family.sample_params(n_examples, rng)
    X = family.sample_data(params, n, rng)
    X_indep = family.sample_data(params, n, rng)
    t = family.statistic(X_indep) - family.true_param(params)
    return X, t, params


def featurize(X: np.ndarray, eps: float = 1e-12):
    """
    Sorted, standardized representation plus auxiliary scale/location inputs.

    Returns:
        z   (N, n)  sorted values, centered by median, scaled by IQR
        aux (N, 2)  [log s, med / s]
        s   (N,)    the per-dataset scale (IQR), used to de-standardize
                    predicted root quantiles
    """
    Xs = np.sort(X, axis=1)
    q25, med, q75 = np.percentile(Xs, [25.0, 50.0, 75.0], axis=1)
    s = np.maximum(q75 - q25, eps)
    z = (Xs - med[:, None]) / s[:, None]
    aux = np.stack([np.log(s), med / s], axis=1)
    return z.astype(np.float32), aux.astype(np.float32), s
