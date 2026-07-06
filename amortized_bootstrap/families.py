"""
Distribution families with parameter priors.

A Family bundles:
  - a prior over its parameters (sample_params)
  - a data sampler conditional on parameters (sample_data)
  - the statistic T_n (statistic)
  - the true functional T(F) as a function of parameters (true_param)

Every training example draws fresh parameters from the prior, so the
target root distribution L(T_n - T(F) | params) varies with the input.
This is the structural fix for the v1 memorization artifact: a constant
output is no longer loss-optimal.

Milestone 1 implements UniformMaxFamily only. Later milestones add
Stable mean, NTS VaR, Pareto/Burr Hill (see RESEARCH_PLAN.md).
"""

import numpy as np
from numpy.random import Generator


class UniformMaxFamily:
    """
    X ~ Uniform(0, theta), T_n = max(X), T(F) = theta.

    Prior: theta ~ LogUniform(theta_min, theta_max).

    Non-regular case: the standard bootstrap is inconsistent for the max
    (P(T* = T_n) -> 1 - 1/e), and the root T_n - theta lives on the wrong
    side of zero for any resampling scheme confined to the observed data.
    """

    name = "uniform_max"

    def __init__(self, theta_min: float = 0.5, theta_max: float = 5.0):
        self.theta_min = theta_min
        self.theta_max = theta_max

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        """Draw theta ~ LogUniform(theta_min, theta_max). Returns (n_params,)."""
        lo, hi = np.log(self.theta_min), np.log(self.theta_max)
        return np.exp(rng.uniform(lo, hi, size=n_params))

    def sample_data(self, params: np.ndarray, n: int, rng: Generator) -> np.ndarray:
        """One dataset of size n per parameter. Returns (n_params, n)."""
        u = rng.uniform(size=(len(params), n))
        return u * params[:, None]

    def statistic(self, x: np.ndarray) -> np.ndarray:
        """T_n = sample maximum, computed row-wise. Returns (n_datasets,)."""
        return np.max(x, axis=1)

    def true_param(self, params: np.ndarray) -> np.ndarray:
        """T(F) = theta."""
        return params

    # ---- analytic truth (uniform max only; used for evaluation) ----

    def true_root_quantiles(self, params: np.ndarray, levels: np.ndarray,
                            n: int) -> np.ndarray:
        """
        Exact quantiles of the root R = T_n - theta given theta.

        P(R <= r | theta) = ((theta + r) / theta)^n on [-theta, 0], so
        q(tau) = theta * (tau^(1/n) - 1).

        Returns (n_params, n_levels).
        """
        return params[:, None] * (levels[None, :] ** (1.0 / n) - 1.0)
