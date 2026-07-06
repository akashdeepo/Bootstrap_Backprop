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


def sample_sas_vectorized(alpha: np.ndarray, size: tuple,
                          rng: Generator) -> np.ndarray:
    """
    Chambers-Mallows-Stuck sampler for symmetric alpha-stable, vectorized
    over PER-ROW alpha (shape (N, 1) broadcast against draws of shape
    (N, n)). Standard scale, location 0. Same formula as the scalar-alpha
    CMS in distributions.py (copied from v1, validated there).
    """
    V = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
    W = rng.exponential(1.0, size=size)
    V = np.clip(V, -np.pi / 2 + 1e-10, np.pi / 2 - 1e-10)
    term1 = np.sin(alpha * V) / (np.cos(V) ** (1.0 / alpha))
    term2 = (np.cos((1.0 - alpha) * V) / W) ** ((1.0 - alpha) / alpha)
    return term1 * term2


class StableMeanFamily:
    """
    X ~ mu + sigma * SaS(alpha), T_n = mean(X), T(F) = mu.

    Priors: alpha ~ U(alpha_min, alpha_max) (restricted to (1, 2) so the
    mean exists), sigma ~ LogUniform(sigma_min, sigma_max), mu ~ U(-2, 2).

    THE flagship non-regular case (Athreya 1987): for alpha < 2 the
    variance is infinite and the standard bootstrap of the mean is
    inconsistent -- the bootstrap distribution converges to a RANDOM limit.
    The root converges at rate n^(1 - 1/alpha), which depends on the
    unknown alpha, so rate-corrected subsampling requires estimating alpha.

    By the stability property, the root is EXACTLY distributed as
        mean(X) - mu ~ SaS(alpha, scale = sigma * n^(1/alpha - 1)),
    so exact evaluation truth only requires standard SaS quantiles
    (stable_table.py).
    """

    name = "stable_mean"
    n_params = 3  # columns: alpha, sigma, mu

    def __init__(self, alpha_min: float = 1.1, alpha_max: float = 1.95,
                 sigma_min: float = 0.5, sigma_max: float = 5.0,
                 mu_range: float = 2.0):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu_range = mu_range

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        """Returns (n_params, 3) array with columns [alpha, sigma, mu]."""
        alpha = rng.uniform(self.alpha_min, self.alpha_max, size=n_params)
        sigma = np.exp(rng.uniform(np.log(self.sigma_min),
                                   np.log(self.sigma_max), size=n_params))
        mu = rng.uniform(-self.mu_range, self.mu_range, size=n_params)
        return np.stack([alpha, sigma, mu], axis=1)

    def sample_data(self, params: np.ndarray, n: int,
                    rng: Generator) -> np.ndarray:
        alpha = params[:, 0:1]
        sigma = params[:, 1:2]
        mu = params[:, 2:3]
        S = sample_sas_vectorized(alpha, (len(params), n), rng)
        return mu + sigma * S

    def statistic(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=1)

    def true_param(self, params: np.ndarray) -> np.ndarray:
        return params[:, 2]

    def root_scale(self, params: np.ndarray, n: int) -> np.ndarray:
        """Exact scale of the root: sigma * n^(1/alpha - 1)."""
        alpha, sigma = params[:, 0], params[:, 1]
        return sigma * n ** (1.0 / alpha - 1.0)


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
