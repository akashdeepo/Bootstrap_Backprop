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

from .statistics import hill_estimator


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


def _kanter_positive_stable(gamma: np.ndarray, rng: Generator) -> np.ndarray:
    """Kanter (1975) positive stable draws, elementwise in gamma (flat)."""
    m = len(gamma)
    V = rng.uniform(-np.pi / 2, np.pi / 2, size=m)
    W = rng.exponential(1.0, size=m)
    V = np.clip(V, -np.pi / 2 + 1e-10, np.pi / 2 - 1e-10)
    phi = V + np.pi / 2
    term1 = np.sin(gamma * phi) / (np.cos(V) ** (1.0 / gamma))
    term2 = (np.cos(V - gamma * phi) / W) ** ((1.0 - gamma) / gamma)
    return term1 * term2


def sample_nts_vectorized(alpha: np.ndarray, theta: np.ndarray,
                          size: tuple, rng: Generator,
                          max_rounds: int = 500) -> np.ndarray:
    """
    Standard symmetric NTS draws (sigma=1, mu=0), vectorized over PER-ROW
    (alpha, theta) of shape (N, 1) against draws of shape (N, n).

    X = sqrt(T) * Z with T ~ TemperedStable(gamma = alpha/2, theta) via
    rejection from a positive stable (accept w.p. exp(-theta * T)), same
    algorithm as v1's scalar sampler (validated there), but with per-entry
    parameters so every dataset can have its own (alpha, theta).
    """
    N, n = size
    gamma_flat = np.broadcast_to(alpha / 2.0, size).reshape(-1)
    theta_flat = np.broadcast_to(theta, size).reshape(-1)

    T = np.empty(N * n)
    remaining = np.arange(N * n)
    for _ in range(max_rounds):
        g = gamma_flat[remaining]
        th = theta_flat[remaining]
        S = _kanter_positive_stable(g, rng)
        ok = (S > 0) & (rng.uniform(size=len(S)) < np.exp(-th * S))
        T[remaining[ok]] = S[ok]
        remaining = remaining[~ok]
        if len(remaining) == 0:
            break
    if len(remaining) > 0:
        raise RuntimeError(f"NTS rejection did not converge for "
                           f"{len(remaining)} entries")

    Z = rng.standard_normal(N * n)
    return (np.sqrt(T) * Z).reshape(size)


class ParetoHillFamily:
    """
    X ~ c * Pareto(alpha, x_min=1), T_n = Hill estimator with k top order
    statistics, T(F) = gamma = 1/alpha.

    Priors: alpha ~ U(alpha_min, alpha_max), c ~ LogUniform(c_min, c_max).
    The Hill statistic is scale-invariant, so c is a nuisance the model
    must ignore; the root T_n - gamma does NOT scale with the data
    (root_scale = 1).

    For EXACT Pareto data the root is analytically known via the Renyi
    representation: the top-k log-spacings are iid exponentials, giving
        H_k ~ gamma * Gamma(k, 1) / k  exactly, for any n and c.
    So truth is analytic, and the Bayes oracle is a 1-D posterior mixture
    (see hill_oracle.py).
    """

    name = "pareto_hill"
    n_params = 2  # columns: alpha, c

    def __init__(self, alpha_min: float = 1.5, alpha_max: float = 4.0,
                 c_min: float = 0.5, c_max: float = 5.0, k: int = 34):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.c_min = c_min
        self.c_max = c_max
        self.k = k

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        alpha = rng.uniform(self.alpha_min, self.alpha_max, size=n_params)
        c = np.exp(rng.uniform(np.log(self.c_min), np.log(self.c_max),
                               size=n_params))
        return np.stack([alpha, c], axis=1)

    def sample_data(self, params: np.ndarray, n: int,
                    rng: Generator) -> np.ndarray:
        alpha = params[:, 0:1]
        c = params[:, 1:2]
        u = rng.uniform(size=(len(params), n))
        return c * u ** (-1.0 / alpha)

    def statistic(self, x: np.ndarray) -> np.ndarray:
        return hill_estimator(x, k=self.k)

    def true_param(self, params: np.ndarray) -> np.ndarray:
        return 1.0 / params[:, 0]

    def true_root_quantiles(self, params: np.ndarray,
                            levels: np.ndarray) -> np.ndarray:
        """Exact: q(tau) = gamma * (GammaPPF(tau; k)/k - 1)."""
        from scipy.stats import gamma as gamma_dist
        g = 1.0 / params[:, 0]
        gk = gamma_dist.ppf(levels, a=self.k) / self.k  # (L,)
        return g[:, None] * (gk[None, :] - 1.0)


class NTSVaRFamily:
    """
    X ~ mu + sigma * NTS_std(alpha, theta), T_n = empirical VaR_0.99
    (np.quantile), T(F) = the true 0.99-quantile.

    Priors: alpha ~ U(1.1, 1.9), theta ~ LogU(0.3, 3),
    sigma ~ LogU(0.5, 5), mu ~ U(-2, 2).

    Symmetric NTS is a location-scale family in (mu, sigma), so
        VaR(F) = mu + sigma * VaR_std(alpha, theta)
    where VaR_std has no closed form but depends on only TWO parameters --
    a precomputed MC grid (nts_truth.py) supplies it for training-target
    centering at any prior draw. Construct via
        NTSVaRFamily(var_std_fn=make_var_std_fn(grid)).
    """

    name = "nts_var99"
    n_params = 4  # columns: alpha, theta, sigma, mu
    var_level = 0.99

    def __init__(self, alpha_min: float = 1.1, alpha_max: float = 1.9,
                 theta_min: float = 0.3, theta_max: float = 3.0,
                 sigma_min: float = 0.5, sigma_max: float = 5.0,
                 mu_range: float = 2.0, var_std_fn=None):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu_range = mu_range
        self.var_std_fn = var_std_fn

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        alpha = rng.uniform(self.alpha_min, self.alpha_max, size=n_params)
        theta = np.exp(rng.uniform(np.log(self.theta_min),
                                   np.log(self.theta_max), size=n_params))
        sigma = np.exp(rng.uniform(np.log(self.sigma_min),
                                   np.log(self.sigma_max), size=n_params))
        mu = rng.uniform(-self.mu_range, self.mu_range, size=n_params)
        return np.stack([alpha, theta, sigma, mu], axis=1)

    def sample_data(self, params: np.ndarray, n: int,
                    rng: Generator) -> np.ndarray:
        alpha = params[:, 0:1]
        theta = params[:, 1:2]
        sigma = params[:, 2:3]
        mu = params[:, 3:4]
        S = sample_nts_vectorized(alpha, theta, (len(params), n), rng)
        return mu + sigma * S

    def statistic(self, x: np.ndarray) -> np.ndarray:
        return np.quantile(x, self.var_level, axis=1)

    def true_param(self, params: np.ndarray) -> np.ndarray:
        if self.var_std_fn is None:
            raise RuntimeError("NTSVaRFamily needs var_std_fn "
                               "(see nts_truth.make_var_std_fn)")
        v_std = self.var_std_fn(params[:, 0], params[:, 1])
        return params[:, 3] + params[:, 2] * v_std


class BetaMaxPriorFamily:
    """
    X = theta * V with F_V(v) = 1 - (1-v)^b on [0,1]; T_n = max,
    T(F) = theta -- with the endpoint contact order b UNKNOWN:
    theta ~ LogUniform(theta_min, theta_max), b ~ U(b_min, b_max).

    The convergence rate of the max is n^(-1/b): the rate itself varies
    across the prior by orders of magnitude. This family exists to test
    whether the network can learn a data-conditional RATE (Milestone 4c):
    the uniform-trained specialist collapses on b != 1 (M4a regime 3),
    and widening the prior over b is the method's own prescribed fix.

    Analytic root quantiles: q(tau) = -theta * (1 - tau^(1/n))^(1/b).
    """

    name = "beta_max_prior"
    n_params = 2  # columns: theta, b

    def __init__(self, theta_min: float = 0.5, theta_max: float = 5.0,
                 b_min: float = 0.4, b_max: float = 2.6):
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.b_min = b_min
        self.b_max = b_max

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        lo, hi = np.log(self.theta_min), np.log(self.theta_max)
        theta = np.exp(rng.uniform(lo, hi, size=n_params))
        b = rng.uniform(self.b_min, self.b_max, size=n_params)
        return np.stack([theta, b], axis=1)

    def sample_data(self, params: np.ndarray, n: int,
                    rng: Generator) -> np.ndarray:
        u = rng.uniform(size=(len(params), n))
        v = 1.0 - (1.0 - u) ** (1.0 / params[:, 1:2])
        return params[:, 0:1] * v

    def statistic(self, x: np.ndarray) -> np.ndarray:
        return np.max(x, axis=1)

    def true_param(self, params: np.ndarray) -> np.ndarray:
        return params[:, 0]

    def true_root_quantiles(self, params: np.ndarray, levels: np.ndarray,
                            n: int) -> np.ndarray:
        return -params[:, 0:1] * (1.0 - levels[None, :] ** (1.0 / n)) \
            ** (1.0 / params[:, 1:2])


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
