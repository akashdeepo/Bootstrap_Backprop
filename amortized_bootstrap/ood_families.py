"""
Out-of-family (OOD) test distributions for Milestone 4.

Each specialist model was trained on ONE family prior; these families
probe what happens when the data-generating process leaves that prior:

  uniform_max  -> BetaMaxFamily(b): density ~ b(1-x/theta)^(b-1) near the
                  endpoint. b=1 is the training family (uniform); b=2
                  (vanishing density) and b=0.5 (diverging density) change
                  the extreme-value limit law itself.
  stable_mean  -> StudentTMeanFamily: t(nu) with nu in (1.2, 1.9) is IN
                  the stable domain of attraction but is not stable (tests
                  tail-driven generalization); nu in (2.5, 8) is the
                  finite-variance CLT regime (tests the other direction).
  pareto_hill  -> BurrHillFamily / FrechetHillFamily: same tail index but
                  different second-order regular variation, so the Hill
                  root is no longer exactly Gamma -- finite-k bias appears.
  nts_var99    -> StudentTVaRFamily: heavy-tailed but a different tail
                  shape than tempered stable.

API matches families.py: sample_params / sample_data / statistic /
true_param (+ true_root_quantiles where analytic). `kind` tells the
evaluation harness which specialist pipeline applies.
"""

import numpy as np
from numpy.random import Generator
from scipy.stats import t as t_dist

from .statistics import hill_estimator


class BetaMaxFamily:
    """X = theta * V, F_V(v) = 1 - (1-v)^b on [0,1]; T_n = max, T(F) = theta.
    Analytic root quantiles: q(tau) = -theta * (1 - tau^(1/n))^(1/b)."""

    kind = 'max'

    def __init__(self, b: float, theta_min: float = 0.5,
                 theta_max: float = 5.0):
        self.b = b
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.name = f"beta_max_b{b:g}"

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        lo, hi = np.log(self.theta_min), np.log(self.theta_max)
        return np.exp(rng.uniform(lo, hi, size=(n_params, 1)))

    def sample_data(self, params, n, rng):
        u = rng.uniform(size=(len(params), n))
        v = 1.0 - (1.0 - u) ** (1.0 / self.b)
        return params[:, 0:1] * v

    def statistic(self, x):
        return np.max(x, axis=1)

    def true_param(self, params):
        return params[:, 0]

    def true_root_quantiles(self, params, levels, n):
        return -params[:, 0:1] * (1.0 - levels[None, :] ** (1.0 / n)) \
            ** (1.0 / self.b)


class StudentTMeanFamily:
    """X = mu + sigma * t(nu); T_n = mean, T(F) = mu (nu > 1)."""

    kind = 'mean'

    def __init__(self, nu_min: float, nu_max: float, name_suffix: str):
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.name = f"t_mean_{name_suffix}"

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        nu = rng.uniform(self.nu_min, self.nu_max, size=n_params)
        sigma = np.exp(rng.uniform(np.log(0.5), np.log(5.0), size=n_params))
        mu = rng.uniform(-2.0, 2.0, size=n_params)
        return np.stack([nu, sigma, mu], axis=1)

    def sample_data(self, params, n, rng):
        nu = params[:, 0:1]
        return params[:, 2:3] + params[:, 1:2] * rng.standard_t(
            np.broadcast_to(nu, (len(params), n)))

    def statistic(self, x):
        return np.mean(x, axis=1)

    def true_param(self, params):
        return params[:, 2]


class BurrHillFamily:
    """P(X/scale > x) = (1 + x^c)^(-k_burr) with c*k_burr = alpha: same tail
    index as Pareto(alpha) but different second-order behavior."""

    kind = 'hill'

    def __init__(self, k: int = 34):
        self.k = k
        self.name = "burr_hill"

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        alpha = rng.uniform(1.5, 4.0, size=n_params)
        c = rng.uniform(1.0, 3.0, size=n_params)
        scale = np.exp(rng.uniform(np.log(0.5), np.log(5.0), size=n_params))
        return np.stack([alpha, c, scale], axis=1)

    def sample_data(self, params, n, rng):
        alpha, c, scale = (params[:, 0:1], params[:, 1:2], params[:, 2:3])
        k_burr = alpha / c
        u = rng.uniform(size=(len(params), n))
        return scale * (u ** (-1.0 / k_burr) - 1.0) ** (1.0 / c)

    def statistic(self, x):
        return hill_estimator(x, k=self.k)

    def true_param(self, params):
        return 1.0 / params[:, 0]


class FrechetHillFamily:
    """Frechet(alpha): F(x) = exp(-(x/scale)^-alpha); tail index alpha."""

    kind = 'hill'

    def __init__(self, k: int = 34):
        self.k = k
        self.name = "frechet_hill"

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        alpha = rng.uniform(1.5, 4.0, size=n_params)
        scale = np.exp(rng.uniform(np.log(0.5), np.log(5.0), size=n_params))
        return np.stack([alpha, scale], axis=1)

    def sample_data(self, params, n, rng):
        u = rng.uniform(size=(len(params), n))
        return params[:, 1:2] * (-np.log(u)) ** (-1.0 / params[:, 0:1])

    def statistic(self, x):
        return hill_estimator(x, k=self.k)

    def true_param(self, params):
        return 1.0 / params[:, 0]


class StudentTVaRFamily:
    """X = mu + sigma * t(nu); T_n = empirical VaR_0.99; T(F) analytic."""

    kind = 'var'
    var_level = 0.99

    def __init__(self, nu_min: float = 2.5, nu_max: float = 8.0):
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.name = "t_var99"

    def sample_params(self, n_params: int, rng: Generator) -> np.ndarray:
        nu = rng.uniform(self.nu_min, self.nu_max, size=n_params)
        sigma = np.exp(rng.uniform(np.log(0.5), np.log(5.0), size=n_params))
        mu = rng.uniform(-2.0, 2.0, size=n_params)
        return np.stack([nu, sigma, mu], axis=1)

    def sample_data(self, params, n, rng):
        nu = params[:, 0:1]
        return params[:, 2:3] + params[:, 1:2] * rng.standard_t(
            np.broadcast_to(nu, (len(params), n)))

    def statistic(self, x):
        return np.quantile(x, self.var_level, axis=1)

    def true_param(self, params):
        return params[:, 2] + params[:, 1] * t_dist.ppf(self.var_level,
                                                        params[:, 0])
