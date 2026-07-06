"""
Distribution samplers for all 5 (T, F) pairs.

Each sampler returns an array of shape (n_samples, sample_size) -- i.e., a batch
of independent samples, each of length `sample_size`.

Implemented:
  - Normal(0, 1)
  - Uniform(0, theta)
  - Pareto(alpha, x_min)
  - Stable(alpha, beta)        via custom CMS algorithm
  - NTS(alpha, theta, beta, mu, sigma)  via subordinator rejection
"""

import numpy as np
from numpy.random import Generator


# ===============================================
# Trivial samplers
# ===============================================

def sample_normal(n_samples: int, sample_size: int, rng: Generator,
                  mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Sample from N(mu, sigma^2). Returns shape (n_samples, sample_size)."""
    return rng.normal(loc=mu, scale=sigma, size=(n_samples, sample_size))


def sample_uniform(n_samples: int, sample_size: int, rng: Generator,
                   theta: float = 1.0) -> np.ndarray:
    """Sample from Uniform(0, theta). Returns shape (n_samples, sample_size)."""
    return rng.uniform(low=0.0, high=theta, size=(n_samples, sample_size))


def sample_pareto(n_samples: int, sample_size: int, rng: Generator,
                  alpha: float = 2.5, x_min: float = 1.0) -> np.ndarray:
    """
    Sample from Pareto(alpha, x_min) via inverse CDF.
    P(X > x) = (x_min / x)^alpha  for x >= x_min.
    Returns shape (n_samples, sample_size).
    """
    u = rng.uniform(size=(n_samples, sample_size))
    return x_min * u ** (-1.0 / alpha)


# ===============================================
# Stable distribution -- custom CMS algorithm
# ===============================================

def _cms_symmetric_stable(alpha: float, size: int, rng: Generator) -> np.ndarray:
    """
    Chambers-Mallows-Stuck algorithm for symmetric stable(alpha, beta=0).

    For beta=0, the CMS formula simplifies to:
        X = sin(alpha * V) / cos(V)^(1/alpha)
            * [cos((1 - alpha) * V) / W]^((1 - alpha) / alpha)

    where V ~ Uniform(-pi/2, pi/2), W ~ Exp(1).

    Returns 1D array of length `size`.
    """
    V = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
    W = rng.exponential(1.0, size=size)

    # Protect against V = 0 exactly (vanishingly rare but causes 0/0)
    V = np.clip(V, -np.pi / 2 + 1e-10, np.pi / 2 - 1e-10)

    term1 = np.sin(alpha * V) / (np.cos(V) ** (1.0 / alpha))
    term2 = (np.cos((1.0 - alpha) * V) / W) ** ((1.0 - alpha) / alpha)
    return term1 * term2


def _cms_positive_stable(gamma: float, size: int, rng: Generator) -> np.ndarray:
    """
    CMS algorithm for totally-skewed positive stable(gamma, beta=1) with gamma in (0, 1).

    Uses the Kanter (1975) representation:
        S = [sin(gamma * (V + pi/2)) / cos(V)^(1/gamma)]
            * [cos(V - gamma * (V + pi/2)) / W]^((1 - gamma) / gamma)

    where V ~ Uniform(-pi/2, pi/2), W ~ Exp(1).
    The result is supported on (0, infinity).

    Returns 1D array of length `size`.
    """
    V = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
    W = rng.exponential(1.0, size=size)

    V = np.clip(V, -np.pi / 2 + 1e-10, np.pi / 2 - 1e-10)

    # Shifted angle for beta=1
    phi = V + np.pi / 2  # in (0, pi)
    term1 = np.sin(gamma * phi) / (np.cos(V) ** (1.0 / gamma))
    angle2 = V - gamma * phi  # = V(1 - gamma) - gamma*pi/2
    term2 = (np.cos(angle2) / W) ** ((1.0 - gamma) / gamma)
    return term1 * term2


def sample_stable(n_samples: int, sample_size: int, rng: Generator,
                  alpha: float = 1.5, beta: float = 0.0) -> np.ndarray:
    """
    Sample from Stable(alpha, beta) with location=0, scale=1.

    Currently only supports beta=0 (symmetric stable).
    Returns shape (n_samples, sample_size).
    """
    if beta != 0.0:
        raise NotImplementedError("Only symmetric stable (beta=0) is implemented.")

    total = n_samples * sample_size
    draws = _cms_symmetric_stable(alpha, total, rng)
    return draws.reshape(n_samples, sample_size)


# ===============================================
# Normal Tempered Stable (NTS) -- subordinator rejection
# ===============================================

def _sample_tempered_stable_subordinator(n_needed: int, gamma: float, theta: float,
                                          rng: Generator) -> np.ndarray:
    """
    Sample from a tempered stable subordinator TS(gamma, theta) via rejection
    from positive stable(gamma).

    Algorithm:
        1. Draw S ~ PositiveStable(gamma)  (support on [0, inf))
        2. Accept with probability exp(-theta * S)
        3. Accepted S values are tempered stable draws

    Acceptance rate = E[exp(-theta * S)] = exp(-theta^gamma).

    Parameters
    ----------
    n_needed : number of samples required
    gamma    : stability index in (0, 1), equals NTS_ALPHA / 2
    theta    : tempering parameter > 0

    Returns
    -------
    1D array of length n_needed, all positive.
    """
    acceptance_rate = np.exp(-theta ** gamma)
    # Oversample to reduce number of rejection rounds
    batch_factor = int(np.ceil(1.5 / acceptance_rate))  # ~4x for our case

    collected = []
    n_collected = 0

    while n_collected < n_needed:
        batch_size = max((n_needed - n_collected) * batch_factor, 1024)
        S = _cms_positive_stable(gamma, batch_size, rng)

        # Reject negatives (shouldn't happen for gamma < 1 with beta=1, but safeguard)
        S = S[S > 0]
        if len(S) == 0:
            continue

        # Accept with probability exp(-theta * S)
        u = rng.uniform(size=len(S))
        accepted = S[u < np.exp(-theta * S)]

        if len(accepted) > 0:
            collected.append(accepted)
            n_collected += len(accepted)

    return np.concatenate(collected)[:n_needed]


def sample_nts(n_samples: int, sample_size: int, rng: Generator,
               alpha: float = 1.5, theta: float = 1.0,
               beta: float = 0.0, mu: float = 0.0,
               sigma: float = 1.0) -> np.ndarray:
    """
    Sample from Normal Tempered Stable NTS(alpha, theta, beta, mu, sigma).

    Construction:
        X = mu + beta * T + sigma * sqrt(T) * Z
    where T ~ TemperedStable(alpha/2, theta), Z ~ N(0, 1), independent.

    For beta=0 (symmetric NTS): X = mu + sigma * sqrt(T) * Z.

    Returns shape (n_samples, sample_size).
    """
    total = n_samples * sample_size
    gamma = alpha / 2.0  # subordinator stability index

    T = _sample_tempered_stable_subordinator(total, gamma, theta, rng)
    Z = rng.standard_normal(total)

    X = mu + beta * T + sigma * np.sqrt(T) * Z
    return X.reshape(n_samples, sample_size)
