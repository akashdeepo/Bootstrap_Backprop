"""
Bayes oracle for the Pareto/Hill family (second regret-to-Bayes anchor,
alongside the uniform-max oracle).

Under alpha ~ U(a1, a2), c ~ LogUniform(c1, c2), X ~ c * Pareto(alpha):
integrating c out of the likelihood alpha^n c^(n alpha) prod x_i^(-alpha-1)
against pi(c) propto 1/c on [c1, min(c2, X_(1))] gives

    p(alpha | X) propto alpha^(n-1) * exp(-alpha * S_L)
                        * (1 - (c1/L)^(n alpha)),  alpha in [a1, a2]

with L = min(c2, X_(1)) and S_L = sum_i log(x_i / L). The posterior
depends on the data only through (L, S_L).

Given alpha, the root of the Hill estimator is EXACTLY
gamma * (Gamma(k)/k - 1) (Renyi representation), so the posterior
predictive CDF is a 1-D mixture

    F(t | X) = E_post[ GammaCDF(k * (1 + t * alpha); k) ]

computed on an alpha grid and inverted on a t grid, chunked over datasets.
"""

import numpy as np
from scipy.special import gammainc

from . import config as cfg


def hill_bayes_root_quantiles(X: np.ndarray, k: int, levels: np.ndarray,
                              a1: float, a2: float, c1: float, c2: float,
                              n_alpha: int = 300, n_t: int = 400,
                              chunk: int = 500) -> np.ndarray:
    """Posterior-predictive root quantiles per dataset. Returns (N, L)."""
    N, n = X.shape
    x_min = np.min(X, axis=1)
    L = np.minimum(c2, x_min)
    S = np.sum(np.log(X), axis=1) - n * np.log(L)

    alpha_g = np.linspace(a1, a2, n_alpha)                    # (A,)
    out = np.empty((N, len(levels)))

    for i0 in range(0, N, chunk):
        Ls = L[i0:i0 + chunk]
        Ss = S[i0:i0 + chunk]
        C = len(Ls)

        # log posterior on the alpha grid, normalized per dataset
        log_post = ((n - 1) * np.log(alpha_g)[None, :]
                    - Ss[:, None] * alpha_g[None, :])         # (C, A)
        # boundary term (1 - (c1/L)^(n alpha)); underflows to 1 usually
        ratio_term = -np.expm1(n * alpha_g[None, :]
                               * np.log(np.maximum(c1 / Ls[:, None], 1e-300)))
        log_post += np.log(np.maximum(ratio_term, 1e-300))
        log_post -= log_post.max(axis=1, keepdims=True)
        w = np.exp(log_post)
        w /= w.sum(axis=1, keepdims=True)                     # (C, A)

        # t grid per dataset, scaled by the posterior-mean gamma
        g_hat = w @ (1.0 / alpha_g)                           # (C,)
        t = g_hat[:, None] * np.linspace(-0.85, 1.2, n_t)[None, :]  # (C, T)

        # F(t | X) = sum_a w_a * GammaCDF(k (1 + t alpha_a); k)
        arg = k * (1.0 + t[:, None, :] * alpha_g[None, :, None])  # (C, A, T)
        F = np.einsum('ca,cat->ct', w,
                      gammainc(k, np.maximum(arg, 0.0)))      # (C, T)

        for c in range(C):
            out[i0 + c] = np.interp(levels, F[c], t[c])

    return out
