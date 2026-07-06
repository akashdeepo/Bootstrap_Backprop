"""
Fast CPU test suite: analytic-vs-MC agreement for the exact root
distributions, sampler validation, calibration round-trips, quantile-head
monotonicity, and seed determinism. Everything runs in seconds without a
GPU.
"""

import numpy as np
import torch
import pytest
from numpy.random import default_rng

from amortized_bootstrap import config as cfg
from amortized_bootstrap.model import QuantileNet
from amortized_bootstrap.training import pinball_loss
from amortized_bootstrap.calibration import (empirical_coverage_curve,
                                             adjusted_levels,
                                             apply_recalibration)
from amortized_bootstrap.datagen import featurize
from amortized_bootstrap.families import (UniformMaxFamily,
                                          ParetoHillFamily,
                                          sample_sas_vectorized)
from amortized_bootstrap.ood_families import BetaMaxFamily


LEVELS = cfg.QUANTILE_LEVELS


def test_quantile_head_is_monotone():
    torch.manual_seed(0)
    net = QuantileNet(n_input=50, n_aux=2, n_levels=19, hidden=32, depth=2)
    z = torch.randn(8, 50)
    aux = torch.randn(8, 2)
    q = net(z, aux)
    assert torch.all(q[:, 1:] >= q[:, :-1]).item()


def test_pinball_loss_hand_computed():
    # one sample t=1.0, two levels 0.25 / 0.75, predictions 0 and 2
    q = torch.tensor([[0.0, 2.0]])
    t = torch.tensor([[1.0]])
    lv = torch.tensor([0.25, 0.75])
    # level .25, diff=1  -> 0.25*1 = 0.25 ; level .75, diff=-1 -> 0.25*1
    expected = (0.25 + 0.25) / 2
    assert abs(pinball_loss(q, t, lv).item() - expected) < 1e-6


def test_recalibration_identity_when_calibrated():
    tau_adj = adjusted_levels(LEVELS, LEVELS.copy())
    assert np.allclose(tau_adj, LEVELS, atol=1e-12)
    q = np.sort(default_rng(0).normal(size=(5, len(LEVELS))), axis=1)
    assert np.allclose(apply_recalibration(q, LEVELS, tau_adj), q,
                       atol=1e-9)


def test_recalibration_fixes_overwide_model():
    rng = default_rng(1)
    n_val = 40_000
    t = rng.normal(size=n_val)
    from scipy.stats import norm
    # model predicts N(0, 2) quantiles -- too wide by 2x
    q_pred = np.tile(2.0 * norm.ppf(LEVELS), (n_val, 1))
    h = empirical_coverage_curve(q_pred, t, LEVELS)
    tau_adj = adjusted_levels(LEVELS, h)
    q_recal = apply_recalibration(q_pred, LEVELS, tau_adj)
    h_after = empirical_coverage_curve(q_recal, t, LEVELS)
    interior = (LEVELS > 0.05) & (LEVELS < 0.95)
    assert np.max(np.abs(h_after[interior] - LEVELS[interior])) < 0.02


def test_uniform_max_analytic_root_matches_mc():
    fam = UniformMaxFamily()
    rng = default_rng(2)
    theta, n, reps = 2.0, 50, 40_000
    x = rng.uniform(0, theta, size=(reps, n))
    roots = x.max(axis=1) - theta
    q_mc = np.quantile(roots, [0.1, 0.5, 0.9])
    q_an = fam.true_root_quantiles(np.array([theta]),
                                   np.array([0.1, 0.5, 0.9]), n)[0]
    assert np.allclose(q_mc, q_an, rtol=0.08)


def test_beta_max_analytic_root_matches_mc():
    fam = BetaMaxFamily(b=2.0)
    rng = default_rng(3)
    params = np.array([[1.5]])
    n, reps = 50, 40_000
    x = fam.sample_data(np.repeat(params, reps, axis=0), n, rng)
    roots = x.max(axis=1) - 1.5
    q_mc = np.quantile(roots, [0.1, 0.5, 0.9])
    q_an = fam.true_root_quantiles(params, np.array([0.1, 0.5, 0.9]), n)[0]
    assert np.allclose(q_mc, q_an, rtol=0.08)


def test_hill_root_is_exactly_gamma_for_pareto():
    fam = ParetoHillFamily(k=20)
    rng = default_rng(4)
    alpha = 2.5
    params = np.tile([alpha, 1.0], (30_000, 1))
    x = fam.sample_data(params, 100, rng)
    roots = fam.statistic(x) - 1.0 / alpha
    q_mc = np.quantile(roots, [0.25, 0.5, 0.75])
    q_an = fam.true_root_quantiles(params[:1],
                                   np.array([0.25, 0.5, 0.75]))[0]
    # compare in units of the root scale gamma/sqrt(k)
    scale = (1.0 / alpha) / np.sqrt(20)
    assert np.max(np.abs(q_mc - q_an)) / scale < 0.05


def test_stable_sampler_gaussian_limit():
    # SaS(alpha -> 2, scale 1) is N(0, 2); check std and symmetry
    rng = default_rng(5)
    draws = sample_sas_vectorized(np.float64(1.9999), (200_000,), rng)
    assert abs(np.std(draws) - np.sqrt(2.0)) < 0.03
    assert abs(np.mean(draws)) < 0.02


def test_fresh_rng_is_deterministic_and_matches_stream():
    a = cfg.fresh_rng(2).uniform(size=10)
    b = cfg.fresh_rng(2).uniform(size=10)
    assert np.array_equal(a, b)
    c = cfg.fresh_rng(3).uniform(size=10)
    assert not np.array_equal(a, c)


def test_featurize_scale_equivariance():
    rng = default_rng(6)
    X = rng.uniform(1.0, 3.0, size=(20, 50))
    z1, aux1, s1 = featurize(X)
    z2, aux2, s2 = featurize(10.0 * X)
    assert np.allclose(z1, z2, atol=1e-5)
    assert np.allclose(s2, 10.0 * s1, rtol=1e-9)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
