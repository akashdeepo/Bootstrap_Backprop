"""
Tests for the pretrained root-network API. These load the actual
committed checkpoints, so they also guard the checkpoint/npz contract
(target_scale, c0, tau_adj present and consistent).
"""

import numpy as np
import pytest
from numpy.random import default_rng

from amortized_bootstrap import pretrained


def test_interval_single_dataset_max():
    x = default_rng(0).uniform(0, 2.0, size=200)
    lo, hi = pretrained.interval(x, statistic="max", level=0.95)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    # endpoint sits above the sample max, and not absurdly far
    assert hi > x.max()
    assert hi < 3.0 * x.max()


def test_interval_batch_shapes_and_level_monotonicity():
    x = default_rng(1).standard_t(3.0, size=(8, 200))
    lo95, hi95 = pretrained.interval(x, statistic="mean", level=0.95)
    lo90, hi90 = pretrained.interval(x, statistic="mean", level=0.90)
    assert lo95.shape == hi95.shape == (8,)
    assert np.all(lo95 < hi95)
    # wider level -> wider interval
    assert np.all(hi95 - lo95 >= hi90 - lo90)


def test_root_quantiles_monotone_all_statistics():
    rng = default_rng(2)
    data = {
        "max": rng.uniform(0, 1.5, size=(4, 200)),
        "mean": rng.standard_t(2.5, size=(4, 200)),
        "hill": (rng.uniform(size=(4, 200)) ** (-1.0 / 2.5)),
        "var99": rng.standard_t(4.0, size=(4, 200)),
    }
    for stat, x in data.items():
        levels, q = pretrained.root_quantiles(x, stat)
        assert q.shape == (4, len(levels))
        assert np.all(np.diff(q, axis=1) >= -1e-9), stat


def test_wrong_n_raises():
    with pytest.raises(ValueError):
        pretrained.interval(np.ones(100), statistic="mean")


def test_unknown_statistic_raises():
    with pytest.raises(ValueError):
        pretrained.interval(np.ones(200), statistic="median")


def test_coverage_sanity_uniform_max():
    # 200 datasets from Uniform(0, theta): in-prior, coverage should be
    # near nominal; loose bound to keep the test robust and fast
    rng = default_rng(3)
    theta = 2.0
    x = rng.uniform(0, theta, size=(200, 200))
    lo, hi = pretrained.interval(x, statistic="max", level=0.95)
    cov = np.mean((theta >= lo) & (theta <= hi))
    assert cov > 0.85, f"coverage {cov:.3f} too far from nominal"


def test_width_tracking_diagnostic_responds():
    rng = default_rng(4)
    x = rng.uniform(0, 1.0, size=(50, 200))
    d = pretrained.width_tracking_diagnostic(x, statistic="max")
    assert d['widths'].shape == (50,)
    assert d['noise_response'] > 0.1
