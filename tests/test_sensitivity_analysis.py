"""Unit tests for the pure sensitivity-analysis math (no torch / GPU needed)."""

import numpy as np
import pytest

from sensitivity.analysis import (
    build_g_curve,
    select_sensitive_levels,
    beta_binomial_weights,
    pchip_interp,
)


ALPHAS = np.linspace(0.0, 1.0, 11)


def test_build_g_curve_shape_and_normalization():
    # Accuracy decreasing with alpha, KID increasing — the typical case.
    ma = np.linspace(0.8, 0.2, 11)
    kid = np.linspace(0.0, 4.0, 11)
    g = build_g_curve(ALPHAS, ma, kid, lam=0.05)
    assert g.shape == ALPHAS.shape
    # KID term is normalized by its max -> at alpha_max it equals 1.0.
    # g(alpha_max) = 1 - ma_max - 1 + lam*1 = -ma_min + lam
    assert g[-1] == pytest.approx(1.0 - ma[-1] - 1.0 + 0.05 * 1.0)


def test_build_g_curve_zero_kid_safe():
    ma = np.linspace(0.8, 0.2, 11)
    kid = np.zeros(11)
    g = build_g_curve(ALPHAS, ma, kid, lam=0.0)
    # No degradation term -> g = 1 - ma exactly.
    np.testing.assert_allclose(g, 1.0 - ma)


def test_select_levels_count_and_range():
    ma = np.linspace(0.9, 0.1, 11)
    kid = np.linspace(0.0, 5.0, 11)
    g = build_g_curve(ALPHAS, ma, kid)
    levels = select_sensitive_levels(ALPHAS, g, num_levels=5)
    assert levels.shape == (5,)
    assert np.all(levels >= 0.0) and np.all(levels <= 1.0)
    # Sorted ascending.
    assert np.all(np.diff(levels) >= 0)


def test_select_levels_flat_g_fallback():
    g = np.zeros_like(ALPHAS)
    levels = select_sensitive_levels(ALPHAS, g, num_levels=4)
    # Flat g -> even spread over the alpha range.
    np.testing.assert_allclose(levels, np.linspace(0.0, 1.0, 4))


def test_beta_binomial_favours_worse_levels():
    # Levels with these accuracies; the lowest-MA level should get most mass.
    ma_at_levels = np.array([0.2, 0.5, 0.7, 0.9])
    w = beta_binomial_weights(ma_at_levels)
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0)
    # Worst (index 0, MA=0.2) gets the largest weight; best (MA=0.9) the smallest.
    assert w.argmax() == 0
    assert w[0] > w[1] > w[2] > w[3]


def test_beta_binomial_order_independence():
    # Same accuracies in a different order -> mass still tracks the worst level.
    ma = np.array([0.9, 0.2, 0.7, 0.5])
    w = beta_binomial_weights(ma)
    assert w.argmax() == 1  # index of MA=0.2


def test_beta_binomial_singleton():
    w = beta_binomial_weights(np.array([0.5]))
    np.testing.assert_allclose(w, [1.0])


def test_pchip_interp_passes_through_samples():
    y = np.array([0.0, 1.0, 0.5, 0.8])
    x = np.array([0.0, 0.3, 0.6, 1.0])
    out = pchip_interp(x, y, x)
    np.testing.assert_allclose(out, y, atol=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
