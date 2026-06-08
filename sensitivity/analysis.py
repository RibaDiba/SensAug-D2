"""Pure sensitivity-analysis math (no torch).

Implements the core of the SensAug method (Zheng et al., arXiv:2406.01425):
given, for one augmentation type, the model accuracy ``MA(alpha)`` and the
perceptual degradation ``KID(alpha)`` measured across a sweep of perturbation
magnitudes ``alpha``, derive

1. the trade-off curve ``g(alpha)``,
2. ``L`` "sensitive" intensity levels, and
3. a Beta-Binomial sampling distribution over those levels that favours the
   worse-performing (lower-accuracy) ones.

All functions operate on plain numpy arrays so they can be unit-tested without a
GPU or a model.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import betabinom


def build_g_curve(
    alphas: np.ndarray,
    ma: np.ndarray,
    kid: np.ndarray,
    lam: float = 0.05,
) -> np.ndarray:
    """Build the SensAug trade-off curve ``g(alpha)``.

    Follows the paper's formulation::

        g(alpha) = 1 - MA(alpha) - KID(alpha) / KID(alpha_max) + lambda * alpha

    * ``1 - MA(alpha)`` grows as the model gets *worse* under the perturbation.
    * ``KID(alpha) / KID(alpha_max)`` is the perceptual degradation, normalized
      by the degradation at the strongest magnitude so it lies in ``[0, 1]``.
    * ``lambda * alpha`` is a mild regularizer that spaces intensity levels out.

    Args:
        alphas: Sweep magnitudes, shape ``(K,)``, assumed sorted ascending.
        ma: Model accuracy at each ``alpha`` (e.g. mean mask mIoU), shape ``(K,)``.
        kid: Perceptual degradation (KID) at each ``alpha``, shape ``(K,)``.
        lam: Regularizer weight ``lambda``.

    Returns:
        ``g(alpha)`` evaluated at ``alphas``, shape ``(K,)``.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    ma = np.asarray(ma, dtype=np.float64)
    kid = np.asarray(kid, dtype=np.float64)

    kid_max = float(kid.max()) if kid.size else 0.0
    if kid_max > 0.0:
        kid_norm = kid / kid_max
    else:
        kid_norm = np.zeros_like(kid)

    return 1.0 - ma - kid_norm + lam * alphas


def pchip_interp(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Monotone-cubic (PCHIP) interpolation of ``y`` over ``x`` at ``query``.

    PCHIP avoids the overshoot of ordinary cubic splines, matching the paper's
    use of shape-preserving interpolation when reconstructing ``g`` curves.

    Args:
        x: Sample locations, shape ``(K,)``, strictly increasing.
        y: Sample values, shape ``(K,)``.
        query: Locations to evaluate at, any shape.

    Returns:
        Interpolated values at ``query`` (clamped to the ``x`` range).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)

    if x.size < 2:
        # Degenerate: nothing to interpolate — broadcast the single value.
        return np.full_like(query, float(y[0]) if y.size else 0.0)

    interp = PchipInterpolator(x, y, extrapolate=False)
    q = np.clip(query, x[0], x[-1])
    return interp(q)


def select_sensitive_levels(
    alphas: np.ndarray,
    g: np.ndarray,
    num_levels: int,
    fine_grid: int = 200,
) -> np.ndarray:
    """Select ``num_levels`` intensity levels equally spaced along ``g``'s range.

    The model is *most sensitive* where ``g`` changes fastest with ``alpha``.
    Sampling magnitudes that are equally spaced in ``g``-value (rather than in
    ``alpha``) therefore concentrates the chosen levels in those high-slope,
    high-sensitivity regions — the behaviour described in the paper.

    Args:
        alphas: Sweep magnitudes, shape ``(K,)``, sorted ascending.
        g: ``g(alpha)`` values, shape ``(K,)``.
        num_levels: Number ``L`` of intensity levels to return.
        fine_grid: Resolution of the PCHIP-interpolated grid used for selection.

    Returns:
        ``num_levels`` magnitudes in ``[alphas[0], alphas[-1]]``, sorted ascending.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    num_levels = int(num_levels)

    if num_levels <= 0:
        return np.array([], dtype=np.float64)
    if alphas.size <= 1:
        return np.full(num_levels, float(alphas[0]) if alphas.size else 0.0)

    a_fine = np.linspace(alphas[0], alphas[-1], fine_grid)
    g_fine = pchip_interp(alphas, g, a_fine)

    g_lo, g_hi = float(g_fine.min()), float(g_fine.max())
    if g_hi - g_lo < 1e-9:
        # g is essentially flat (no sensitivity signal) — fall back to an even
        # spread over the magnitude range.
        return np.linspace(alphas[0], alphas[-1], num_levels)

    # Target g-values equally spaced across the observed g range; map each back
    # to the alpha whose g is closest.
    targets = np.linspace(g_lo, g_hi, num_levels)
    levels = np.array(
        [a_fine[int(np.argmin(np.abs(g_fine - t)))] for t in targets],
        dtype=np.float64,
    )

    levels = np.unique(levels)
    if levels.size < num_levels:
        # Duplicates collapsed (e.g. plateaus) — top up with an even spread.
        extra = np.linspace(alphas[0], alphas[-1], num_levels)
        levels = np.unique(np.concatenate([levels, extra]))[:num_levels]
    return np.sort(levels)


def beta_binomial_weights(
    ma_at_levels: np.ndarray,
    a: float = 1.0,
    b: float = 3.0,
) -> np.ndarray:
    """Beta-Binomial sampling weights over levels, favouring worse-performing ones.

    Levels are ranked by accuracy ``MA`` (ascending — worst first) and assigned
    Beta-Binomial probability mass.  With ``a < b`` the mass concentrates on the
    low ranks, i.e. the levels where the model performs *worst*, so training
    samples those sensitive magnitudes more often.

    Args:
        ma_at_levels: Model accuracy at each selected level, shape ``(L,)``.
        a: Beta-Binomial ``alpha`` shape parameter.
        b: Beta-Binomial ``beta`` shape parameter.

    Returns:
        Probability vector aligned with the *input order* of ``ma_at_levels``,
        shape ``(L,)``, summing to 1.
    """
    ma = np.asarray(ma_at_levels, dtype=np.float64)
    n = ma.size
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    # pmf over ranks 0..n-1; rank 0 (highest mass when a<b) = worst performer.
    ranks = np.arange(n)
    pmf = betabinom.pmf(ranks, n - 1, a, b)
    pmf = np.where(np.isfinite(pmf), pmf, 0.0)

    # Worst-performing level (lowest MA) -> rank 0.  ``argsort`` of MA ascending
    # gives the order; invert it to map each level to its rank.
    order = np.argsort(ma, kind="stable")
    rank_of_level = np.empty(n, dtype=int)
    rank_of_level[order] = ranks

    weights = pmf[rank_of_level]
    total = weights.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return weights / total
