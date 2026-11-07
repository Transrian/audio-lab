"""Confidence intervals that resample whole groups, not single utterances.

Utterances of one speaker (or one FLEURS sentence) share an accent, a recording
and a topic, so they fail together. Resampling utterances alone would pretend
those failures are independent and give intervals that are too narrow, which is
the classic way an evaluation claims a difference that is not there. Resampling
groups keeps the correlation: every draw takes whole speakers.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# Ten thousand draws is the usual choice for a percentile interval: enough that
# the interval itself moves by less than the rounding of the report.
DRAWS = 10_000
SEED = 20261107
CONFIDENCE = 0.95

Totals = Mapping[str, tuple[float, float]]

# How many draws are resampled at once; see :func:`_draw_sums`.
DRAW_CHUNK = 500


def ratio(totals: Totals) -> float:
    """Sum of numerators over sum of denominators, 0 when nothing was counted."""
    numerator = sum(pair[0] for pair in totals.values())
    denominator = sum(pair[1] for pair in totals.values())
    return numerator / denominator if denominator else 0.0


def difference_ratio(totals_a: Totals, totals_b: Totals) -> float:
    """The same ratio for B minus the one for A, for paired panels."""
    return ratio(totals_b) - ratio(totals_a)


def ratio_interval(
    totals: Totals, *, draws: int = DRAWS, seed: int = SEED, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Percentile interval of :func:`ratio` over resampled groups."""
    sums = _draw_sums(_matrix(totals), draws=draws, seed=seed)
    return _read_interval(_ratios(sums[:, 0], sums[:, 1]), confidence)


def difference_interval(
    totals_a: Totals,
    totals_b: Totals,
    *,
    draws: int = DRAWS,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """Percentile interval of the paired difference ``B - A`` over resampled groups.

    The two runs are resampled together: a draw takes the same groups from both,
    which is what makes the difference paired. For each draw the two ratios are
    formed separately and then subtracted, exactly as the reported difference is.
    """
    shared = sorted(set(totals_a) & set(totals_b))
    values = np.array([(*totals_a[group], *totals_b[group]) for group in shared], dtype=float)
    sums = _draw_sums(values, draws=draws, seed=seed)
    difference = _ratios(sums[:, 2], sums[:, 3]) - _ratios(sums[:, 0], sums[:, 1])
    return _read_interval(difference, confidence)


def _matrix(totals: Totals) -> np.ndarray:
    """The totals as one row per group, in sorted group order.

    Sorting is what makes an interval a function of the numbers alone: the same
    panel gives the same interval whatever order its groups were stored in.
    """
    return np.array([totals[group] for group in sorted(totals)], dtype=float).reshape(-1, 2)


def _draw_sums(values: np.ndarray, *, draws: int, seed: int) -> np.ndarray:
    """``draws`` resamples of whole groups with replacement, each one summed.

    Ten thousand draws of a 350 group panel is a few hundred megabytes if it is
    built at once, so the draws are made in chunks; the generator is consumed in
    one fixed order, so the result does not depend on the chunk size.
    """
    if not len(values):
        # No group means no draw, and the interval of no draws is the point 0.
        return np.zeros((0, values.shape[1]), dtype=float)
    rng = np.random.default_rng(seed)
    sums = np.empty((draws, values.shape[1]), dtype=float)
    for start in range(0, draws, DRAW_CHUNK):
        size = min(DRAW_CHUNK, draws - start)
        picked = rng.integers(0, len(values), size=(size, len(values)))
        sums[start : start + size] = values[picked].sum(axis=1)
    return sums


def _ratios(numerators: np.ndarray, denominators: np.ndarray) -> np.ndarray:
    """Numerator over denominator per draw, 0 where nothing was counted."""
    return np.divide(
        numerators, denominators, out=np.zeros_like(numerators), where=denominators != 0
    )


def _read_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    """The percentile interval of the draws, or a point when there is nothing to draw."""
    if not values.size:
        return (0.0, 0.0)
    low, high = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    return (float(np.percentile(values, low * 100)), float(np.percentile(values, high * 100)))


def verdict(low: float, high: float, name_a: str, name_b: str) -> str:
    """A difference is only a difference when the interval stays away from zero."""
    if low > 0:
        return f"{name_a} better"
    if high < 0:
        return f"{name_b} better"
    return "not separable"
