"""Confidence intervals that resample whole groups, not single utterances.

Utterances of one speaker (or one FLEURS sentence) share an accent, a recording
and a topic, so they fail together. Resampling utterances alone would pretend
those failures are independent and give intervals that are too narrow, which is
the classic way an evaluation claims a difference that is not there. Resampling
groups keeps the correlation: every draw takes whole speakers.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping

import numpy as np

# Ten thousand draws is the usual choice for a percentile interval: enough that
# the interval itself moves by less than the rounding of the report.
DRAWS = 10_000
SEED = 20261107
CONFIDENCE = 0.95

Totals = Mapping[str, tuple[float, float]]
Statistic = Callable[[Totals], float]


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
    return interval(totals, ratio, draws=draws, seed=seed, confidence=confidence)


def difference_interval(
    totals_a: Totals,
    totals_b: Totals,
    *,
    draws: int = DRAWS,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """Percentile interval of the paired difference ``B - A`` over resampled groups."""
    shared = sorted(set(totals_a) & set(totals_b))
    return interval(
        {
            group: (totals_a[group][0], totals_a[group][1], totals_b[group][0], totals_b[group][1])
            for group in shared
        },
        lambda totals: difference_ratio(
            {group: pair[:2] for group, pair in totals.items()},
            {group: pair[2:] for group, pair in totals.items()},
        ),
        draws=draws,
        seed=seed,
        confidence=confidence,
    )


def interval(
    totals: Totals,
    statistic: Statistic,
    *,
    draws: int = DRAWS,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """Draw whole groups with replacement and read the interval off the draws.

    The draw depends only on ``seed`` and on the sorted group names, so the same
    totals always produce the same interval.
    """
    names = sorted(totals)
    if not names:
        return (0.0, 0.0)
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = {}
        for _ in range(len(names)):
            name = names[rng.randrange(len(names))]
            first = totals[name]
            running = sample.get(name, (0.0,) * len(first))
            sample[name] = tuple(total + value for total, value in zip(running, first, strict=True))
        values.append(statistic(sample))
    values.sort()
    low, high = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    return (float(np.percentile(values, low * 100)), float(np.percentile(values, high * 100)))


def verdict(low: float, high: float, name_a: str, name_b: str) -> str:
    """A difference is only a difference when the interval stays away from zero."""
    if low > 0:
        return f"{name_a} better"
    if high < 0:
        return f"{name_b} better"
    return "not separable"
