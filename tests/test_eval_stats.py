"""Statistics: grouped bootstrap intervals and the paired verdict."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audiolab.eval import stats  # noqa: E402


def totals(spec: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    return spec


def test_ratio_is_a_sum_over_a_sum():
    assert stats.ratio(totals({"a": (1.0, 4.0), "b": (1.0, 4.0)})) == 0.25
    assert stats.ratio({}) == 0.0


def test_interval_brackets_the_point_estimate():
    many = {f"speaker-{index:02d}": (float(index % 3), 10.0) for index in range(40)}
    low, high = stats.ratio_interval(many, draws=400)
    assert low < stats.ratio(many) < high
    assert 0.0 <= low <= high <= 1.0


def test_interval_is_reproducible():
    many = {f"speaker-{index:02d}": (float(index % 3), 10.0) for index in range(40)}
    assert stats.ratio_interval(many, draws=200) == stats.ratio_interval(many, draws=200)


def test_a_single_group_gives_a_degenerate_interval():
    assert stats.ratio_interval(totals({"only": (1.0, 4.0)}), draws=50) == (0.25, 0.25)


def test_no_groups_gives_no_spread():
    assert stats.ratio_interval({}, draws=50) == (0.0, 0.0)


def test_identical_groups_give_the_interval_of_that_one_value():
    """Hand-checkable: every resample of identical groups is the same panel.

    Six groups of (2 errors, 5 words) sum to (12, 30) in any draw, so the ratio
    is 0.4 in every one of them and the interval has no width.
    """
    totals_ = {f"g{index}": (2.0, 5.0) for index in range(6)}
    assert stats.ratio_interval(totals_, draws=500) == (0.4, 0.4)


def test_a_missing_group_cannot_take_the_interval_below_the_point():
    """Hand-checkable: one group carries all the errors, and a draw usually has it.

    A draw that misses the only group with errors counts nothing at all, which
    is the point 0, so the interval reaches from a very unlikely 0 to the point
    value: the whole spread of this panel is the question "did we pick it".
    """
    totals_ = {"errors": (3.0, 3.0), "clean": (0.0, 3.0)}
    low, high = stats.ratio_interval(totals_, draws=2_000)
    assert low == 0.0
    assert 0.0 < high <= 1.0


def test_the_interval_does_not_depend_on_the_order_of_the_groups():
    """The same numbers give the same interval, however they were assembled."""
    first = {"s1": (3.0, 10.0), "s2": (1.0, 10.0), "s3": (0.0, 10.0)}
    second = {"s3": (0.0, 10.0), "s1": (3.0, 10.0), "s2": (1.0, 10.0)}
    assert stats.ratio_interval(first, draws=300) == stats.ratio_interval(second, draws=300)


def test_difference_interval_is_zero_for_identical_runs():
    many = {f"speaker-{index:02d}": (float(index % 3), 10.0) for index in range(20)}
    low, high = stats.difference_interval(many, dict(many), draws=200)
    assert low == high == 0.0
    assert stats.verdict(low, high, "A", "B") == "not separable"


def test_difference_interval_finds_a_clear_difference():
    good = {f"speaker-{index:02d}": (0.0, 100.0) for index in range(20)}
    bad = {f"speaker-{index:02d}": (30.0, 100.0) for index in range(20)}
    low, high = stats.difference_interval(good, bad, draws=300)
    assert low > 0.0
    assert stats.verdict(low, high, "good", "bad") == "good better"


def test_verdict_names_the_better_run_either_way():
    assert stats.verdict(-0.2, -0.05, "A", "B") == "B better"
    assert stats.verdict(-0.05, 0.05, "A", "B") == "not separable"


def test_difference_uses_only_the_groups_both_runs_share():
    a = {"s1": (1.0, 10.0), "s2": (2.0, 10.0)}
    b = {"s2": (2.0, 10.0), "s3": (3.0, 10.0)}
    low, high = stats.difference_interval(a, b, draws=50)
    assert low <= 0.0 <= high  # only s2 is compared, and it is identical there


def test_the_difference_interval_is_the_interval_of_its_draws():
    """Hand-checkable only against each other: the two paths must not drift.

    The totals vary across groups, so a difference in seed, in draw order or in
    which columns are read moves the numbers; the empty shared case is the
    documented point 0 of no draws.
    """
    a = {f"s{index:02d}": (float(index % 5), 10.0 + index / 7) for index in range(30)}
    b = {f"s{index:02d}": (float((index * 3) % 7), 10.0 + index / 7) for index in range(30)}
    for seed in (1, 20261107):
        assert stats.interval_of(
            stats.difference_draws(a, b, draws=500, seed=seed)
        ) == stats.difference_interval(a, b, draws=500, seed=seed)
    assert stats.interval_of(stats.difference_draws(a, b, draws=500)) == stats.difference_interval(
        a, b, draws=500
    )
    # No shared group: no draw to read, and the interval of no draw is the point 0.
    assert stats.difference_draws({"a": (1.0, 4.0)}, {"b": (2.0, 4.0)}, draws=50).size == 0
    assert stats.interval_of(stats.difference_draws({}, {"b": (2.0, 4.0)}, draws=50)) == (0.0, 0.0)


def test_a_non_default_confidence_moves_the_interval_of_the_same_draws():
    """The delegated path still reads the confidence it was asked for."""
    a = {f"s{index:02d}": (float(index % 5), 10.0) for index in range(30)}
    b = {f"s{index:02d}": (float((index * 3) % 7), 10.0) for index in range(30)}
    narrow = stats.difference_interval(a, b, draws=500, seed=3, confidence=0.5)
    wide = stats.difference_interval(a, b, draws=500, seed=3, confidence=0.99)
    assert narrow[0] > wide[0] and narrow[1] < wide[1]
    assert narrow == stats.interval_of(stats.difference_draws(a, b, draws=500, seed=3), 0.5)
