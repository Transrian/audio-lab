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


def test_difference_ratio_subtracts_b_from_a():
    a = {"s1": (1.0, 10.0)}
    b = {"s1": (2.0, 10.0)}
    assert stats.difference_ratio(a, b) == 0.1
