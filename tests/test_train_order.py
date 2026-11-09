"""The order a run reads its clips in: seeded, and resumable at any batch."""

from __future__ import annotations

from itertools import islice

import pytest

from audiolab.train.order import ClipOrder


def batches_of(indices: list[int], batch_size: int = 3, drop_last: bool = False) -> list:
    """What a DataLoader over these indices yields, as plain lists."""
    batches = [indices[at : at + batch_size] for at in range(0, len(indices), batch_size)]
    return [one for one in batches if len(one) == batch_size or not drop_last]


def order(start: int = 0, *, seed: int = 7, drop_last: bool = False) -> ClipOrder:
    read = lambda indices: batches_of(indices, drop_last=drop_last)  # noqa: E731
    clips = ClipOrder(10, 3, seed, read, drop_last=drop_last)
    clips.start = start
    return clips


def test_every_epoch_reads_every_clip_once_in_an_order_of_its_own():
    clips = order()
    first, second = clips.epoch(0), clips.epoch(1)
    assert sorted(first) == sorted(second) == list(range(10))
    assert first != second
    assert order().epoch(0) == first, "the order depends on the seed and the epoch alone"
    assert order(seed=8).epoch(0) != first


def test_batches_go_on_from_one_epoch_to_the_next_without_end():
    batches = list(islice(order(), 9))
    assert [len(batch) for batch in batches] == [3, 3, 3, 1] * 2 + [3]
    assert sum(batches[:4], []) == order().epoch(0)


@pytest.mark.parametrize("start", [1, 3, 4, 6, 11])
def test_a_resumed_order_reads_the_batches_the_run_that_went_on_would_read(start):
    went_on = list(islice(order(), 20))
    assert list(islice(order(start), 20 - start)) == went_on[start:]


def test_with_drop_last_an_epoch_is_the_full_batches_only():
    clips = order(drop_last=True)
    assert clips.batches_per_epoch == 3
    went_on = list(islice(clips, 12))
    assert all(len(batch) == 3 for batch in went_on)
    assert list(islice(order(4, drop_last=True), 8)) == went_on[4:]


def test_too_few_clips_for_one_full_batch_are_refused():
    with pytest.raises(ValueError, match="do not fill one batch"):
        ClipOrder(2, 3, 7, batches_of, drop_last=True)
