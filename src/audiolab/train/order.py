"""The order a run reads its clips in, so that a resumed run reads what was next.

NeMo hands Lightning a plain PyTorch DataLoader that shuffles with the global
random generator. Resumed from a checkpoint, that loader starts a new shuffle
from the first batch: the clips the stopped run had not reached yet are not the
ones the resumed run reads, so a run stopped and resumed is not the run that
would have gone through. Lightning can resume a loader only if the loader can
say where it is, and this one cannot.

:class:`ClipOrder` replaces it. Every epoch is a permutation of the clips drawn
from the run's seed and the epoch's number alone, so it is the same in every
process; the only state is the number of batches already trained, which the
resume slot carries. A run resumed after ``n`` batches reads, from the epoch
``n`` falls in, the clips after those ``n`` batches, then the next epochs as
the run that did not stop would have. The random state (dither, SpecAugment,
dropout) is not this module's: the resume slot carries it (``finetune.py``).

This module needs no PyTorch: ``read`` is what turns some clip indices into
batches (a DataLoader over those indices, in ``finetune.py``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Iterator


class ClipOrder:
    """An endless sequence of batches, epoch after epoch, that can start at any batch."""

    def __init__(
        self,
        clips: int,
        batch_size: int,
        seed: int,
        read: Callable[[list[int]], Iterable],
        *,
        drop_last: bool = False,
    ) -> None:
        self.clips = clips
        self.batch_size = batch_size
        self.seed = seed
        self.read = read
        self.batches_per_epoch = clips // batch_size if drop_last else math.ceil(clips / batch_size)
        if self.batches_per_epoch < 1:
            raise ValueError(f"{clips} clips do not fill one batch of {batch_size}")
        # Batches already trained when the run started: 0, or what the slot says.
        self.start = 0

    def epoch(self, number: int) -> list[int]:
        """The clips of one epoch, in the order they are read."""
        order = list(range(self.clips))
        random.Random(f"{self.seed}/{number}").shuffle(order)
        return order

    def __iter__(self) -> Iterator:
        number, skipped = divmod(self.start, self.batches_per_epoch)
        while True:
            yield from self.read(self.epoch(number)[skipped * self.batch_size :])
            number, skipped = number + 1, 0
