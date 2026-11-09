"""Running one function over many items, a few at a time (D-022).

Each item is one call that mostly waits on the network: a request to a served
model, or a page fetched from a website. What the other side takes at once
varies, so how many run together is the caller's to say, one by default.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

from audiolab.core.stop import stop_requested


def map_requests[T, R](
    ask: Callable[[T], R],
    items: Iterable[T],
    concurrency: int = 1,
    *,
    should_stop: Callable[[], bool] = stop_requested,
    failing: threading.Event | None = None,
) -> list[R]:
    """``ask`` on every item, ``concurrency`` at a time; the results in the items' order.

    This is how every caller makes its calls N at a time (D-022). Calls to a
    served model are those of one model only: a caller that asks several
    models through a hot-swapping server sends every request of one model,
    through one call, before the next model's (D-020), so concurrency never
    mixes two models.

    The items are taken in order, one at a time, as a worker becomes free, so
    an iterator that is still producing them (a talk still downloading) is read
    as the requests go. No item starts once ``should_stop()`` is true, by
    default once a stop was asked for (D-021): the items in flight finish, and
    the results returned are those of the items started, the first ones. A
    caller therefore calls :func:`~audiolab.core.stop.raise_if_stopped` right
    after, so that a stopped run is not taken for a finished one. An ``ask``
    that fails, or an interrupted caller, sets ``failing``: no item starts
    either, an ``ask`` watching it can end early, and the first failure is
    raised once the items in flight are done.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, not {concurrency}")
    failing = threading.Event() if failing is None else failing
    numbered = enumerate(items)
    taking = threading.Lock()
    results: dict[int, R] = {}
    failures: list[BaseException] = []

    def work() -> None:
        try:
            while True:
                with taking:
                    stopped = failing.is_set() or should_stop()
                    entry = None if stopped else next(numbered, None)
                if entry is None:
                    return
                index, item = entry
                results[index] = ask(item)
        except BaseException as failure:
            # The first failure is the one raised: an ask that ends early because
            # of it fails after it.
            failures.append(failure)
            failing.set()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        workers = [pool.submit(work) for _ in range(concurrency)]
        try:
            for worker in workers:
                worker.result()
        except BaseException:
            failing.set()
            raise
    if failures:
        raise failures[0]
    return [results[index] for index in range(len(results))]
