"""Running one function over many items, a few at a time (D-022).

The items are requests to a served model: what an engine takes at once varies,
so how many run together is the caller's to say, one by default.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor


def never() -> bool:
    """The stop condition of a run nothing asks to stop."""
    return False


def map_requests[T, R](
    ask: Callable[[T], R],
    items: Iterable[T],
    concurrency: int = 1,
    *,
    should_stop: Callable[[], bool] = never,
    failing: threading.Event | None = None,
) -> list[R]:
    """``ask`` on every item, ``concurrency`` at a time; the results in the items' order.

    This is how every caller sends requests N at a time (D-022). It runs the
    requests of one model only: a caller that asks several models through a
    hot-swapping server sends every request of one model, through one call,
    before the next model's (D-020), so concurrency never mixes two models.

    The items are taken in order, one at a time, as a worker becomes free, so
    an iterator that is still producing them (a talk still downloading) is read
    as the requests go. No item starts once ``should_stop()`` is true: the items
    in flight finish, and the results returned are those of the items started,
    the first ones. An ``ask`` that fails, or an interrupted caller, sets
    ``failing``: no item starts either, an ``ask`` watching it can end early,
    and the failure is raised once the items in flight are done.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, not {concurrency}")
    failing = threading.Event() if failing is None else failing
    numbered = enumerate(items)
    taking = threading.Lock()
    results: dict[int, R] = {}

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
        except BaseException:
            failing.set()
            raise

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        workers = [pool.submit(work) for _ in range(concurrency)]
        try:
            for worker in workers:
                worker.result()
        except BaseException:
            failing.set()
            raise
    return [results[index] for index in range(len(results))]
