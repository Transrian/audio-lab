"""Progress bars for long tasks, on a terminal only (D-021).

A bar goes to standard error and only when standard error is a terminal: a run
under systemd writes its output to a log file, where a bar would be thousands
of redrawn lines. What a log file keeps are the plain lines a task prints anyway
(a loss every so many updates, a count every so many clips); :func:`say` prints
them without breaking a bar that is on screen.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

from tqdm import tqdm


def progress(
    items: Iterable | None = None,
    *,
    desc: str,
    unit: str = "it",
    total: float | None = None,
    initial: float = 0,
) -> tqdm:
    """A bar over ``items``, or over ``total`` steps moved by ``update(n)``.

    ``unit="B"`` counts bytes and shows them as KiB, MiB, GiB.
    """
    in_bytes = unit == "B"
    return tqdm(
        items,
        desc=desc,
        unit=unit,
        total=total,
        initial=initial,
        unit_scale=in_bytes,
        unit_divisor=1024 if in_bytes else 1000,
        file=sys.stderr,
        disable=None,  # on a terminal only
        dynamic_ncols=True,
    )


def say(line: str) -> None:
    """Print one line of a task's record, above any bar on screen."""
    tqdm.write(line, file=sys.stdout)
