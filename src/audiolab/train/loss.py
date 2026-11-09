"""The loss a run reports: the number in its log, and how it is read back.

A training step returns the loss of the batch it just saw, and with gradient
accumulation Lightning divides it by the number of accumulated batches before
the backward pass, and passes that divided value on to the hooks. The training
loop multiplies it back, and the log holds the mean batch loss since the
previous line: the same scale whatever the accumulation is.

Nothing here needs PyTorch: the training loop hands over the value it received,
and these functions turn a sequence of batch losses into the loss of the run.
"""

from __future__ import annotations

import json
from pathlib import Path

from audiolab.train.run import log_line


def batch_loss(outputs):
    """The loss inside what a training step returned, if it returned one."""
    if isinstance(outputs, dict):
        return outputs.get("loss")
    return outputs


def as_float(loss) -> float | None:
    """The value of a loss, or ``None`` when there is nothing to read.

    A step returns a tensor, and detaching it before it crosses to the CPU keeps
    the graph and the GPU out of the log file.
    """
    if loss is None:
        return None
    if isinstance(loss, (int, float)):
        return float(loss)
    for hook in ("detach", "float", "cpu"):
        if not hasattr(loss, hook):
            return None
        loss = getattr(loss, hook)()
    return float(loss)


class UpdateLoss:
    """Every so many updates, the mean loss of the batches trained on since the last line.

    NeMo counts an update in ``trainer.global_step``, which moves once per
    optimizer update: with gradient accumulation only the last batch of the group
    carries the new number, and the earlier ones repeat the previous update. The
    mean over the batches since the last line is the loss the run is at, on the
    same scale whatever the accumulation is.

    A resumed run starts at the update its slot holds, which the stopped run
    already logged: ``start`` keeps that line from being written twice, with
    the loss of only the batches the resume has trained so far.
    """

    def __init__(self, every: int, start: int = 0) -> None:
        self.every = every
        self.logged = start
        self._batch_losses: list[float] = []

    def batch(self, update: int, loss) -> tuple[int, float] | None:
        """Take one batch loss; return the update and its loss when a line is due.

        ``None`` means "not now": the update has not moved, or it is not a
        multiple of ``every`` yet.
        """
        value = as_float(loss)
        if value is not None:
            self._batch_losses.append(value)
        if update == self.logged or update % self.every or not self._batch_losses:
            return None
        mean = sum(self._batch_losses) / len(self._batch_losses)
        self._batch_losses.clear()
        self.logged = update
        return update, mean


def last_logged_loss(log_path: Path) -> float:
    """The loss of the last logged update: what the run ended at."""
    last = float("nan")
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = float(json.loads(line)["loss"])
    return last


def write_loss_line(path: Path, update: int, loss: float, seconds: float) -> None:
    """Append one line of the run's log: the update, its loss, and the elapsed time."""
    log_line(path, {"update": update, "loss": loss, "seconds": round(seconds, 3)})
