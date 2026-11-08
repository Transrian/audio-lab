"""The loss a run reports: one number per logged interval, whatever the batch.

These are the rules behind ``train-log.jsonl`` and the ``final_loss`` of
``train.json``. They live outside the training loop, which needs PyTorch, so
they can be checked here without a GPU.
"""

from __future__ import annotations

import json

from audiolab.train.loss import UpdateLoss, as_float, batch_loss, last_logged_loss


class FakeLoss:
    """A loss tensor, as far as reading it goes."""

    def __init__(self, value: float):
        self.value = value
        self.hooks = []

    def detach(self):
        self.hooks.append("detach")
        return self

    def float(self):
        self.hooks.append("float")
        return self

    def cpu(self):
        self.hooks.append("cpu")
        return self

    def __float__(self) -> float:
        return self.value


def test_a_loss_tensor_is_read_without_leaving_the_gpu():
    loss = FakeLoss(2.5)
    assert as_float(loss) == 2.5
    assert loss.hooks == ["detach", "float", "cpu"]


def test_something_that_is_not_a_loss_is_not_read():
    assert as_float(None) is None
    assert as_float("2.5") is None
    assert as_float(object()) is None
    assert as_float(2.5) == 2.5


def test_a_step_that_returned_no_loss_is_not_read():
    assert batch_loss({"loss": FakeLoss(1.0)}).value == 1.0
    assert batch_loss({}) is None
    assert batch_loss(FakeLoss(3.0)).value == 3.0


def test_a_line_holds_the_mean_of_the_batches_since_the_last_line(tmp_path):
    """One batch's loss is not what the updates since the last line did."""
    updates = UpdateLoss(every=2)
    assert updates.batch(1, 1.0) is None  # an odd update: not due yet
    assert updates.batch(2, 3.0) == (2, 2.0)
    assert updates.batch(4, 5.0) == (4, 5.0)
    # A batch that carries no loss does not count as one.
    assert updates.batch(6, None) is None
    assert updates.batch(6, 4.0) == (6, 4.0)


def test_an_accumulated_update_is_the_mean_of_its_batches():
    """With gradient accumulation only the last batch carries the new step.

    The line must report what that update trained on - the mean of its two
    batches - not the last batch's loss, and not their sum.
    """
    updates = UpdateLoss(every=1)
    assert updates.batch(0, 1.0) is None  # before the update: global_step is still 0
    assert updates.batch(1, 3.0) == (1, 2.0)
    assert updates.batch(1, 2.0) is None  # another batch of the same update, not a line


def test_a_repeated_update_number_is_never_logged_twice():
    updates = UpdateLoss(every=1)
    assert updates.batch(1, 1.0) == (1, 1.0)
    assert updates.batch(1, 1.0) is None
    assert updates.batch(2, 1.0) == (2, 1.0)


def test_the_final_loss_is_the_last_line_of_the_log(tmp_path):
    assert last_logged_loss(tmp_path / "absent.jsonl") != last_logged_loss(
        tmp_path / "absent.jsonl"
    )  # nan: a run that logged nothing has no final loss
    path = tmp_path / "train-log.jsonl"
    lines = [json.dumps({"update": update, "loss": loss}) for update, loss in ((1, 3.0), (2, 2.0))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert last_logged_loss(path) == 2.0
