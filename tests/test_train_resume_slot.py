"""A run stopped and resumed trains what the run that went on would have trained.

This needs PyTorch, Lightning and NeMo, which the default environment does not
install, so it is skipped there; run it on the CPU with the environment of
`scripts/nemo-venv.sh`:
`CUDA_VISIBLE_DEVICES= .venv-nemo/bin/python -m pytest tests/test_train_resume_slot.py`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("lightning.pytorch")
pytest.importorskip("nemo")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from audiolab.core.stop import request_stop  # noqa: E402
from audiolab.train.finetune import ResumeSlot, clip_order  # noqa: E402
from audiolab.train.order import ClipOrder  # noqa: E402
from audiolab.train.run import TrainRunError  # noqa: E402

CLIPS = 10
BATCH = 3
ACCUMULATION = 2
UPDATES = 8


class Tiny(pl.LightningModule):
    """A linear regression that records the clips of every batch it trains on."""

    def __init__(self, order: ClipOrder) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layer = torch.nn.Linear(2, 1)
        self.order = order
        self.seen: list[list[int]] = []

    def train_dataloader(self):
        return self.order

    def training_step(self, batch, batch_idx):
        inputs, targets, clips = batch
        self.seen.append(clips.tolist())
        return ((self.layer(inputs).squeeze(1) - targets) ** 2).mean()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.1)


class StopAtBatch(pl.Callback):
    """Ask for a stop after one batch, as a Ctrl-C in the middle of an update would."""

    def __init__(self, batch: int) -> None:
        self.batch = batch
        self.batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches += 1
        if self.batches == self.batch:
            request_stop()


def train(slot, *, also=(), resume: bool = False):
    generator = torch.Generator().manual_seed(1)
    data = TensorDataset(
        torch.randn(CLIPS, 2, generator=generator),
        torch.randn(CLIPS, generator=generator),
        torch.arange(CLIPS),
    )
    order = ClipOrder(CLIPS, BATCH, 7, lambda indices: DataLoader(data, BATCH, sampler=indices))
    model = Tiny(order)
    callback = ResumeSlot(slot, every=3, order=order)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=UPDATES,
        accumulate_grad_batches=ACCUMULATION,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[callback, *also],
    )
    trainer.fit(model, ckpt_path=str(slot) if resume else None)
    return model, callback, trainer


def test_a_stop_ends_after_the_update_in_progress_and_the_resume_goes_on_exactly(
    tmp_path, run_again
):
    went_on, _, _ = train(tmp_path / "whole.ckpt")

    slot = tmp_path / "resume.ckpt"
    # The fifth batch is the first half of the third update: the stop waits for
    # the second half, then writes the slot.
    stopped, callback, trainer = train(slot, also=[StopAtBatch(5)])
    assert callback.stopped and trainer.global_step == 3
    assert len(stopped.seen) == 6
    assert slot.is_file() and not slot.with_name("resume.ckpt.partial").exists()

    run_again()
    resumed, callback, trainer = train(slot, resume=True)
    assert not callback.stopped and trainer.global_step == UPDATES
    assert stopped.seen + resumed.seen == went_on.seen
    for after_resume, uninterrupted in zip(resumed.parameters(), went_on.parameters(), strict=True):
        assert torch.equal(after_resume, uninterrupted)


def test_the_slot_is_rewritten_between_two_updates_every_so_many(tmp_path):
    slot = tmp_path / "resume.ckpt"
    written = []

    class Watch(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if slot.is_file():
                state = torch.load(slot, weights_only=False)
                written.append((state["global_step"], state["callbacks"]["ResumeSlot"]))

    train(slot, also=[Watch()])
    steps = sorted({step for step, _ in written})
    assert steps == [3, 6]
    assert dict(written)[3] == {"batches_trained": 3 * ACCUMULATION}


class CrashAtBatch(StopAtBatch):
    """Die after one batch, as a run killed in the middle of an update would."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches += 1
        if self.batches == self.batch:
            raise RuntimeError("killed")


def test_a_run_that_died_goes_on_exactly_from_the_last_periodic_slot(tmp_path):
    went_on, _, _ = train(tmp_path / "whole.ckpt")

    slot = tmp_path / "resume.ckpt"
    with pytest.raises(RuntimeError, match="killed"):
        # The 13th batch is the first half of update 7; the slot was rewritten after update 6.
        train(slot, also=[CrashAtBatch(6 * ACCUMULATION + 1)])
    assert torch.load(slot, weights_only=False)["global_step"] == 6

    resumed, _, trainer = train(slot, resume=True)
    assert trainer.global_step == UPDATES
    assert resumed.seen == went_on.seen[6 * ACCUMULATION :], "the half update lost is read again"
    for after_resume, uninterrupted in zip(resumed.parameters(), went_on.parameters(), strict=True):
        assert torch.equal(after_resume, uninterrupted)


def test_a_slot_that_does_not_say_how_many_batches_it_trained_is_refused(tmp_path, run_again):
    slot = tmp_path / "resume.ckpt"
    train(slot, also=[StopAtBatch(1)])
    older = torch.load(slot, weights_only=False)
    del older["callbacks"]["ResumeSlot"]  # what Lightning's ModelCheckpoint wrote
    torch.save(older, slot)

    run_again()
    with pytest.raises(TrainRunError, match="--fresh"):
        train(slot, resume=True)


def test_only_a_shuffled_loader_of_numbered_clips_is_read_in_the_clip_order():
    data = TensorDataset(torch.zeros(CLIPS, 2), torch.zeros(CLIPS), torch.arange(CLIPS))
    order = clip_order(DataLoader(data, BATCH, shuffle=True), seed=7)
    assert order.batches_per_epoch == 4
    with pytest.raises(TrainRunError, match="settings.dataset_config"):
        clip_order(DataLoader(data, BATCH), seed=7)
