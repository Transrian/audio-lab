"""Fine-tune one checkpoint on our own GPU, with NeMo's own training path.

This is the module that needs the heavy stack, and it is the only one: the
config, the settings it imposes on NeMo, the clips a run trains on and the run
directory are written so they can be read and tested with the plain install
(see `requirements/nemo.txt`).

It loads the base checkpoint, hands NeMo our manifest and our two learning rates,
trains for the number of updates the config asks for while rewriting one resume
slot, and writes the fine-tuned checkpoint into the run directory. A run asked to
stop (Ctrl-C, SIGTERM) stops after the update in progress and leaves only the
slot, which the same command resumes from (`ResumeSlot`, D-021). Only the
training settings (see `settings.py`) are ours; the model keeps its preprocessor,
its augmentation and its loss as the checkpoint carries them. Every weight is
trained (batch normalization keeps the base model's statistics, see
`FrozenBatchNormStatistics`) and no adapter is added, so the result is one
ordinary checkpoint, served and evaluated like any other.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.fabric.utilities.seed import pl_worker_init_function
from lightning.pytorch.callbacks import Callback
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, IterableDataset, RandomSampler

from audiolab.core.checkpoint import Checkpoint
from audiolab.core.progress import progress, say
from audiolab.core.stop import install_signal_handlers, raise_if_stopped, stop_requested
from audiolab.train.config import TrainConfig
from audiolab.train.loss import UpdateLoss, as_float, batch_loss, last_logged_loss, write_loss_line
from audiolab.train.order import ClipOrder
from audiolab.train.run import MODEL_NAME, RESUME_NAME, TrainRunError
from audiolab.train.settings import (
    check_rates,
    dataset_config,
    optim_config,
    optim_param_groups,
)


@dataclass(frozen=True)
class TrainResult:
    """What the run measured, for ``train.json``."""

    final_loss: float
    updates: int
    wall_seconds: float
    peak_vram_gib: float
    device: str
    stopped: bool = False  # asked to stop: the slot holds the run, there is no model.nemo


class LossLog(Callback):
    """Append the loss to the run's log every so many updates.

    A JSON line per interval rather than a terminal plot: the log survives the
    process, can be read while the run goes, and is what ``train.json`` reports
    the final loss from. Which number a line holds is decided in `loss.py`, which
    needs no PyTorch - here it is only Lightning's spelling of that rule.
    """

    def __init__(self, path: Path, every: int, print_line=say) -> None:
        self.path = path
        self.print_line = print_line
        self.updates = UpdateLoss(every)
        self.started = time.monotonic()
        self.bar = None

    def on_train_start(self, trainer, pl_module) -> None:
        self.updates = UpdateLoss(self.updates.every, start=trainer.global_step)
        self.bar = progress(
            desc="training", unit="update", total=trainer.max_steps, initial=trainer.global_step
        )

    def on_train_end(self, trainer, pl_module) -> None:
        self.bar.close()

    def on_exception(self, trainer, pl_module, exception) -> None:
        # A run that fails never reaches on_train_end; its bar is closed here.
        if self.bar is not None:
            self.bar.close()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.bar.update(trainer.global_step - self.bar.n)
        # Lightning hands this hook the loss already divided by the accumulation
        # (it is the value it backpropagates); multiplying it back gives the
        # batch's own loss.
        loss = as_float(batch_loss(outputs))
        if loss is not None:
            loss *= trainer.accumulate_grad_batches
        logged = self.updates.batch(int(trainer.global_step), loss)
        if logged is None:
            return
        update, value = logged
        write_loss_line(self.path, update, value, time.monotonic() - self.started)
        self.print_line(f"update {update}: loss {value:.4f}")


class FrozenBatchNormStatistics(Callback):
    """Keep the batch normalization statistics of the base model during training.

    The encoder's convolution blocks normalize with running statistics that
    inference uses as they are. In training mode every batch updates them; with
    a few clips per batch, padded to the longest, they drifted far enough to
    break inference while the training loss kept falling
    (docs/stt/screening.md, "The batch normalization bug"). Holding the layers
    in evaluation mode keeps the statistics; their scale and shift still train
    like every other weight.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        # Lightning puts the whole model back in training mode at the start of
        # an epoch, so this is redone at every batch rather than once.
        for module in pl_module.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


class ResumeSlot(Callback):
    """The one resume slot, and the stop a Ctrl-C or a SIGTERM asks for (D-021).

    The slot is rewritten every ``every`` updates, so a run that dies goes on
    from the last rewrite, and once more when a run is asked to stop, so a run
    that stops goes on from where it stopped. Only one file is ever kept: a run
    never fills a disk with checkpoints. It is written to a temporary name and
    renamed, so a run killed while writing it keeps the slot before.

    The slot is written between two updates only: at the start of the batch
    after an update, or when training ends. Lightning counts a batch as done
    after ``on_train_batch_end``, so a slot written in that hook (as Lightning's
    ModelCheckpoint does) resumes one batch behind, and with gradient
    accumulation the first update after the resume would sum fewer batches than
    the others.

    A stop is honoured once the update in progress is done. Lightning installs
    its own SIGTERM handler while it trains, which stops after the batch, in the
    middle of an accumulation, and saves nothing; training takes the handlers
    back when it starts. The slot also carries the number of batches trained,
    which tells the :class:`ClipOrder` where the resumed run reads from, and the
    random state at that point (dither, SpecAugment's masks, dropout, the TDT
    loss's choice of loss), put back before the first batch of the resume: the
    resumed run trains what the run that went on would have, to the bit when
    the run is deterministic (see :func:`redraw_cudnn_dropout` for the one
    random state the generators do not hold).
    """

    def __init__(self, path: Path, every: int, order: ClipOrder) -> None:
        self.path = path
        self.every = every
        self.order = order
        self.batches_trained = 0
        self.saved_at = -1  # the update the slot on disk was written after
        self.stopped = False
        self.restored = False
        self.random_state = None  # restored, waiting for the first batch

    def state_dict(self) -> dict:
        return {"batches_trained": self.batches_trained, "random_state": random_state()}

    def load_state_dict(self, state_dict: dict) -> None:
        # Lightning restores this before it asks for the first batch.
        self.batches_trained = self.order.start = state_dict["batches_trained"]
        self.random_state = state_dict.get("random_state")
        self.restored = True

    def on_train_start(self, trainer, pl_module) -> None:
        if trainer.global_step > 0 and not self.restored:
            # A slot that Lightning's ModelCheckpoint wrote, before this callback:
            # it does not say which clips were trained, so no resume is exact.
            raise TrainRunError(
                f"{self.path} does not say how many batches it trained, so it cannot be "
                "resumed in the clip order; resume it with the audiolab that wrote it, "
                "or start a new run with --fresh"
            )
        install_signal_handlers()
        self.saved_at = trainer.global_step

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if self.random_state is not None:
            # Only now: starting the data loader draws from the random state, as
            # it did once at the start of the run that went on.
            set_random_state(self.random_state)
            self.random_state = None
        redraw_cudnn_dropout()
        if trainer.global_step != self.saved_at and trainer.global_step % self.every == 0:
            self.save(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.batches_trained += 1
        update_done = self.batches_trained % trainer.accumulate_grad_batches == 0
        last_update = trainer.global_step >= trainer.max_steps
        if update_done and not last_update and stop_requested():
            trainer.should_stop = True
            self.stopped = True

    def on_train_end(self, trainer, pl_module) -> None:
        if self.stopped:
            self.save(trainer)

    def save(self, trainer) -> None:
        written = self.path.with_name(self.path.name + ".partial")
        trainer.save_checkpoint(str(written))
        os.replace(written, self.path)
        self.saved_at = trainer.global_step


def random_state() -> dict:
    """Every random generator training draws from, as it stands."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def redraw_cudnn_dropout() -> None:
    """Make cuDNN draw the dropout of an LSTM from the CUDA generator again.

    On a GPU, an LSTM of several layers (the prediction network of a transducer)
    runs in cuDNN, which draws the dropout between its layers from a random state
    of its own, outside the CUDA generator, so the slot does not hold it. PyTorch
    draws that state again from the generator at the first LSTM after the
    generator is set. Setting the generator to its own state before every batch
    makes the generator the only random state, in the run that goes on as in the
    resumed one.
    """
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch.cuda.get_rng_state_all())


def clip_order(loader: DataLoader, seed: int) -> ClipOrder:
    """NeMo's training loader, read in the order of a :class:`ClipOrder`.

    Every piece of it is kept — its dataset, batch size, collation, workers —
    except the shuffle. Lightning seeds the workers of the loaders it sees, and
    it does not see these, so they are seeded the way it would seed them. Only
    a shuffled loader over a dataset of numbered clips can be read this way;
    any other is refused rather than read in another order than it would be.
    """
    if (
        not isinstance(loader.sampler, RandomSampler)
        or isinstance(loader.dataset, IterableDataset)
        or loader.batch_size is None
    ):
        raise TrainRunError(
            f"NeMo built a training loader that is not shuffled batches of numbered clips "
            f"({type(loader.dataset).__name__}, {type(loader.sampler).__name__}, "
            f"batch size {loader.batch_size}); the train_ds of settings.dataset_config "
            "must keep use_lhotse and is_tarred off and shuffle on"
        )

    def read(indices: list[int]) -> DataLoader:
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=indices,
            collate_fn=loader.collate_fn,
            drop_last=loader.drop_last,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            worker_init_fn=partial(pl_worker_init_function, rank=0),
        )

    return ClipOrder(len(loader.dataset), loader.batch_size, seed, read, drop_last=loader.drop_last)


def fine_tune(
    config: TrainConfig,
    base: Checkpoint,
    manifest_path: Path,
    directory: Path,
    log_path: Path,
    print_line=say,
) -> TrainResult:
    """Train, write the checkpoint into the run directory, and report what happened."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "training needs a GPU: this process sees none (check 'audiolab gpus', and "
            "CUDA_VISIBLE_DEVICES if it was set)"
        )
    # The same seed every time: the same clips in the same order with the same
    # augmentations, so two runs of one config are comparable.
    pl.seed_everything(config.seed, workers=True)

    print_line(f"loading the base model from {base}")
    if base.path:
        model = ASRModel.restore_from(restore_path=str(base.path), map_location="cpu")
    else:
        model = ASRModel.from_pretrained(model_name=base.hf_id, map_location="cpu")

    # Everything that decides what this run trains is set here, in one place.
    with open_dict(model.cfg):
        model.cfg.train_ds = OmegaConf.create(dataset_config(config, manifest_path))
        model.cfg.optim = OmegaConf.create(optim_config(config))
        model.cfg.optim_param_groups = OmegaConf.create(optim_param_groups(config))
        # No validation loop: a short run does not need one, and a held-out WER
        # of one dataset would not decide anything. What decides is measured by
        # 'audiolab eval' on frozen panels, and monitoring inside training is
        # allowed when it is useful (decision 0008).
        model.cfg.validation_ds = {}
    model.setup_training_data(model.cfg.train_ds)
    order = clip_order(model._train_dl, config.seed)
    model._train_dl = order  # what NeMo hands Lightning as the training loader

    slot = directory / RESUME_NAME
    resume = ResumeSlot(slot, config.resume_every, order)
    trainer = pl.Trainer(
        devices=1,
        max_steps=config.updates,
        accumulate_grad_batches=config.grad_accumulation,
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=config.log_every,
        logger=False,
        enable_progress_bar=False,  # ours is LossLog's, which also goes to a terminal only
        enable_checkpointing=False,  # the resume slot is the only checkpoint
        callbacks=[
            resume,
            FrozenBatchNormStatistics(),
            LossLog(log_path, config.log_every, print_line),
        ],
    )
    # Build the optimizer once now, so a wrong learning rate costs a second
    # instead of an hour of GPU time. Lightning builds it again at fit time, from
    # the same configuration.
    model.setup_optimization(model.cfg.optim)
    check_rates(model, config)

    raise_if_stopped("training stopped before it started")
    print_line(
        f"training {config.updates} updates of batch {config.batch_size}"
        f" (x{config.grad_accumulation} accumulation) on {torch.cuda.get_device_name(0)}"
    )
    started = time.monotonic()
    trainer.fit(model, ckpt_path=str(slot) if slot.is_file() else None)
    wall_seconds = time.monotonic() - started

    if not resume.stopped:
        model.save_to(str(directory / MODEL_NAME))
    return TrainResult(
        final_loss=last_logged_loss(log_path),
        updates=int(trainer.global_step),
        wall_seconds=wall_seconds,
        peak_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
        device=torch.cuda.get_device_name(0),
        stopped=resume.stopped,
    )
