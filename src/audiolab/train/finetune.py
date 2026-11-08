"""Fine-tune one checkpoint on our own GPU, with NeMo's own training path.

This is the module that needs the heavy stack, and it is the only one: the
config, the settings it imposes on NeMo, the clips a run trains on and the run
directory are written so they can be read and tested with the plain install
(see `requirements/nemo.txt`).

It loads the base checkpoint, hands NeMo our manifest and our two learning rates,
trains for the number of updates the config asks for while rewriting one resume
slot, and writes the fine-tuned checkpoint into the run directory. Only the
training settings (see `settings.py`) are ours; the model keeps its preprocessor,
its augmentation and its loss as the checkpoint carries them. Every weight is
trained (batch normalization keeps the base model's statistics, see
`FrozenBatchNormStatistics`) and no adapter is added, so the result is one
ordinary checkpoint, served and evaluated like any other.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf, open_dict

from audiolab.core.checkpoint import Checkpoint
from audiolab.train.config import TrainConfig
from audiolab.train.loss import UpdateLoss, as_float, batch_loss, last_logged_loss, write_loss_line
from audiolab.train.run import MODEL_NAME, RESUME_NAME
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


class LossLog(Callback):
    """Append the loss to the run's log every so many updates.

    A JSON line per interval rather than a terminal plot: the log survives the
    process, can be read while the run goes, and is what ``train.json`` reports
    the final loss from. Which number a line holds is decided in `loss.py`, which
    needs no PyTorch - here it is only Lightning's spelling of that rule.
    """

    def __init__(self, path: Path, every: int, print_line=print) -> None:
        self.path = path
        self.print_line = print_line
        self.updates = UpdateLoss(every)
        self.started = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
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

    The encoder's convolution blocks normalize with statistics that the model
    learned on large batches, and that inference uses as they are. In training
    mode each batch would replace them, a little at every step, with its own:
    with a few clips per batch, padded to the longest, the padding frames pull
    them away from what real audio looks like. A run measured on 2026-11-08 lost
    its English that way while its training loss kept falling. Holding the
    layers in evaluation mode keeps the statistics; their scale and shift still
    train like every other weight.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        # Lightning puts the whole model back in training mode at the start of
        # an epoch, so this is redone at every batch rather than once.
        for module in pl_module.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def fine_tune(
    config: TrainConfig,
    base: Checkpoint,
    manifest_path: Path,
    directory: Path,
    log_path: Path,
    print_line=print,
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

    # One resume slot, rewritten as the run goes: a run never fills a disk with
    # checkpoints, and a run that dies goes on from the last rewrite instead of
    # from the start.
    resume = ModelCheckpoint(
        dirpath=str(directory),
        filename=RESUME_NAME.removesuffix(".ckpt"),
        every_n_train_steps=config.resume_every,
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
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
        enable_progress_bar=False,
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

    slot = directory / RESUME_NAME
    print_line(
        f"training {config.updates} updates of batch {config.batch_size}"
        f" (x{config.grad_accumulation} accumulation) on {torch.cuda.get_device_name(0)}"
    )
    started = time.monotonic()
    trainer.fit(model, ckpt_path=str(slot) if slot.is_file() else None)
    wall_seconds = time.monotonic() - started

    model.save_to(str(directory / MODEL_NAME))
    return TrainResult(
        final_loss=last_logged_loss(log_path),
        updates=int(trainer.global_step),
        wall_seconds=wall_seconds,
        peak_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
        device=torch.cuda.get_device_name(0),
    )
