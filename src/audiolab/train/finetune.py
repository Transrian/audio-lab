"""Fine-tune one checkpoint on our own GPU, with NeMo's own training path.

This is the module that needs the heavy stack, and it is the only one: the
config, the clips a run trains on and the run directory are all written so they
can be read and tested with the plain install (see `requirements/nemo.txt`).

What a run does, in order: load the base checkpoint, hand NeMo our manifest and
our two learning rates, train for a fixed number of updates while rewriting one
resume slot, then write the fine-tuned checkpoint into the run directory. Every
weight is trained; nothing is frozen and no adapters are added, so the result is
one ordinary checkpoint, served and evaluated like any other.

The learning rates are NeMo's own mechanism: `optim_param_groups` names the
parts of the model whose rate differs and NeMo builds the optimizer from that
and from `optim_config`. The dataset settings are ours rather than the base
model's, because a fine-tune reads a different manifest at a different batch
size; everything else about the model (its preprocessor, its augmentation, its
loss) is left exactly as the checkpoint carries it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf, open_dict

from audiolab.core.audio import SAMPLE_RATE
from audiolab.core.checkpoint import Checkpoint
from audiolab.train.config import TrainConfig
from audiolab.train.run import MODEL_NAME, RESUME_NAME, log_line

# The encoder and the preprocessor it reads features from move together; the
# decoder and the joint are the "heads" a fine-tune usually moves faster.
ENCODER_PARTS = ("preprocessor", "encoder")
HEAD_PARTS = ("decoder", "joint")
# AdamW settings for this kind of model. The second moment is the one that pays
# for long utterances.
BETAS = (0.9, 0.98)


@dataclass(frozen=True)
class TrainResult:
    """What the run measured, for ``train.json``."""

    final_loss: float
    updates: int
    wall_seconds: float
    peak_vram_gib: float
    device: str


def optim_param_groups(config: TrainConfig) -> dict[str, dict[str, float]]:
    """The learning rate of each part of the model, in NeMo's own vocabulary."""
    return {
        **{part: {"lr": config.lr_encoder} for part in ENCODER_PARTS},
        **{part: {"lr": config.lr_heads} for part in HEAD_PARTS},
    }


def optim_config(config: TrainConfig) -> dict:
    """AdamW over those parts, with the schedule this run asked for.

    ``lr`` is the fallback for parameters outside the named parts. The run checks
    that there are none, so the two rates above are the two rates used.
    """
    if config.schedule == "cosine":
        # Warm up, then decay to nothing by the last update.
        sched = {"name": "CosineAnnealing", "warmup_steps": config.warmup_updates, "min_lr": 0.0}
    else:
        # Warm up, then hold the rate the run asked for.
        sched = {"name": "WarmupPolicy", "warmup_steps": config.warmup_updates}
    # NeMo fills this in from the trainer as well; giving it here means the
    # schedule can also be built before the trainer exists, which is when it is
    # checked.
    sched["max_steps"] = config.updates
    return {
        "name": "adamw",
        "lr": config.lr_heads,
        "betas": list(BETAS),
        "weight_decay": config.weight_decay,
        "sched": sched,
    }


def dataset_config(config: TrainConfig, manifest_path: Path) -> dict:
    """The dataset NeMo reads: our manifest, at the rate the prepared audio is in.

    The clips were filtered when they were drawn, so nothing here drops a clip:
    what a run sees is exactly what the config says.
    """
    return {
        "manifest_filepath": str(manifest_path),
        "sample_rate": SAMPLE_RATE,
        "batch_size": config.batch_size,
        # NeMo's plain manifest reader, not its lhotse one: our manifest is JSON
        # lines with absolute audio paths.
        "use_lhotse": False,
        "is_tarred": False,
        "shuffle": True,
        "num_workers": config.workers,
        "pin_memory": True,
        "max_duration": config.max_clip_seconds,
        "min_duration": 0.0,
        "trim_silence": False,
        "use_start_end_token": True,
    }


def check_rates(model, config: TrainConfig) -> None:
    """Check on the loaded model that the two rates really went where asked.

    A part named differently in this model would move at the fallback rate
    without complaining, so the parameters themselves are looked up before any
    GPU time is spent.
    """
    groups = model._optimizer.param_groups
    if any(not group["params"] for group in groups):
        raise ValueError("the optimizer holds an empty parameter group")
    wanted = {part: config.lr_encoder for part in ENCODER_PARTS}
    wanted.update({part: config.lr_heads for part in HEAD_PARTS})
    for part, rate in wanted.items():
        module = getattr(model, part, None)
        if module is None:
            raise ValueError(
                f"the base model has no part named {part!r}, so its rate cannot be set"
            )
        owned = {id(parameter) for parameter in module.parameters()}
        holders = [group for group in groups if owned & {id(p) for p in group["params"]}]
        if len(holders) != 1:
            raise ValueError(f"the parameters of {part!r} are spread over {len(holders)} groups")
        if holders[0]["lr"] != rate:
            raise ValueError(f"the parameters of {part!r} train at {holders[0]['lr']}, not {rate}")
    rates = {group["lr"] for group in groups}
    if rates - {config.lr_encoder, config.lr_heads}:
        raise ValueError(f"the optimizer holds learning rates the config did not ask for: {rates}")


class LossLog(Callback):
    """Append the loss to the run's log every so many updates.

    A JSON line per interval rather than a terminal plot: the log survives the
    process, can be read while the run goes, and is what ``train.json`` reports
    the final loss from.
    """

    def __init__(self, path: Path, every: int, print_line=print) -> None:
        self.path = path
        self.every = every
        self.print_line = print_line
        self.logged = 0
        self.started = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        update = int(trainer.global_step)
        # With gradient accumulation, several batches belong to one update.
        if update == self.logged or update % self.every:
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return
        value = float(loss.detach().float().cpu())
        self.logged = update
        seconds = time.monotonic() - self.started
        log_line(self.path, {"update": update, "loss": value, "seconds": round(seconds, 3)})
        self.print_line(f"update {update}: loss {value:.4f}")


def last_logged_loss(log_path: Path) -> float:
    """The loss of the last logged update: what the run ended at."""
    last = float("nan")
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = float(json.loads(line)["loss"])
    return last


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
        # No validation loop: a few hundred held-out clips of one dataset say
        # little about a model (decision 0008). What decides anything is measured
        # by 'audiolab eval' on frozen panels.
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
        callbacks=[resume, LossLog(log_path, config.log_every, print_line)],
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
