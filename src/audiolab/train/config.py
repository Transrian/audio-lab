"""Fine-tuning one model, as written in ``configs/train/<name>.yaml``.

One file describes one run and nothing else (decision 0007). The file's name is
the run's name: it is the directory under ``$AUDIOLAB_RUNS/train/`` and the
model name the run's own eval config uses, so the two can never drift apart.

Every value here is a choice a person made. What the run *did* is written next
to the weights it produced, in ``train.json``
(see [training.md](../../../docs/stt/training.md)).
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from audiolab.core.checkpoint import Checkpoint, CheckpointError

# What a training config may say. Anything else is a mistake, and a mistake here
# costs hours of GPU time, so unknown keys are refused rather than ignored.
TRAIN_FIELDS = (
    "base_model",
    "base_model_env",
    "dataset",
    "clips",
    "updates",
    "batch_size",
    "grad_accumulation",
    "lr_encoder",
    "lr_heads",
    "weight_decay",
    "schedule",
    "warmup_updates",
    "precision",
    "grad_clip",
    "max_clip_seconds",
    "workers",
    "gpu",
    "seed",
    "resume_every",
    "log_every",
)
# The keys with no sensible guess: a run that picked its own size, batch or rate
# would be another run than the file says, so these must be written down.
REQUIRED_FIELDS = ("dataset", "clips", "updates", "batch_size", "lr_encoder", "lr_heads")
SCHEDULES = ("constant", "cosine")
PRECISIONS = ("bf16-mixed", "16-mixed", "32")
# The config's name becomes a directory and a model id, so it is kept boring.
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class TrainConfigError(ValueError):
    """A training config that cannot be used as written."""


@dataclass(frozen=True)
class TrainConfig:
    """One fine-tuning run: what to start from, what to train on, and how."""

    name: str
    # The weights to start from: a local ``.nemo``, or a Hugging Face id. A
    # config committed here names a public model; a machine that holds a local
    # copy sets ``base_model_env`` and trains from the copy.
    base_model: str
    dataset: str
    # How many clips the run trains on, drawn once from the dataset.
    clips: int
    # Optimizer updates, not epochs: this is what stops the run.
    updates: int
    batch_size: int
    # Two learning rates: the encoder (with its preprocessor) and the heads
    # (decoder and joint). Fine-tuning usually wants the heads to move faster.
    lr_encoder: float
    lr_heads: float
    base_model_env: str = ""
    # Gradient accumulation lets a larger effective batch fit in memory: the
    # number of updates does not change, only how many clips each one sees.
    grad_accumulation: int = 1
    weight_decay: float = 1.0e-3
    schedule: str = "constant"
    warmup_updates: int = 0
    precision: str = "bf16-mixed"
    # Gradient norm clipping; 0 disables it.
    grad_clip: float = 1.0
    # Clips longer than this are left out, so a batch has a predictable cost.
    max_clip_seconds: float = 20.0
    workers: int = 4
    gpu: str = "0"
    seed: int = 20261107
    # How often the single resume slot is rewritten, in updates.
    resume_every: int = 500
    # How often a loss line is appended to the run's log.
    log_every: int = 50
    source: str = field(default="", compare=False)

    @classmethod
    def from_file(cls, path: str | Path) -> TrainConfig:
        """Read a training config; the file's stem names the run."""
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as failure:
            raise TrainConfigError(f"{path}: no such training config") from failure
        if not isinstance(raw, dict):
            raise TrainConfigError(f"{path}: a training config is a mapping of keys to values")
        name = path.stem
        if not NAME_PATTERN.match(name):
            raise TrainConfigError(
                f"{path}: {name!r} cannot name a run; use lowercase letters, digits, "
                "'.', '-' or '_'"
            )
        return cls.from_mapping(raw, source=str(path), name=name)

    @classmethod
    def from_mapping(
        cls, raw: dict, *, source: str = "the training config", name: str
    ) -> TrainConfig:
        unknown = sorted(set(raw) - set(TRAIN_FIELDS))
        if unknown:
            raise TrainConfigError(f"{source}: unknown training keys: {', '.join(unknown)}")
        missing = [
            key
            for key in ("base_model", *REQUIRED_FIELDS)
            if raw.get(key) is None or (isinstance(raw[key], str) and not raw[key].strip())
        ]
        if missing:
            raise TrainConfigError(f"{source}: missing required keys: {', '.join(missing)}")
        config = cls(
            name=name,
            base_model=str(raw["base_model"]).strip(),
            base_model_env=str(raw.get("base_model_env") or "").strip(),
            dataset=str(raw["dataset"]).strip(),
            clips=int(raw["clips"]),
            updates=int(raw["updates"]),
            batch_size=int(raw["batch_size"]),
            grad_accumulation=int(raw.get("grad_accumulation", cls.grad_accumulation)),
            lr_encoder=float(raw["lr_encoder"]),
            lr_heads=float(raw["lr_heads"]),
            weight_decay=float(raw.get("weight_decay", cls.weight_decay)),
            schedule=str(raw.get("schedule", cls.schedule)),
            warmup_updates=int(raw.get("warmup_updates", cls.warmup_updates)),
            precision=str(raw.get("precision", cls.precision)),
            grad_clip=float(raw.get("grad_clip", cls.grad_clip)),
            max_clip_seconds=float(raw.get("max_clip_seconds", cls.max_clip_seconds)),
            workers=int(raw.get("workers", cls.workers)),
            gpu=str(raw.get("gpu", cls.gpu)),
            seed=int(raw.get("seed", cls.seed)),
            resume_every=int(raw.get("resume_every", cls.resume_every)),
            log_every=int(raw.get("log_every", cls.log_every)),
            source=source,
        )
        config._check()
        return config

    def _check(self) -> None:
        """Refuse a run that could only fail later, after loading a model."""
        where = self.source or self.name
        counts = {
            "clips": self.clips,
            "updates": self.updates,
            "batch_size": self.batch_size,
            "grad_accumulation": self.grad_accumulation,
            "resume_every": self.resume_every,
            "log_every": self.log_every,
        }
        for key, value in counts.items():
            if value < 1:
                raise TrainConfigError(f"{where}: {key} must be at least 1, not {value}")
        if self.workers < 0:
            raise TrainConfigError(f"{where}: workers cannot be negative")
        if self.clips < self.batch_size:
            raise TrainConfigError(
                f"{where}: clips ({self.clips}) is smaller than batch_size ({self.batch_size}), "
                "so no batch could ever be built"
            )
        if self.lr_encoder <= 0 or self.lr_heads <= 0:
            raise TrainConfigError(f"{where}: learning rates must be above 0")
        if self.lr_encoder > 1 or self.lr_heads > 1:
            raise TrainConfigError(f"{where}: learning rates above 1 are a typo, not a choice")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise TrainConfigError(f"{where}: weight_decay and grad_clip cannot be negative")
        if self.schedule not in SCHEDULES:
            raise TrainConfigError(
                f"{where}: unknown schedule {self.schedule!r}; known: {', '.join(SCHEDULES)}"
            )
        if self.warmup_updates < 0 or self.warmup_updates >= self.updates:
            raise TrainConfigError(
                f"{where}: warmup_updates must be at least 0 and below updates ({self.updates})"
            )
        if self.precision not in PRECISIONS:
            raise TrainConfigError(
                f"{where}: unknown precision {self.precision!r}; known: {', '.join(PRECISIONS)}"
            )
        if self.max_clip_seconds < 1.0:
            raise TrainConfigError(f"{where}: max_clip_seconds must be at least 1")
        if not NAME_PATTERN.match(self.dataset):
            raise TrainConfigError(f"{where}: {self.dataset!r} is not a prepared dataset name")

    def weights(self) -> Checkpoint:
        """The base model to start from, after the local override this file cannot hold."""
        reference = (os.environ.get(self.base_model_env, "") if self.base_model_env else "").strip()
        try:
            return Checkpoint.from_reference(reference or self.base_model)
        except CheckpointError as failure:
            raise TrainConfigError(f"{self.source or self.name}: {failure}") from failure

    def to_dict(self) -> dict:
        """This run's choices, for the record it leaves behind."""
        recorded = asdict(self)
        recorded.pop("source", None)
        return recorded
