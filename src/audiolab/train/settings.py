"""The settings one run imposes on NeMo: its dataset, its optimizer, its rates.

These are decisions about *what* a run does, written as NeMo's own configuration,
and they are separated from the training loop so they can be read and tested
without PyTorch: `dataset_config` says what the run reads, `optim_config` and
`optim_param_groups` say how it learns, and `check_rates` checks on the loaded
model that the two rates landed where the config wanted them.

Only the training loop itself needs the heavy stack.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.core.audio import SAMPLE_RATE
from audiolab.train.config import TrainConfig

# The encoder and the preprocessor it reads features from move together; the
# decoder and the joint are the "heads" a fine-tune usually moves faster. The
# preprocessor of this family has no weights of its own, and a part with no
# weights is skipped rather than refused: naming it keeps the intent readable,
# and covers families whose feature extractor does train.
ENCODER_PARTS = ("preprocessor", "encoder")
HEAD_PARTS = ("decoder", "joint")
# AdamW's betas. The second moment forgets faster than the usual 0.999, which
# suits speech batches whose gradient scale moves with the clip lengths.
BETAS = (0.9, 0.98)


def optim_param_groups(config: TrainConfig) -> dict[str, dict[str, float]]:
    """The learning rate of each part of the model, in NeMo's own vocabulary."""
    return {
        **{part: {"lr": config.lr_encoder} for part in ENCODER_PARTS},
        **{part: {"lr": config.lr_heads} for part in HEAD_PARTS},
    }


def optim_config(config: TrainConfig) -> dict:
    """AdamW over those parts, with the schedule this run asked for.

    ``lr`` is what NeMo gives a parameter that no named part claims. ``check_rates``
    refuses such a model, so the fallback never applies and the two rates of the
    config are the two rates in use.
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
        # NeMo wraps a transcript in <bos>/<eos> when this is on. The v3 tokenizer
        # carries bos = eos = -1, so nothing is added either way today; NeMo's own
        # fine-tuning recipes leave it off, and another checkpoint's tokenizer
        # would otherwise put two tokens in every transcript that are not text.
        "use_start_end_token": False,
    }


def _rate(group: dict) -> float:
    """The rate a group was created with.

    NeMo's warmup policy steps the scheduler once while it is built, so by the
    time a run can look, ``lr`` is no longer the rate the config asked for.
    ``initial_lr`` is what the schedule started from, and what a run means.
    """
    return group.get("initial_lr", group["lr"])


def check_rates(model, config: TrainConfig) -> None:
    """Check on the loaded model that this run trains what it says it trains.

    Four things can go wrong quietly, and each makes the result mean something
    else: a part named differently in this model moves at the fallback rate, a
    parameter no named part owns does too, a part whose weights are spread over
    two groups follows neither rate, and a frozen weight turns a full fine-tune
    into a partial one. All four are looked for here, before any GPU time is
    spent.
    """
    groups = model._optimizer.param_groups
    wanted = {part: config.lr_encoder for part in ENCODER_PARTS}
    wanted.update({part: config.lr_heads for part in HEAD_PARTS})
    owner: dict[int, str] = {}
    for part, rate in wanted.items():
        module = getattr(model, part, None)
        if module is None:
            raise ValueError(
                f"the base model has no part named {part!r}, so its rate cannot be set"
            )
        parameters = list(module.parameters())
        if not parameters:
            # The feature extractor of this family computes its filters rather
            # than learning them; a part with no weights has no rate.
            continue
        owner.update({id(parameter): part for parameter in parameters})
        owned = {id(parameter) for parameter in parameters}
        holders = [group for group in groups if owned & {id(p) for p in group["params"]}]
        if not holders:
            raise ValueError(
                f"the optimizer holds no parameter of {part!r}: it is frozen, or the part is "
                "not the one the model trains"
            )
        if len(holders) != 1:
            raise ValueError(f"the parameters of {part!r} are spread over {len(holders)} groups")
        held = _rate(holders[0])
        if held != rate:
            raise ValueError(f"the parameters of {part!r} train at {held}, not {rate}")

    rates = {_rate(group) for group in groups}
    if rates - {config.lr_encoder, config.lr_heads}:
        raise ValueError(f"the optimizer holds learning rates the config did not ask for: {rates}")

    # Everything the optimizer holds, and everything the model holds, is a
    # parameter of one of those parts: nothing trains unaccounted for, at a rate
    # no check above looked at.
    strays = [id(p) for group in groups for p in group["params"] if id(p) not in owner]
    if strays:
        raise ValueError(
            f"the optimizer holds {len(strays)} parameters of no named part, so they would train "
            f"at the fallback rate; the parts are {', '.join(sorted(wanted))}"
        )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"{name} is frozen, so this run would not train the whole model")
        if id(parameter) not in owner:
            raise ValueError(
                f"{name} belongs to no part this run sets a rate for; the parts are "
                f"{', '.join(sorted(wanted))}"
            )
