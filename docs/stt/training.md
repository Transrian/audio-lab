# Fine-tuning a model

This page explains how `audiolab train` fine-tunes one checkpoint on one dataset,
what it writes, how a stopped run goes on, and what it deliberately does not do.
The code lives in `src/audiolab/train/`.

A run is one config file in `configs/train/`, the way every experiment in this
repository is one config file
([decision 0007](../decisions/accepted/0007-single-package-single-command.md)).
The file's name is the run's name: `configs/train/fleurs-fr-v1.yaml` trains into
`$AUDIOLAB_RUNS/train/fleurs-fr-v1/`.

## 1. The environment it needs

NeMo, PyTorch and the CUDA compiler are **not** dependencies of the project.
`uv sync` stays light, CI stays fast, and scoring a model through an API needs
none of it. Training needs all of it, in one environment of its own:

```bash
bash scripts/nemo-venv.sh                  # creates ./.venv-nemo
.venv-nemo/bin/audiolab train --config configs/train/example.yaml
```

`requirements/nemo.txt` holds the exact versions and says why each is pinned.
Two of those pins are worth knowing about before a run:

- **A CUDA compiler.** A TDT model's loss is a CUDA kernel that Numba compiles
  the first time a training step runs. A machine with a system CUDA toolkit
  already has one; `requirements/nemo.txt` also installs `nvcc` and the CUDA
  headers as wheels, so the environment works on a machine without one. The
  versions matter: a toolkit newer than the driver accepts fails with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` at the first step, which is why the
  versions are pinned together with PyTorch's build.
- **`--index-strategy unsafe-best-match`**, which `scripts/nemo-venv.sh` passes
  on the command line: the PyTorch index also carries stale copies of unrelated
  packages, and the flag makes uv look at both indexes.

> **A stopgap, pending an owner decision.** These pins belong in an optional
> extra of `pyproject.toml`, where anyone could install them with one command.
> That is not possible today: the version of `nvidia-curand-cu12` that
> PyTorch's CUDA 12.8 index resolves to looks like a private address to
> `scripts/check_repo.py`, which refuses `uv.lock`. Until the owner decides
> whether that check should skip `uv.lock`, the heavy stack lives in
> `requirements/nemo.txt` and in its own environment, and this page describes it
> that way.

Nothing in `src/audiolab/train/` is imported by the rest of the package: the
tests, the data commands and the evaluation all run without NeMo. Only the
module that runs the training loop imports PyTorch, and the settings a run
imposes on NeMo are kept separate from that loop so they can be read and tested
without it.

## 2. The config file

```yaml
# configs/train/example.yaml
base_model: nvidia/parakeet-tdt-0.6b-v3   # a Hub id, or the path of a .nemo file
base_model_env: AUDIOLAB_PARAKEET_NEMO    # optional: a variable that names a local copy
dataset: fleurs-fr                        # a dataset under $AUDIOLAB_DATA
clips: 4000                               # how many clips one run trains on
updates: 5000                             # optimizer updates, not epochs
batch_size: 8
grad_accumulation: 2                      # clips per update = 8 x 2
lr_encoder: 1.0e-5                        # the encoder and its preprocessor
lr_heads: 3.0e-5                          # the decoder and the joint
weight_decay: 1.0e-3
schedule: constant                        # or: cosine
warmup_updates: 0
precision: bf16-mixed                     # or: 16-mixed, 32
grad_clip: 1.0
max_clip_seconds: 15.0                    # longer clips are left out of the run
workers: 4                                # dataloader processes
gpu: "0"                                  # an index or a UUID prefix
seed: 20261107
resume_every: 500                         # updates between resume-slot rewrites
log_every: 50                             # updates between log lines
```

`base_model`, `dataset`, `clips`, `updates`, `batch_size`, `lr_encoder` and
`lr_heads` are required: none of them has a value a run could guess, and a run
that guessed one would train something other than what the file says. Every
other field has a default and may be left out: `base_model_env` (`""`),
`grad_accumulation` (1), `weight_decay` (1e-3), `schedule` (`constant`),
`warmup_updates` (0), `precision` (`bf16-mixed`), `grad_clip` (1.0),
`max_clip_seconds` (20), `workers` (4), `gpu` (0), `seed` (20261107),
`resume_every` (500) and `log_every` (50).
**An unknown key is refused**, so a typo cannot silently train something else
than the file says. `clips` must be at least `batch_size`, `warmup_updates` must
be smaller than `updates`, and the two learning rates must be positive.

`base_model_env` is how a machine keeps a local copy of a checkpoint without
putting a path in the repository ([D-015](../owner/directives.md)): the variable
wins over `base_model` when it is set. The checkpoint is identified by the
sha256 of its file, or by its Hub id when it is not a local file, so `train.json`
says which weights the run started from without naming a machine.

`gpu` accepts an index (`1`) or a UUID prefix (`GPU-1a2b`); `--gpu` overrides it.
`audiolab gpus` lists what the machine has.

## 3. What a run does

1. It draws `clips` clips from the dataset's manifest: clips that are shorter
   than 0.3 s, longer than `max_clip_seconds` or carry no text are left out, and
   the rest are sampled with `seed` and sorted. The step prints how many were
   usable. Nothing is written before this succeeds, so a run that cannot find its
   data costs no directory.
2. It writes a NeMo manifest inside the run directory: one line per clip with an
   absolute path, its duration and its text. NeMo reads that, not our manifest.
3. It loads the base model on the CPU, replaces its training data settings with
   the run's, and builds AdamW through NeMo's own parameter groups: the encoder
   (and its preprocessor) at `lr_encoder`, the decoder and the joint at
   `lr_heads`, with the weight decay of the config. Before any GPU time is spent
   on an update, it checks on the loaded model that each part's parameters really
   sit in the group that carries their rate.
4. It trains with PyTorch Lightning: `updates` optimizer updates,
   `accumulate_grad_batches=grad_accumulation`, `max_steps=updates`, the
   precision and gradient clipping of the config, and a constant schedule with
   `warmup_updates` warmup steps — or a warmup followed by a cosine decay to
   `min_lr` when `schedule: cosine`.
5. Every `log_every` updates a line goes to `train-log.jsonl` (the update, the
   loss, and the seconds since the run started) and to the terminal. The loss of
   a line is the **mean of the batch losses seen since the previous line**: a
   training step returns the loss of one batch, and with
   `accumulate_grad_batches` above 1 that is not what the update optimised —
   PyTorch Lightning divides the value for the backward pass and averages the
   gradients over the batches, but never scales the number it reports. Averaging
   over the batches of the interval is the loss the run is at, on the same scale
   whether or not the config accumulates. That is the whole progress a run
   reports: this recipe watches the loss and nothing else.
   Training monitoring and official evaluation are separate things
   ([decision 0008](../decisions/accepted/0008-training-monitoring-vs-official-evaluation.md)),
   and a validation WER inside the training process can be added here later
   without touching how results are measured.
6. At the end it saves the model, writes `train.json` and a ready-to-use
   `model.yaml`, and deletes the resume slot.

## 4. What a finished run leaves behind

`$AUDIOLAB_RUNS/train/<name>/` holds:

| file | what it is |
|---|---|
| `config.json` | what the run was started as: the config, the base model and its identity, and the clips it drew. Written before the first update, never rewritten; a resume is checked against it (§5) |
| `manifest.jsonl` | the clips of this run, as NeMo reads them: absolute path, duration, text |
| `train-log.jsonl` | one line per logged update: `update`, `loss` (the mean of the batch losses seen since the previous line), `seconds` since the run started |
| `model.nemo` | the fine-tuned checkpoint, ready to serve |
| `model.yaml` | a model config for it, with a `serve:` block naming `model.nemo`: `audiolab eval --model <run>/model.yaml --suite configs/suites/fast.yaml` scores it on the spot |
| `train.json` | what the run did: the config, the dataset, the number of usable clips, the sha256 of the sampled clip ids, the base model and its identity, whether it went on from a resume slot, the git commit, the wall-clock time, the final logged loss, the updates per second, the peak GPU memory and the device name |

`train.json` is written last, and its presence is what makes a run *finished*:
`audiolab train` on a finished run refuses to start and says so. `--fresh`
trains it again from the base model, keeping the old files until the new run
writes over them.

A run directory is written outside the repository, and it names the machine it
ran on twice: the device in `train.json` and the audio paths in the manifest.
Nothing else in it is machine-specific, and none of it is ever committed.

## 5. Stopping and going on

Every `resume_every` updates (500 by default) the run rewrites **one** file,
`resume.ckpt`, which holds the weights *and* the optimizer state. It is a rolling
slot: the next rewrite replaces it, so a run keeps at most one, and a crash never
leaves a pile of checkpoints to choose from.

`audiolab train` on an unfinished run directory goes on from that slot, and says
`resuming from the last resume slot` when it does. Interrupting it (Ctrl-C, a
killed process, a reboot) is therefore safe: at most `resume_every` updates are
lost. When the run finishes, the slot is deleted: the final model is `model.nemo`.

A slot belongs to the run that wrote it, so a resume has to be that run: the
config in hand is compared with `config.json`, and a difference — a field of the
config, the base model's identity, or the clips the same seed now draws from a
dataset that was prepared again — refuses to resume and names what changed. Put
the old value back, or start another run with `--fresh`. A slot with no
`config.json` beside it is refused too: nothing on disk would say which run it
belongs to. `train.json` records whether the run was resumed.

The slot is the size of the weights plus the optimizer state — about 7 GiB for a
0.6 B model — and writing it takes a few seconds, so a small `resume_every` costs
time; the default is a good compromise, and the smoke tests that use 5 or 10 pay
for it visibly.

## 6. Memory, speed and the size of a batch

The TDT loss materialises a joint of `frames x tokens x vocabulary` for every
clip of a batch, so its memory grows much faster than the number of clips: it is
the clip *length* that decides what fits. Measured on a 32 GB GPU (a 0.6 B
Parakeet model, `bf16-mixed`, clips of a French dataset), starting from the
checkpoint's own `fused_batch_size`:

| batch | clips up to | peak memory | where it goes |
|---|---|---|---|
| 16 | 10 s | 21.9 GiB | fine |
| 8 | 15 s | 25.1 GiB | fine, with little room to spare |
| 8 | 20 s | — | out of memory part-way |
| 16 | 20 s | — | out of memory |

So `max_clip_seconds` and `batch_size` are a pair, and `grad_accumulation` is how
a run keeps its effective batch without the memory: 8 clips twice is the batch of
16 at the memory cost of 8, at twice the time per update. A run that does not fit
fails with `torch.OutOfMemoryError` on an unlucky batch, not at the first one.

Speed, on the same GPU with clips up to 10 s: about 0.5 s per update at batch 16
once the run is warm, and the low tens of clips per second. The first updates are
much slower — 8 to 30 s — because the loss kernel is compiled for the shapes of
the first batches it meets; a run of a few updates spends most of its time there,
a run of thousands does not notice. `updates_per_second` in `train.json` is the
average over the whole run, so it is low for short runs and low `resume_every`
values, and steady for real ones.

## 7. What a run does not do

- **No validation loop and no WER during training**: the loss curve is the only
  signal, and a loss is not a measure of quality. Score the checkpoint with
  `audiolab eval` — that is what the generated `model.yaml` is for. Training
  monitoring stays inside the training process
  ([decision 0008](../decisions/accepted/0008-training-monitoring-vs-official-evaluation.md)).
- **No checkpoint selection or early stopping.** A run trains the number of
  updates its config names and saves the model it ends with.
- **No dataset mixing.** One run reads one dataset; combining datasets, and
  measuring what each contributes, is a later phase ([roadmap](../roadmap.md)).
- **No frozen or sealed data.** A run reads whatever `dataset` names. Training on
  an evaluation panel would make its score meaningless, and nothing here prevents
  it: the choice of dataset is the person's.
