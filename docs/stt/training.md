# Fine-tuning a model

This page explains how `audiolab train` fine-tunes one checkpoint on one
dataset — or on a mix of them —, what it writes, how a stopped run goes on, and
what it deliberately does not do. The code lives in `src/audiolab/train/`.

An `audiolab train` run is described by one config file in `configs/train/`
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

The pins cannot live in `uv.lock` today: one CUDA package version that PyTorch's
index resolves trips the private-address check in `scripts/check_repo.py`. Where
they belong instead, and what has to change for that, is
[decision 0012](../decisions/proposed/0012-heavy-model-stack-as-a-locked-extra.md)
— proposed, not decided. Until the owner rules on it, the heavy stack lives in
`requirements/nemo.txt` and in its own environment, and this page describes that
stopgap.

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

`base_model`, `updates`, `batch_size`, `lr_encoder` and `lr_heads` are required
in every config, and a config must name its data as either `dataset` + `clips`
(one dataset) or `datasets` (a mix, below): both, or neither, is refused. None
of them has a value a run could guess, and a run that guessed one would train
something other than what the file says. Every
other field has a default and may be left out: `base_model_env` (`""`),
`grad_accumulation` (1), `weight_decay` (1e-3), `schedule` (`constant`),
`warmup_updates` (0), `precision` (`bf16-mixed`), `grad_clip` (1.0),
`max_clip_seconds` (20), `workers` (4), `gpu` (0), `seed` (20261107),
`resume_every` (500) and `log_every` (50).
**An unknown key is refused**, so a typo cannot silently train something else
than the file says. `clips` must be at least `batch_size`, `warmup_updates` must
be smaller than `updates`, and the two learning rates must be positive.

`base_model_env` is how a machine keeps a local copy of a checkpoint while the
config names the public one: the variable wins over `base_model` when it is set,
so no committed file holds a path of this machine. Nothing a run reads or writes
enters the repository — no voice recordings or transcripts
([D-013](../owner/directives.md)), and no datasets, weights or run outputs. The
checkpoint is identified by the
sha256 of its file, or by its Hub id when it is not a local file, so `train.json`
says which weights the run started from without naming a machine.

`gpu` accepts an index (`1`) or a UUID prefix (`GPU-1a2b`); `audiolab gpus` lists
what the machine has. `audiolab train` has no `--gpu`: the GPU is part of the run,
so it belongs in the config (and in `config.json`, which a resume is checked
against).

### Mixing datasets

Instead of `dataset` + `clips`, a config may name several datasets, with a
share for each:

```yaml
datasets:
  - name: granary-fr-train
    clips: 30000
  - name: peertube-tech-fr-train
    clips: 20000
```

The names are prepared datasets, unique, each with an integer clip count of at
least 1, and no other key in an entry. The run's clip count is the **sum** of
the shares, and that total must still cover a batch (at least `batch_size`).
`configs/train/example-mix.yaml` is a commented template.

**Shares are clip shares.** Each dataset is drawn once at the start of the run,
and then every clip of every dataset is seen in the shuffled epochs, like a
single dataset. The shares decide how much of each dataset the run contains —
not the order the batches arrive in, and not how each epoch splits between the
datasets.

**One seed per dataset.** A dataset of a mix is drawn with the seed derived
from the run's `seed` and the dataset's name alone, so adding a dataset to a
mix — or reordering them — does not change the clips drawn from the other
datasets. The per-dataset draws are what the run records, per dataset.

**Duplicate audio is refused.** If the same recording was prepared into two
datasets of the mix (the same talk in two sets), the run is refused and lists
the first five duplicates: training it twice would count the same audio against
two shares. A repetition inside one dataset is that dataset's business, not
the mix's.

### The leakage guard

Every run — single dataset or mix — is checked before its model is loaded,
against the evaluation datasets of the suite: the datasets behind the panels
listed in `configs/suites/full.yaml`. A run is refused when

- a training clip is the **audio** of an evaluation clip (the same `sha256` —
  the hash of the prepared clip's PCM, so this is that prepared audio byte for
  byte), or
- a training clip is **named after** an evaluation dataset (its id prefix is
  one of the suite's dataset names).

An evaluation dataset that is not prepared on this machine cannot be checked:
the run prints one warning line naming it and goes on with a partial check,
rather than failing. The check prints a one-line summary of what it did.

What the guard does **not** do: it does not compare group ids across different
corpora, because ids collide by accident across corpora, and it does not
recognize a recording that is only the *same one*: the same audio cut, trimmed,
resampled, re-encoded or levelled differently hashes to something else and
passes. Keeping evaluation recordings out of a training preparation stays the
job of the fetchers, through their `evaluation_groups` exclusion; the guard is
the last check, on the clips a run is actually about to train. What it checked
is recorded in `train.json` under `leakage_guard` — the datasets checked, the
missing ones, the number of training clips checked. It is never in `config.json`,
because a resume is checked against that file field by field, and a new key
there would refuse every run written before the guard existed.

## 3. What a run does

1. It draws `clips` clips from the dataset's manifest: clips that are shorter
   than 0.3 s, longer than `max_clip_seconds` or carry no text are left out, and
   the rest are sampled with `seed` and sorted. The step prints how many were
   usable. For a mix, it draws each dataset this way — with the dataset's own
   seed — and the run trains the concatenation, sorted by id, after the
   [leakage guard](#the-leakage-guard) has passed. Nothing is written before
   this succeeds, so a run that cannot find its data costs no directory.
2. It writes a NeMo manifest inside the run directory: one line per clip with an
   absolute path, its duration and its text. NeMo reads that, not our manifest.
3. It loads the base model on the CPU, replaces its training data settings with
   the run's, and builds AdamW through NeMo's own parameter groups: the encoder
   (and its preprocessor) at `lr_encoder`, the decoder and the joint at
   `lr_heads`, with the weight decay of the config. Before any GPU time is spent
   on an update, it checks on the loaded model that each part's parameters really
   sit in the group that carries their rate. NeMo's training loader is kept but
   for its shuffle: every epoch reads the clips in a permutation drawn from
   `seed` and the epoch number alone (`src/audiolab/train/order.py`), so a run
   that is resumed reads the clips it would have read had it not stopped. The
   order is `random.Random(f"{seed}/{epoch}")` and every update sums exactly
   `grad_accumulation` batches, so a run trained before this order is not
   reproduced bit for bit.
4. It trains with PyTorch Lightning: `updates` optimizer updates,
   `accumulate_grad_batches=grad_accumulation`, `max_steps=updates`, the
   precision and gradient clipping of the config, and a constant schedule with
   `warmup_updates` warmup steps — or a warmup followed by a cosine decay to
   `min_lr` when `schedule: cosine`. Batch normalization keeps the base model's
   statistics: its layers stay in evaluation mode, and only their scale and
   shift train. Updated from batches of a few padded clips, the statistics
   drifted enough to wreck inference while the training loss kept falling
   ([screening](screening.md#the-batch-normalization-bug)); the test
   `tests/test_train_batch_norm.py` runs in the NeMo environment only.
5. Every `log_every` updates a line goes to `train-log.jsonl` (the update, the
   loss, and the seconds since the run started) and to the terminal. The loss of
   a line is the **mean of the batch losses seen since the previous line**.
   With `accumulate_grad_batches` above 1, PyTorch Lightning divides each batch
   loss by that number before the backward pass and hands the divided value to
   the hooks; the loop multiplies it back, so a line is on the same scale
   whether or not the config accumulates. On a terminal a progress bar also
   counts the updates (never in a log file). That is the whole progress a run
   reports: this recipe watches the loss and nothing else. A validation WER
   inside the training process is allowed — it is monitoring, not a result
   ([decision 0008](../decisions/accepted/0008-training-monitoring-vs-official-evaluation.md))
   — and would be added here.
6. At the end it saves the model, writes `train.json` and a ready-to-use
   `model.yaml`, and deletes the resume slot.

## 4. What a finished run leaves behind

`$AUDIOLAB_RUNS/train/<name>/` holds:

| file | what it is |
|---|---|
| `config.json` | what the run was started as: the config, the base model and its identity, and the clips it drew — for a mix, the clip hash per dataset as well. Written before the first update, never rewritten; a resume is checked against it (§5) |
| `manifest.jsonl` | the clips of this run, as NeMo reads them: absolute path, duration, text |
| `train-log.jsonl` | one line per logged update: `update`, `loss` (the mean of the batch losses seen since the previous line), `seconds` since the run started |
| `model.nemo` | the fine-tuned checkpoint, ready to serve |
| `model.yaml` | a model config for it, with a `serve:` block naming `model.nemo`: `audiolab eval --model <run>/model.yaml --suite configs/suites/fast.yaml` scores it on the spot |
| `train.json` | what the run did: the config, the dataset (for a mix, `datasets` — per-dataset clip counts, usable clips and clip hash — instead), the number of usable clips, the sha256 of the sampled clip ids, the base model and its identity, whether it went on from a resume slot, the git commit, the wall-clock time, the final logged loss, the updates per second, the peak GPU memory, the device name, and the `leakage_guard` block (§2) |

`train.json` is written last, and its presence is what makes a run *finished*:
`audiolab train` on a finished run refuses to start and says so.
`audiolab train --fresh` deletes that directory first — its resume slot, its log
and its checkpoint — and trains the config again from the base model.

A run directory is written outside the repository, and it names the machine it
ran on twice: the device in `train.json` and the audio paths in the manifest.
Nothing else in it is machine-specific, and none of it is ever committed.

## 5. Stopping and going on

Every `resume_every` updates (500 by default) the run rewrites **one** file,
`resume.ckpt`, which holds the weights *and* the optimizer state. It is a rolling
slot: the next rewrite replaces it, so a run keeps at most one, and a crash never
leaves a pile of checkpoints to choose from.

`audiolab train` on an unfinished run directory goes on from that slot, and says
`resuming from the last resume slot` when it does. When the run finishes, the
slot is deleted: the final model is `model.nemo`.

To stop a run, press Ctrl-C or send it SIGTERM (`kill <pid>`, `systemctl stop`).
The run finishes the update in progress (with `grad_accumulation` above 1, the
batches of that update), rewrites the slot, prints

```
audiolab: training of <name> stopped after update <n> of <updates>, kept in <run>/resume.ckpt; run the same command again to resume
```

and exits with code 130. A stopped run writes neither `model.nemo` nor
`train.json` nor `model.yaml`: the directory holds `config.json`, the manifest,
the log so far and the slot. The same `audiolab train` command goes on from update
`<n>`, on the clips that came next, with the random state it stopped at (below).
A second Ctrl-C or SIGTERM ends the process at once; so do a crash, `kill -9` and
a reboot. Then the last rewrite of the slot is what the run goes on from, so at
most `resume_every` updates are lost. The slot is written to
`resume.ckpt.partial` and renamed, so a process killed while writing it keeps the
slot before.

A resumed run trains what the run that went on would have, to the bit when the
run is deterministic. The slot carries the random generators of Python, NumPy,
PyTorch and every GPU, which draw the dither, SpecAugment's masks, the dropouts
and the batches the TDT loss trains with the plain RNN-T loss instead; the resume
puts them back before its first batch. One random state lives outside them: on
a GPU, cuDNN draws the dropout between the two layers of the prediction
network's LSTM from a state of its own, and PyTorch draws that state again from
the GPU's generator only after the generator is set. So the run sets the
generator to its own state before every batch, and the cuDNN state is drawn from
what the slot carries (`redraw_cudnn_dropout` in
`src/audiolab/train/finetune.py`). An audiolab without this redraw drew other
LSTM masks, so a run it trained is not reproduced bit for bit.
`tests/test_train_resume_slot.py`, which runs in the NeMo environment only,
checks with a small model that draws from PyTorch's and Python's generators and
runs a two-layer LSTM with dropout that a run stopped in the middle of an update, or killed after
a rewrite of the slot, and resumed ends with the weights of the run that went on,
to the bit: on the CPU, and on the GPU when the machine has one.

Under systemd, `KillMode=mixed` sends SIGTERM to `audiolab` alone and lets it stop
its data loader workers itself, and `SuccessExitStatus=130` makes a stopped run
count as a clean stop rather than a failure. `TimeoutStopSec` has to cover one
update and one slot write: a stop took about a minute in our runs, and a slot
write took up to five when two runs wrote theirs at once, so give it ten minutes.

On a GPU (2026-11-09), a 120-update run stopped by SIGINT and then SIGTERM, and
killed once outright, resumed each time from its slot and finished; its losses
up to the first stop were those of an uninterrupted twin to the last digit, and
after a resume they stayed within the twin's noise only: its slot did not carry
the random state yet. With the random state and the cuDNN redraw, an 80-update run
stopped by SIGTERM after update 51 and resumed logged the losses of its twin to
the last digit at updates 70 and 80.

A slot belongs to the run that wrote it, so a resume has to be that run: the
config in hand is compared with `config.json`, and a difference — a field of the
config, the base model's identity, or the clips the same seed now draws from a
dataset that was prepared again — refuses to resume and names what changed. Put
the old value back, or start another run with `--fresh`. A slot with no
`config.json` beside it is refused too: nothing on disk would say which run it
belongs to, and so is a slot that does not record how many batches it trained
(written by an audiolab older than the clip order of §3): resume it with that
audiolab, or start over with `--fresh`. `train.json` records whether the run was
resumed.

The slot is the size of the weights plus the optimizer state — about 7 GiB for a
0.6 B model — and writing it takes about a minute, so a small `resume_every` costs
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

- **No validation loop and no WER during training**: this is a choice for the
  short runs the project needs, not a rule. A held-out WER inside the training
  process is monitoring, and stays as simple as possible
  ([decision 0008](../decisions/accepted/0008-training-monitoring-vs-official-evaluation.md));
  it can be added later without touching how results are measured. Until then the
  loss curve is the only signal while a run goes, and a loss is not a measure of
  quality: score the checkpoint with `audiolab eval` — that is what the generated
  `model.yaml` is for.
- **No checkpoint selection or early stopping.** A run trains the number of
  updates its config names and saves the model it ends with.
