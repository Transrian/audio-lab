# 0012 — The heavy model stack becomes a locked, optional extra

- Scope: global
- Derived from: D-010, decision 0003

## Decision

Training a checkpoint and serving one locally need PyTorch, NeMo, Numba and a CUDA
compiler — about ten gigabytes of wheels, and a machine with an NVIDIA GPU. They
are not dependencies of the project: they live in `requirements/nemo.txt`, which
[`scripts/nemo-venv.sh`](../../../scripts/nemo-venv.sh) installs into its own
`.venv-nemo`.

That should change into something a user installs with one command:

```toml
[project.optional-dependencies]
nemo = ["nemo_toolkit[asr]==...", "torch==...", ...]
```

locked in `uv.lock` like every other dependency, so that `uv sync --extra nemo`
is all it takes to fine-tune or serve a local model. The default install stays
light: scoring a model through an API keeps needing nothing heavy
([D-015](../../owner/directives.md)).

One check stands in the way, and it is the owner's to decide:

> `scripts/check_repo.py` refuses anything that looks like a private address in a
> committed text file, and it reads `uv.lock` too. The CUDA wheel that PyTorch's
> own index resolves, `nvidia-curand-cu12`, carries a version whose four numbers
> (starting with `10.`) read as a private address to that pattern. So the pins
> cannot be in `uv.lock` at all today.

**The question:** may `scripts/check_repo.py` skip `uv.lock` (or exempt the two
CUDA packages by name)? A lock is machine-generated and holds resolved versions
of public packages, while the pattern exists to catch addresses in hand-written
files, so skipping it looks safe — but that is the owner's call.

## Why

- Anyone who follows the README should reach a working environment with one
  command, and two environments that are installed in different ways is exactly
  the kind of thing [D-010](../../owner/directives.md) asks to avoid.
- A lock that cannot be generated is worse than none: a hand-maintained pin list
  drifts from `uv.lock`, and the differences show up as confusing resolver
  conflicts (for example a package that pins `numpy<2.3` while the light install
  asks for a newer one).
- CI must stay fast: the heavy stack stays out of the default environment either
  way, whether it is an extra or a separate requirements file
  ([decision 0003](../accepted/0003-ci-on-every-push.md)).

## Alternatives considered

- **Keeping the stopgap as it is** (a requirements file and a second
  environment). What runs tonight, 2026-11-08; it works, at the cost of a second
  environment, a second install command and an `AUDIOLAB_NEMO_PYTHON` variable,
  with the pins outside the lock.
- **Running the heavy stack in a CUDA container.** Tried on 2026-11-08 and
  dropped: the CUDA images ship no Python, so the environment cannot be created
  inside one without more machinery than this project wants.
