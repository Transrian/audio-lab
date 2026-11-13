# Exploratory experiments

Versioned code for investigating a question without making it a supported
feature of the maintained `audiolab` core
([decision 0007](../docs/decisions/accepted/0007-single-package-single-command.md)).

| Experiment | Purpose | Entry point |
|---|---|---|
| [Source screening](source-screening/README.md) | Render comparisons of already evaluated per-source training runs | [table.py](source-screening/table.py) |
| [Normalization review](normalization-review/README.md) | Re-score finished screening runs with contraction and elision rules on top of the frozen normalizer (plan 07, P-04; diagnostic only) | [rescore.py](normalization-review/rescore.py) |
| [Label-text check](label-text/README.md) | Report how many training clips hold a character the base model cannot write, after each preparation | [check_text.py](label-text/check_text.py) |
| [Storage inventory](storage-inventory/README.md) | Read-only size, reference and keep/candidate classification of the cache, data and run directories (plan 09, P-07) | [inventory.py](storage-inventory/inventory.py) |
| [Leakage audit](leakage-audit/README.md) | Reproduce the VoxPopuli session-overlap table from the two prepared manifests (plan 01, T8) | [sessions.py](leakage-audit/sessions.py) |

The [research index](../docs/research/README.md) links questions and findings.
The [backlog](../docs/backlog.md) tracks open work and pauses; creating an
experiment directory does not resume a paused campaign.

## One directory per topic, only as much structure as needed

An experiment can have its own script, configuration and dependency environment:

```text
experiments/<topic>/
  README.md          # question, status, exact commands, evidence and limits
  run.py             # only when new runnable code is needed; name it for its job
  config.yaml        # when configuration is useful
```

This is an example, not a scaffold every topic must copy. A literature review
needs no script, and an existing core command may need only a configuration.
Keep a result in one place: link the relevant research or domain page rather
than repeating its protocol and tables in the experiment README.

Each runnable experiment documents its environment, inputs, exact invocation,
output location and limits. Pin any additional dependencies in the topic's own
requirements or environment specification; do not add them to the default
installation solely for that exploration. Put local virtual environments in
`.venv/`, which is already ignored. Do not commit datasets, audio, weights,
checkpoints, caches or generated run outputs; use external storage and reference
the artifacts in the research note.

## Dependency boundary

Use `audiolab.core` and the existing data/evaluation components when they fit.
The experiment may depend on the core; the core must not import the experiment.
Do not add a general framework or core options for a speculative use case. A
small local helper can be preferable to a premature shared abstraction. Code
moves into the core only when there is a concrete reason to maintain it there.

Add fast synthetic CPU tests under [`tests/`](../tests/) for new behavior. The
normal repository checks must not need a real GPU, downloaded dataset or model
weights. Run GPU campaigns separately and document how; all applicable rules,
including selectable GPUs, caching, progress and stop/resume for long tasks,
still apply. Raise a conflicting rule for owner review rather than silently
exempting research code.

Official STT comparisons reuse the
[common API-based evaluation protocol](../docs/stt/evaluation.md). Never access a
sealed final set for exploratory tuning or replace official metrics with a
private experimental scorer. See the
[research boundaries](../docs/research/README.md#boundaries).

A negative or inconclusive experiment may remain here with its conclusion and
reproduction instructions. It need not become a new subcommand or permanent
maintenance commitment.
