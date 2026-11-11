# Label-text check: report, never rewrite

**Status:** the checker a preparation is checked with after it is written; the
script reports, it never re-prepares, rewrites, moves or deletes anything.

[`check_text.py`](check_text.py) reads the manifest of every named prepared
dataset (`$AUDIOLAB_DATA/<dataset>/manifest.jsonl`), calls
[`audiolab.data.train.unwritable`](../../src/audiolab/data/train.py) on every
clip's `text`, and reports, per dataset and per unwritable character, how many
clips contain it. Since 2026-11-09 every training label is written through
`plain_label`, so a preparation that ran that code holds no character the base
model's tokenizer cannot write: **a dataset with no such character prints
nothing** — silence is the pass. Called with no dataset name it checks every
`*-train*` directory of the data root.

## Environment and inputs

From the repository root, use the default Python 3.12 environment
(`uv sync`); the experiment calls the core's label rule
(`audiolab.data.train`) and manifest reader (`audiolab.data.store`) and adds
no dependency.

The input is read-only: the prepared datasets from the environment,

```bash
set -a; . ./.env; set +a
```

`$AUDIOLAB_DATA`. Help needs neither data nor environment variables:

```bash
uv run python experiments/label-text/check_text.py --help
```

## Reproduce the check

```bash
set -a; . ./.env; set +a
# Every prepared training set:
uv run python experiments/label-text/check_text.py
# One preparation right after it is written:
uv run python experiments/label-text/check_text.py granary-fr-train-40k
```

`--data-dir` overrides the root. The table goes to standard output; a missing
data root, a dataset without a `manifest.jsonl`, or a malformed manifest is a
standard-error error with exit `2`.

## Output

One Markdown table, `| dataset | character | clips |`: one row per dataset and
per character its clip labels hold, where `clips` is the number of clips that
contain the character (a clip that holds it twice counts once). The rows of
one dataset come in descending clip count, then character order. A dataset
without a row is clean and prints nothing; so must the check of every
preparation written since `plain_label` (2026-11-09).

## Limits

- The check reads the manifest's `text` field — the label as written — and
  calls the same `unwritable` the preparation itself uses, so the report and
  the preparation cannot disagree about the vocabulary. It never opens the
  audio and says nothing about it.
- It counts clips, not occurrences: two occurrences in one clip are one row
  of one clip.
- A set that predates `plain_label` is reported with the characters it holds;
  what a re-preparation would do with each is the rule's own business
  (`plain_label` maps, `unwritable` excludes). The script never performs a
  re-preparation — it touches what `src/audiolab/data/` produces, a strong area
  in [`AGENTS.md`](../../AGENTS.md).
- Characters the rule maps to a replacement (`…` to `...`, `—` to a break) are
  still reported where a label holds them, because the label as written is
  what a model trained on that set learns.

[Tests](../../tests/test_check_text.py) run the script on synthetic datasets:
the table format, the per-clip counting, the silence of a clean dataset, the
`*-train*` default, the missing-manifest failure, the malformed-manifest
failure, and a snapshot assertion that nothing under the data root is created,
written or renamed.
