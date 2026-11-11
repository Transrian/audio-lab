# Storage inventory (plan 09, P-07): list, never delete

**Status:** the read-only inventory behind the retention question (Q8, P-07).
The script classifies, the owner decides: it never proposes a deletion in code
and never deletes, moves or renames anything. The keep list below is this
experiment's own statement of what must not be proposed for deletion, and the
home of the full reasons; the report rows only point at it.

**Outcome (2026-11-10):** the owner answered the retention question; the
unreferenced candidates were freed, the doc-cited diagnostic run kept.
Machine-specific detail stays in the ignored local notes.

[`inventory.py`](inventory.py) walks the three storage roots named by the
environment and reports, per directory, what the repository says about it:

- size (an `os.scandir` walk with `stat`, read-only; not `du` parsing), file
  count, newest file mtime;
- who references the name: `docs/`, `configs/` and `experiments/` (this topic
  directory excluded), the `configs/train/*.yaml` `dataset:` values and the
  `configs/panels/*.txt` `# dataset:` headers;
- one of these classes: `keep (rule above)`, `keep (referenced)`,
  `candidate: re-creatable`, `candidate: unreferenced`, `unknown`.

## The keep list

The `keep (rule above)` logic is:

- the prepared datasets used by a frozen panel or a `configs/train/*.yaml`;
- the talk-source answer caches: their model answers are kept, the downloaded
  audio inside them is a candidate (the second table splits the two);
- `screen-peertube-fr`, the control run of the noise experiment (plan 02);
- `mls-fr-train` and `librispeech-en-train`, where the noise experiment takes
  its babble noise from (plan 01 T4);
- `nasa-en-train.curly-2026-11-09` and `screen-nasa-en`, until the NASA
  re-screening is documented (plan 05 Part C), and
  `icsi-en-train.pre-plain-label-2026-11-11`, the frozen input of the ICSI
  screening: its 2026-11-08 run trained on it, so re-making that screen means
  keeping it ([datasets card](../../docs/stt/datasets.md));
- the evaluation run directories cited by `docs/stt/screening.md` or
  `docs/stt/evaluation.md`, plus the transcript cache that holds the per-clip
  answers those docs quote. The docs can also quote a run's numbers in prose
  without naming the directory; those runs are listed in the script's
  `DOC_CITED_EVAL_RUNS` constant, each with the doc section its citation comes
  from.

Each rule carries a comment in the script.

The special cases:

- **talk caches**: the downloaded audio (re-downloadable from the pinned
  listings) is separated from the model answers (expensive, and what makes
  larger preparations cheap, plan 03) using the file layout
  [`src/audiolab/data/talks.py`](../../src/audiolab/data/talks.py) writes
  (`<talk id>.wav` audio, `<talk id>.<model>.jsonl` answers, NASA's cached
  `pages/`), and both sizes are reported per source;
- **screening checkpoints**: each `model.nemo` is sized, the runs a plan still
  uses are the keeps, the rest are candidates — a screen checkpoint is
  retrainable when a committed `configs/train/*.yaml` of its own exists, and
  one without such a recipe is not retrainable from the repository. The small
  evaluation answers of a checkpoint (`runs/eval/<checkpoint>`, the transcript
  cache) must stay whatever class the checkpoint gets, because they are what
  the numbers in the docs were computed from;
- **job logs** (`$AUDIOLAB_RUNS/label-logs`, and the like): disposable,
  regenerable output of long jobs, classed by name. They are still only ever
  listed as a candidate, like everything else: nothing is deleted without the
  owner's go.

The report rows carry the short version of these reasons; this page and the
keep list above are where the full ones live.

Entries the script cannot classify are `unknown` and are reported, not
improvised (global rule 13).

## Environment and inputs

From the repository root, use the default Python 3.12 environment
(`uv sync`); the experiment reads the core's registries
(`audiolab.data.fetch`, `audiolab.data.talk_sources`) for the dataset and
talk-source names, and adds no dependency.

Inputs are read-only:

- the storage roots from the environment:

  ```bash
  set -a; . ./.env; set +a
  ```

  `$AUDIOLAB_CACHE` (download cache), `$AUDIOLAB_DATA` (prepared datasets),
  `$AUDIOLAB_RUNS` (runs and transcript cache);
- the repository's `docs/`, `configs/` and `experiments/` (default: the
  repository that contains this script).

No server is started, no subprocess runs, no network request is sent, and
nothing is written under the storage roots - not even a temporary file. Help
needs neither data nor environment variables:

```bash
uv run python experiments/storage-inventory/inventory.py --help
```

## Reproduce the inventory

```bash
set -a; . ./.env; set +a
uv run python experiments/storage-inventory/inventory.py \
    > /path/outside/the/repo/inventory.md
```

`--data-dir`, `--runs-dir`, `--cache-dir` and `--repo-root` override the
environment and the default repository root. The Markdown report goes to
standard output. Exit codes: `0` for a complete inventory, `2` when a root is
missing or not a directory (standard error, no partial report), and `3` when
some path could not be read: the report then lists it under
"Warnings: paths that could not be read" and the affected rows are measured
as 0 B. The result is pasted into the local plan notes (git-ignored, like all
machine facts); commit only this experiment, never the report.

## Output

Root totals and free space, then one table per root (cache, data, runs, and
one level below into `runs/train` and `runs/eval`): path, size, file count,
newest mtime, class, and the references/notes that led to it. A second table
splits every talk cache into downloaded audio (candidate) and model answers
(keep). The per-class totals count every leaf row exactly once - the
`runs/train` and `runs/eval` containers are summary rows, not leaves.

## Limits

- The reference check matches the name as an identifier: no word character or
  hyphen may touch it, so `screen-peertube-fr` does not match inside
  `screen-peertube-fr-train`, while a file name like `screen-mls-fr.yaml`
  counts as a reference. A name the repository only mentions in prose still
  counts; reading the notes column is part of the review.
- "Re-creatable" means the dataset is prepared (its `manifest.jsonl` exists)
  and the dataset is in the script's explicit pinned list (`PINNED_DATASETS`):
  its fetcher pins the download to a fixed digest (a digest table, the ESC-50
  csv sha, or the Hub's published sha256 of a pinned revision), so the archive
  can be fetched again and verified. The list is written out by name, not
  guessed from the fetchers' source text, and a test pins it: a fetcher that
  starts or stops pinning changes that test, never a cache row silently. A
  download whose source can change or vanish under us is not called
  re-creatable, even when it looks fetchable again.
- The screening checkpoints are candidates (see the special cases above), with
  one exception made explicit: a checkpoint without a `configs/train/*.yaml` of
  its own is not retrainable from the repository, and the table says so.
- The talk split is by file name, following the layout `talks.py` writes; a
  file with an unexpected name lands in "other input", never in the answers.
- A path that cannot be read (permissions, a vanished file) is counted as 0 B
  and listed in the report's warnings section, which makes the run exit `3`;
  the script never guesses at an unreadable size.
- Sizes are apparent (stated) sizes of every file, including partial
  downloads and logs; the newest mtime is the newest file's, not the
  directory's.

[Tests](../../tests/test_storage_inventory.py) run the script on synthetic
roots: every classification above, the talk split, the totals, and a snapshot
assertion that nothing under the storage roots is created, written or
renamed.
