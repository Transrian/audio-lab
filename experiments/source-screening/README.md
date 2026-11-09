# Source-screening reports

**Status:** reporting helper for existing per-source STT screens. Follow-up
training and data work remain paused in the [backlog](../../docs/backlog.md).
The [screening document](../../docs/stt/screening.md) is the home of the question,
method, runs, results and scientific limits; this page documents only the helper.

[`table.py`](table.py) compares each named candidate with an explicit baseline.
It calls the same comparison and renderer as `audiolab compare`, with the frozen
bootstrap defaults. It adds no metric, scorer, ranking or model request. A
separate table shows absolute formatted WER from the stored summaries.

## Environment and inputs

From the repository root, use the default Python 3.12 environment (`uv sync`).
No additional dependencies, model stack or GPU are needed.

Inputs are read-only:

- finished evaluation directories containing `run.json`, `records.jsonl` and
  `summary.json`;
- their original frozen panel files (`--panels-dir`, default `configs/panels`);
- dataset manifests containing the references and groups (`--data-dir`, default
  `$AUDIOLAB_DATA`, then the package default).

No audio is opened, no server is started and no network request is sent. Help
needs neither data nor environment variables:

```bash
uv run python experiments/source-screening/table.py --help
```

## Reproduce a report

Use the campaign's frozen scoring code and original dataset manifests. Name
every candidate explicitly; there is no default source list or hidden baseline.

```bash
uv run python experiments/source-screening/table.py \
    --baseline /path/to/runs/eval/base/full \
    --candidate granary=/path/to/runs/eval/screen-granary-fr/full \
    --candidate icsi=/path/to/runs/eval/screen-icsi-en/full \
    --data-dir /path/to/data \
    > /path/to/runs/screening-table.md
```

Repeat `--candidate NAME=PATH` for further candidates; labels must be unique.
Output is Markdown on standard output. The script writes no input files and
creates no directories. Keep generated reports and other run artifacts outside
the repository; commit only curated findings in their documented home.

## Output and validation

Each candidate gets an official paired comparison: normalized WER on speech,
hallucination rate on non-speech, the `candidate - baseline` difference, its
95 % interval and the official verdict. The common renderer preserves exclusion
counts and IDs, scoring errors and small-group interval cautions. Item and group
counts are also shown per panel.

**The baseline value is per candidate, not shared across all columns.** Runs
with failed answers can have different paired subsets. Failed answers remain
visible as exclusions; they are not treated as correct answers or silently
removed from the report.

The formatted-WER table reads each run's own summary, including a genuine zero.
Its means may cover different answered subsets. It is not a paired verdict and
has no new confidence interval.

The script refuses missing or malformed inputs, incomplete panel coverage,
unknown recorded IDs/panels, changed panel-file hashes and mismatched panel sets.
It rejects sealed final panels: this is an exploratory screening helper, not a
final-test command. Success exits `0`; input or argument errors exit `2` on
standard error, without a partial report on standard output.

## Limits

The script recomputes paired comparisons from retained answers; it never
retranscribes or changes stored scores. Formatted WER is read as stored: it does
not verify summary freshness or the full provenance of manifest contents. Use
matching records, summaries, manifests and the campaign's scoring revision. If a
summary must be rebuilt, use `audiolab score` under its original scoring rules,
not an unreviewed change to the campaign protocol.

One short run per source does not establish a general ranking or choose a mix.
The screening's scientific limits still apply. Training metadata and the
historical tables are not regenerated automatically by this helper.
