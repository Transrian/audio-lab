# Leakage audit — the VoxPopuli session overlap

**Status:** the audit behind the 2026-11-13 figures in the
[leakage table](../../docs/stt/datasets.md#training-sources), in the
[`voxpopuli-en-nonnative` card](../../docs/stt/datasets.md#voxpopuli-en-nonnative--accented-english-from-the-european-parliament)
and in [proposed decision 0014](../../docs/decisions/proposed/0014-voxpopuli-shared-sessions.md).
The open question those figures feed is whether the shared
sessions disqualify the train clips; this directory only reproduces the
figures.

## Definition and population

A *session* is the **prefix of the VoxPopuli clip id before the language
tag** — the part that names the plenary sitting:
`20090310-0900-PLENARY-7` in `20090310-0900-PLENARY-7-en_20090310-11:10:07_2`.
Clips of one session share the sitting, the room and the microphones. The
population is **the prepared data, not the corpora**: the 737-clip
`voxpopuli-en-nonnative` evaluation set and the prepared 12 000-clip
`voxpopuli-fr-train` sample — the same population the docs' denominators
("of the source's 12 000") refer to.

## Environment and inputs

From the repository root, use the default Python 3.12 environment
(`uv sync`). No additional dependencies, model stack or GPU are needed.

Inputs are read-only: the two prepared manifests
(`<data-dir>/voxpopuli-en-nonnative/manifest.jsonl` and
`<data-dir>/voxpopuli-fr-train/manifest.jsonl`). No audio is opened, no server
is started and no network request is sent; help needs neither data nor
environment variables.

## Exact command and output

```bash
uv run python experiments/leakage-audit/sessions.py --help
uv run python experiments/leakage-audit/sessions.py
```

On the prepared data of 2026-11-13:

```text
session = clip id's prefix before the language tag; population = the prepared manifests
  voxpopuli-en-nonnative: 737 clips, 307 sessions
  voxpopuli-fr-train: 12000 clips, 2265 sessions
shared sessions: 206 of 307
  evaluation clips in shared sessions: 480 of 737 (1.55 h)
  train clips in shared sessions: 1956 of 12000 (4.67 h)
```

## Limits

The audit is over prepared manifests: re-preparing either set changes the
figures, and the corpora hold far more sittings than the prepared sample. The
script asserts that every clip id holds a session before its language tag and
fails loudly otherwise; it does not re-derive the sessions from the corpora's
own metadata.
