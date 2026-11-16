# Leakage audit

Three read-only audits over prepared manifests, in this directory:

- the [VoxPopuli session overlap](#the-voxpopuli-session-overlap) (2026-11-13);
- the [content-level leakage over 6-word text shingles](#content-level-leakage-6-word-text-shingles) (2026-11-15);
- the [rebuild impact report](#the-rebuild-impact-report) (2026-11-16).

## The VoxPopuli session overlap

**Status:** the audit behind the 2026-11-13 figures in the
[leakage table](../../docs/stt/datasets.md#training-sources), in the
[`voxpopuli-en-nonnative` card](../../docs/stt/datasets.md#voxpopuli-en-nonnative--accented-english-from-the-european-parliament)
and in [decision 0014](../../docs/decisions/accepted/0014-voxpopuli-shared-sessions.md).
The question those figures fed — whether the shared
sessions disqualify the train clips — is answered by that decision (option 2:
drop them); this directory only reproduces the
figures.

### Definition and population

A *session* is the **prefix of the VoxPopuli clip id before the language
tag** — the part that names the plenary sitting:
`20090310-0900-PLENARY-7` in `20090310-0900-PLENARY-7-en_20090310-11:10:07_2`.
Clips of one session share the sitting, the room and the microphones. The
population is **the prepared data, not the corpora**: the 737-clip
`voxpopuli-en-nonnative` evaluation set and the prepared 12 000-clip
`voxpopuli-fr-train` sample — the same population the docs' denominators
("of the source's 12 000") refer to.

### Environment and inputs

From the repository root, use the default Python 3.12 environment
(`uv sync`). No additional dependencies, model stack or GPU are needed.

Inputs are read-only: the two prepared manifests
(`<data-dir>/voxpopuli-en-nonnative/manifest.jsonl` and
`<data-dir>/voxpopuli-fr-train/manifest.jsonl`). No audio is opened, no server
is started and no network request is sent; help needs neither data nor
environment variables.

### Exact command and output

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

### Limits

The audit is over prepared manifests: re-preparing either set changes the
figures, and the corpora hold far more sittings than the prepared sample. The
script asserts that every clip id holds a session before its language tag and
fails loudly otherwise; it does not re-derive the sessions from the corpora's
own metadata.

## Content-level leakage (6-word text shingles)

**Status:** 2026-11-15. The confirmed duplicate pairs are committed in
[`src/audiolab/data/duplicates.json`](../../src/audiolab/data/duplicates.json).
Every reported pair below is classified by the leak policy: a leak is the
same speech as an evaluation group — a re-upload, a republication or a
replay; a shared talk or podcast host, or the same text read by another
corpus speaker, is not a leak.

**Question.** Which training groups hold the same speech as an evaluation
group — a re-upload, a republication or a replay — and which text overlap is
only "the same text, another reader" or a shared corpus speaker (not a leak)?

**Method.** Words are NFKD-lowercased `[a-z0-9]+` tokens; the 6-word shingles
of every group are compared. An evaluation group and a training group are
reported when they share at least 15 shingles and the shared shingles are at
least 5 % of the evaluation group's. Pairs of *talk groups* (`group_kind =
"recording"`) inside two different training sets, and inside one set, are
reported at the same k ≥ 15 floor with a ratio of at least 30 % of the
smaller group; the re-upload shortlist keeps the pairs at 75 % or more with
different group ids. "Same speech" is confirmed by (a) a ratio of at least
30 %, or (b) for talk groups, the mean word recall of the evaluation clips'
words in the time-overlapping training clips of at least 0.7 (clip starts
from the trailing centiseconds of the clip id). A pair between 5 % and 30 %
without that recall is triaged on the shared text itself: a recurring jingle
or stock intro is not a re-upload, a compilation that replays an episode is.
Shared audio is checked by clip sha256; shared speakers by the
corpus-provided speaker id.

**Population.** Prepared sets only: the 15 prepared evaluation sets with
panel files in `configs/panels/` (the two non-speech sets excluded) and the
25 prepared training directories (names containing `-train`, with the
`.pre-*` and `.curly-*` renames excluded). The audit reads manifests only —
no audio, no server, no network; it is CPU-only and reads one manifest at a
time, keeping the groups in memory (about 3.5 GB peak on this data).

**Exact command.**

```bash
uv run python experiments/leakage-audit/content.py --help
uv run python experiments/leakage-audit/content.py --triage   # full per-pair evidence
```

(`--data-dir` / `--panels-dir` override `$AUDIOLAB_DATA` / `configs/panels`.)

**Results, 2026-11-15.** 518 evaluation-vs-training pairs were reported:

| Triage category | Pairs | Leak? |
|---|---|---|
| Same recording, ratio ≥ 30 %: the three `peertube-edu-captions-fr` talks re-uploaded in the training sets (word recall 0.93 / 0.84 / 0.81 over the time-overlapping clips) | 3 | Yes — the same speech (re-upload) |
| Same recording, ratio ≥ 30 %: the re-released Terry Fong interview, in all three NASA training sets (40.5 / 40.5 / 33.0 %) | 3 | Yes — the same speech (republication) |
| Replay below the 30 % / 0.7 gates: the 2017 end-of-year test compilation replays two training episodes in segments (20.6 % and 9.0 % of its 6-word shingles); classified as replay by inspecting the shared text, not by threshold | 2 id-pairs (6 set-pairs) | Yes — the same speech (replay) |
| Same text, another reader: SLR57 short read-task sentences read or spoken by different speakers (0 shared audio sha256, 0 matching corpus speaker numbers) | 374 | No — the same text read by another corpus speaker |
| Same text, another reader: one LibriSpeech chapter pair (8463-294828 ↔ 2764-36617) under different corpus speaker ids, 0 shared audio sha256 | 3 | No — the same text read by another corpus speaker |
| Recurring show intro: the shared text is the show's stock opening (host line / "On this podcast, we bring in the experts…" / "Our solar system is a wondrous place…" / "Welcome to Small Steps, Giant Leaps…"), repeated in every episode of the show (5.1–8.4 %, k≈23–46); no conversation text shared, 0 shared audio; not a leak (a shared host and a stock intro, no conversation text) — per family: ascension-looking-back-at-the-space-shuttle 88; gravity-assist-the-kuiper-belt-with-alan-stern 25; pathways 5; dna-sequencing 5; apollo-legend-to-shuttle-trailblazer 5; small-steps-giant-leaps-episode-104-workplace-safety-culture 1 | 129 | No — a shared host and a stock intro, no conversation text |

Training-vs-training: 2 196 talk-group pairs across two sets at ratio ≥ 30 %
(mostly by design: the 12k ⊂ 40k/30k supersets, the same episodes under the
12k / 30k / qwen label sets, talks shared across the three PeerTube
listings), 42 pairs inside one set, and a re-upload shortlist of 58 rows at
ratio ≥ 75 % with different group ids: 18 distinct PeerTube id-pairs (in
[`duplicates.json`](../../src/audiolab/data/duplicates.json), which also
holds the three evaluation↔training re-upload pairs), 16 distinct granary
id-pairs, and the NASA `introducing-nasas-curious-universe-season-two`
↔ `crew-1-the-astronauts` pair — the granary pairs and the NASA pair are
kept in this table only: `duplicates.json` covers the PeerTube and NASA
listings, and the `introducing` episode is not in the NASA listing (the
Curious Universe show is excluded from it, so the rebuilt training sets will
not read it). The 79.8 % shortlist row is
`nasa-en-train-30k:crew-1-the-astronauts` ↔
`nasa-en-train-qwen:introducing-nasas-curious-universe-season-two` (75 of
94 six-word shingles); `nasa-en-train:introducing-nasas-curious-universe-season-two`
↔ `nasa-en-train-30k:crew-1-the-astronauts` is 71.0 % (66 of 93), and the
episode also replays `gravity-assist-life-in-the-clouds-with-david-j-smith`
at 58.1 % (54 of 93), both below the 75 % shortlist. The 16 granary
id-pairs: J6MMI5tsnJk↔McLkUdBrlXg, KWQcxTPKZMU↔RfZPaQ4D3F0,
zB-T6Urw3zy↔XojSl5HZR64, sBvYAg3ADK8↔FEQqqOKYrvy,
M2Z2S7rstIQ↔aMuBAqtMmfA,
06LOOzSIAcc↔GZZxF4ID89A, wf5_2_uUiP8↔EdogqOHuB-A,
JvZRK0c8hE0↔{M2Z2S7rstIQ, aMuBAqtMmfA, gpTL6tbUhhc},
{3zwOPq3HJD8, ZMfbCNYzBb4, hsoj91JTjJU},
{M2Z2S7rstIQ, aMuBAqtMmfA}↔gpTL6tbUhhc, kgdA3-q9ZTA↔o95HCvmGPKc.

**Limits.** The audit covers prepared sets only: `nasa-en-train-30k` holds
575 episodes while 122 of the listing's 708 non-test episodes (the 768
listing minus the 60 reserved test episodes) hold in none of the three
prepared NASA training sets and were never prepared, and the 2017 end-of-year
compilation is only about 30 % explained by the two replayed episodes; a
re-run of this script on the rebuilt NASA training sets will confirm that no
`nasa-en-test` group has a ≥ 5 % pair other than the stock-intro pattern. The
two replayed-episode pairs sit below the 30 % / 0.7 gates because the
compilation cuts and re-edits the segments (clip boundaries do not line up:
0 shared sha256, time-overlap word recall 0.16–0.26); they were classified as
replays by inspecting the shared text, not by threshold. Disclosure, not a
removal: about one clip of each affected test episode (3–4 for Gravity
Assist) is the show's stock intro, which the training sets also contain. The
`-qwen` ratios differ from the 12k ratios because the qwen set carries
different labels on the same audio. Every reported pair of this run falls in
one of the categories above.

## The rebuild impact report

**Status:** 2026-11-16. The read-only script that measures what the corrected
fetchers' exclusion rules remove from every prepared training set, and which
sets that forces to be re-prepared. The table over the current data root lands
here when the last evaluation set is prepared; the script is safe to run in
the meantime and is safe to re-run any number of times.

**Question.** For each prepared training set, how many clips does each
exclusion rule remove, and which sets must a re-preparation re-run because of
it?

**Method.** The rules are never re-implemented here: every table column is the
committed function the fetcher itself uses, applied to the set's current
`manifest.jsonl`. The one-to-one mapping (a reviewer can check every number
against the fetcher's code):

| Table column | The committed function |
|---|---|
| `eval class` | [`duplicate_classes`](../../src/audiolab/data/duplicates.py) + [`excluded_talks`](../../src/audiolab/data/duplicates.py), with the evaluation groups the prepared evaluation sets of the source hold — [`evaluation_groups`](../../src/audiolab/data/train.py) over the source's `evaluation_datasets`, so a v2 evaluation set counts as soon as it is prepared |
| `in-set re-upload` | the fetchers' committed selection restricted to one set's clips: the first talk of each duplicate class in the reading order ([`reading_order`](../../src/audiolab/data/talk_listing.py) with the seed of [`SEED`](../../src/audiolab/data/talks.py) and the source's `spread_by`), after the evaluation exclusion the fetchers make first — the loops of `fetch_talks` and `fetch_nasa_en_train` |
| `nasa reserved` | [`RESERVED_TEST_EPISODES`](../../src/audiolab/data/nasa.py) through the same `excluded_talks` |
| `slr57 speaker` | [`excluded_train_speakers`](../../src/audiolab/data/african_french.py) over the groups of the prepared `african-accented-fr-test` |
| `vox session` | the committed pin [`EVALUATION_SESSIONS`](../../src/audiolab/data/voxpopuli_train.py) with [`session_of`](../../src/audiolab/data/voxpopuli_train.py) per clip id |
| `babble` | the committed pin [`load_babble_utterances`](../../src/audiolab/data/noisy.py) with [`babble_exclusions`](../../src/audiolab/data/librispeech_pc.py) per clip's utterance id |

**Population.** Every prepared training set: a directory of the data root
(default `$AUDIOLAB_DATA`) whose name contains `-train`, excluding every
`.pre-*` and `.curly-*` rename, holding a `manifest.jsonl`. The report first
states which evaluation sets it found prepared, so an evaluation set that is
missing at run time is visible, not silent.

**Environment and inputs.** The default Python 3.12 environment (`uv sync`);
no additional dependencies, no audio, no server, no network. Nothing is
written; the same data root gives the same table on every machine.

**Exact command and output.**

```bash
uv run python experiments/leakage-audit/impact.py
uv run python experiments/leakage-audit/impact.py --data-dir <root>
```

Exit codes: 0 a table is printed; 1 a manifest is malformed (the set is named
on stderr); 2 the data root holds no prepared training set.

The table has one line per set: `clips`, the set's current manifest size; each
rule column, the clips that rule removes; `total`, the union — a clip several
rules remove counts once (the two NASA rules cover the same clips on a machine
where both `nasa-en-test` and `nasa-en-test-v2` are prepared). Clips of talks
that are no longer in the source's current pinned listing (the set was
prepared from an earlier one) carry no rule count; the report lists them under
the table, and the re-preparation reads the current listing, so they leave the
set on their own.

The **rebuild list** is the sets a re-preparation with the corrected fetchers
must re-run: every set where one of the evaluation rules removes a clip; every
sized (`-30k` / `-40k`) set where the within-set re-upload rule removes a
clip; and every set derived from a set of the list — the `-<k>k` larger
preparation of its source ([`base_dataset`](../../src/audiolab/data/datasets.py))
and `nasa-en-train-qwen`, the re-labeling of `nasa-en-train`
([`relabel`](../../src/audiolab/data/relabel.py)). A 12 000-clip set that only
loses within-set re-uploads is not rebuilt: its re-preparation for any other
reason drops them anyway. The two babble pools `mls-fr-train` and
`librispeech-en-train` are never rebuilt: their manifests are the committed
babble of the noisy evaluation sets ([`noisy`](../../src/audiolab/data/noisy.py)),
and a noisy preparation whose pool no longer gives the committed utterances is
refused.

**Limits.** The report measures the current manifests: it predicts what the
re-preparation removes, it does not run it. Its counts are as good as the
pinned listings and pins it reads — a re-pinned listing changes the duplicate
classes and the reading order, and the stale-listing section of the output
names what that leaves behind.

**Table over the current data root.** *(To be filled in when the last
evaluation set is prepared.)*
