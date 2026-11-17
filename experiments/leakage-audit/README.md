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

Over the re-prepared data, runs of 2026-11-17 — the same command with
`--train` pointed at each set:

```text
  voxpopuli-fr-train: 12000 clips, 2136 sessions
shared sessions: 0 of 307

  voxpopuli-fr-train-40k: 24259 clips, 2280 sessions
shared sessions: 0 of 307
```

The positive control held: the same command over the set of 2026-11-08, kept
as `voxpopuli-fr-train.pre-leakfix-2026-11-15` until it was deleted on
2026-11-17 at the owner's request, gave 206 of 307 shared sessions and 1 956
of its 12 000 clips, and over `voxpopuli-fr-train-40k.pre-leakfix-2026-11-15`,
deleted the same day, gave 219 of 307 and 4 815 clips.

### Limits

The audit is over prepared manifests: re-preparing either set changes the
figures, and the corpora hold far more sittings than the prepared sample. The
script asserts that every clip id holds a session before its language tag and
fails loudly otherwise; it does not re-derive the sessions from the corpora's
own metadata.

## Content-level leakage (6-word text shingles)

**Status:** 2026-11-15, and a re-run of 2026-11-17 over the re-prepared data.
The confirmed duplicate pairs are committed in
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

**Population.** Prepared sets only: the prepared evaluation sets with panel
files in `configs/panels/` (the two non-speech sets excluded — 15 in the run
of 2026-11-15, 17 in the re-run of 2026-11-17) and the 25 prepared training
directories (names containing `-train`, with the `.pre-*` and `.curly-*`
renames excluded). The audit reads manifests only —
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

**Results, 2026-11-17, over the re-prepared data.** 659 evaluation-vs-training
pairs were reported, and none is a leak: **0 pair holds the same recording** —
the eight leak pairs of the run of 2026-11-15 (the three PeerTube re-uploads,
the three pairs of the re-released interview, the two compilation replays) left
the training sets with the re-preparations of 2026-11-16 and 2026-11-17 — and
**0 of the 659 pairs shares an audio `sha256`**. The 659 split into 377 of the
same text read by another corpus speaker (374 of the SLR57 short read-task
sentences, 3 of the one LibriSpeech chapter pair of the table above) and 282 of
the recurring show intro of the table above, in the 5–30 % band (141 per NASA
test set, all triaged as the shows' scripted stock openings, no conversation
text shared). Population: 17 prepared evaluation sets and 25 prepared training
sets.

The same re-run's training-vs-training part reports **2 129 talk-group pairs
across two sets** at ratio ≥ 30 % (mostly by design: the 12k ⊂ 40k/30k
supersets, the same episodes under the 12k / 30k / qwen label sets, talks
shared across the three PeerTube listings), **22 pairs inside one set**, and a
re-upload shortlist of 37 rows at ratio ≥ 75 % with different group ids, 18
distinct id-pairs: the 16 granary id-pairs of the run of 2026-11-15 — the same
videos under a second id; granary is not a talk source: no duplicate-class
rule covers its videos, so the set keeps them (a repetition inside one set) —
the ministry talk re-uploaded in the edu and
tech sets (`b01320a7…` ↔ `c18d134b…`, 99.5 %, the 12k and 40k edu sets against
`peertube-tech-fr-train-40k`), and the Capitole re-upload pair
(`44325ad5…` ↔ `66095949…`, 78.8–79.6 %, the 12k set against itself and against
the 40k and tech sets). The two PeerTube pairs are one duplicate class each
over the pinned union listing (both id-pairs are committed in
[`duplicates.json`](../../src/audiolab/data/duplicates.json)); the 2026-11-15
NASA shortlist pair is gone, the `introducing` episode being in none of the
rebuilt NASA training sets (the run of 2026-11-15, above).

**Limits.** The audit covers prepared sets only: `nasa-en-train-30k` holds
570 episodes while 138 of the listing's 708 non-test episodes (the 768
listing minus the 60 reserved test episodes) hold in none of the three
prepared NASA training sets and were never prepared, and the 2017 end-of-year
compilation is only about 30 % explained by the two replayed episodes; the
re-run of 2026-11-17 above confirms that no `nasa-en-test` group has a
≥ 5 % pair other than the stock-intro pattern, and no pair of any evaluation
set shares a recording or an audio `sha256` with a training set. The
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
sets that forces to be re-prepared. The table over the current data root,
printed below, is the run of 2026-11-17, taken once every set of the run of
2026-11-16's rebuild list had been re-prepared (the rebuild list below it is
now empty); the script is safe to re-run any number of times, and a re-run
over a changed data root gives the new table.

**Question.** For each prepared training set, how many clips does each
exclusion rule remove, and which sets must a re-preparation re-run because of
it?

**Method.** The rules are never re-implemented here: every table column is the
committed function the fetcher itself uses, applied to the set's current
`manifest.jsonl` — the one exception being the `eval class` column on the NASA
sets, which reads the prepared test sets' groups while `fetch_nasa_en_train`
itself reads only the reserved list; those groups are the reserved episodes
(60, the v2's a subset of them), so the column can never add a clip. The
one-to-one mapping (a reviewer can check every number against the fetcher's
code):

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
prepared from an earlier one) carry no rule count; the report lists them, with
their clip counts, under the table, and when the set is re-prepared the
preparation reads the current listing, so those clips stay out of it on their
own.

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

**Table over the current data root.** Run of 2026-11-17, over the current data
root, once every set of the run of 2026-11-16's rebuild list had been
re-prepared.

Evaluation sets found prepared (the evaluation-class rule reads their
manifests):

- peertube-edu-captions-fr (evaluation set of peertube-edu-fr-train): prepared, 113 groups
- peertube-edu-captions-fr-v2 (evaluation set of peertube-edu-fr-train): prepared, 126 groups
- peertube-edu-captions-fr (evaluation set of peertube-tech-fr-train): prepared, 113 groups
- peertube-edu-captions-fr-v2 (evaluation set of peertube-tech-fr-train): prepared, 126 groups
- nasa-en-test (evaluation set of nasa-en-train): prepared, 60 groups
- nasa-en-test-v2 (evaluation set of nasa-en-train): prepared, 59 groups
- african-accented-fr-test (evaluation set of african-accented-fr-train): prepared, 12 groups

Committed pins: 307 voxpopuli evaluation sessions; 1024 babble utterance
entries (1012 unique) over 3 noisy sets; 60 nasa reserved test episodes.

| set | clips | eval class | in-set re-upload | nasa reserved | slr57 speaker | vox session | babble | total |
|---|---|---|---|---|---|---|---|---|
| african-accented-fr-train | 11424 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ami-fr-speakers-en-train | 3971 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| debconf-en-train | 12000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| debconf-en-train-30k | 30000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fosdem-en-train | 12000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fosdem-en-train-30k | 30000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| granary-fr-train | 12000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| granary-fr-train-40k | 37971 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| icsi-en-train | 8746 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| librispeech-en-train | 12000 | 0 | 0 | 0 | 0 | 0 | 488 | 488 |
| librispeech-pc-en-train | 12000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| librispeech-pc-en-train-40k | 25449 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mls-fr-train | 12000 | 0 | 0 | 0 | 0 | 0 | 524 | 524 |
| nasa-en-train | 12000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nasa-en-train-30k | 30000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nasa-en-train-qwen | 12000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| notsofar-en-train | 2330 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| peertube-edu-fr-train | 12000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| peertube-edu-fr-train-40k | 40000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| peertube-fr-train | 12000 | 0 | 154 | 0 | 0 | 0 | 0 | 154 |
| peertube-fr-train-40k | 40000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| peertube-tech-fr-train-40k | 40000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pxcorpus-fr-train | 1912 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| voxpopuli-fr-train | 12000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| voxpopuli-fr-train-40k | 24259 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| total | 458062 | 0 | 154 | 0 | 0 | 0 | 1012 | 1166 |

The `total` column is the union: a clip several rules remove (the two nasa
rules cover the same clips) counts once; the rule columns may count it twice.
The only set with a non-zero `total` outside the two committed babble pools is
`peertube-fr-train`: its 154 within-set re-uploads stay in the frozen
12 000-clip set (the rebuild-list rule above), so they are what the 1 166
union of the run consists of together with the babble pools.

Clips of talks that are no longer in the source's pinned listing (the set was
prepared from an earlier listing) carry no rule count; when the set is
re-prepared, the preparation reads the current listing, so these clips stay
out of it on their own:

- none: every talk and NASA set holds only clips of talks of its source's current pinned listing

Rebuild list — the sets a re-preparation with the corrected fetchers must
re-run:

- (none)

Not rebuilt:

- librispeech-en-train — babble pool: the noisy sets' committed babble
- mls-fr-train — babble pool: the noisy sets' committed babble
- peertube-fr-train — only in-set re-uploads in a 12 000-clip set
