# 0017 — Campaign 1 test freeze

- Scope: stt
- Derived from: D-005, D-009, [decision 0002](../accepted/0002-frozen-evaluation-and-sealed-test-set.md)
- Decided by: proposed by an agent, 2026-11-21; the campaign runs under it as a working assumption
- Frozen at `9843311`, the `origin/main` commit of 2026-11-21;
  `configs/suites/` and `configs/panels/` are unchanged since `ce46aba`, the
  commit the campaign's base runs were started from (`git diff ce46aba 9843311
  -- configs/suites configs/panels` is empty)

## Decision

[Decision 0002](../accepted/0002-frozen-evaluation-and-sealed-test-set.md)
requires a training campaign to fix its test sets and its scoring rules before
it starts. Campaign 1 — the decision-0004 baseline recipe, run on the mix of
[proposed decision 0016](0016-training-mix-v1.md) option 1 without the noise
clause ([0015](0015-noise-augmentation-in-the-baseline-recipe.md) option 1) and
scored under normalizer v2 ([0013](0013-normalizer-v2-contractions.md)), as
`configs/train/mix-v1.yaml` records — starts from this decision: the panels, the
scoring rules and the sealed set below are frozen for the whole campaign, and
the candidate rule in § 4 is written before any candidate number exists.

### 1. The test sets

The campaign scores two suites, `configs/suites/fast.yaml` and
`configs/suites/full.yaml`, as they stand at `9843311` — 18 panels each. The
tables give each panel and the sha256 of its frozen id list under
`configs/panels/`, which is the hash every campaign run records in its
`run.json`. No panel is added, removed or changed during the campaign: adding
panels to a suite needs the owner's agreement (backlog [DS-01](../../backlog.md#data-campaign)),
and an existing panel never changes.

`full` panels:

| panel | sha256 of `configs/panels/<panel>.txt` |
|---|---|
| `fleurs-fr-full` | `5776d4ce1843107f2598206c91dc92f39b63c0068a321cc0be785532a6866cb4` |
| `mls-fr-full` | `5556eb9bca4c81d2fdb90b428fe491a89f7e2a7342874bda63f547d9cd1c152f` |
| `fleurs-en-full` | `ad70e11bdef146c51e97885555ef5b10d38387a289a563153ad0ae2f6954f202` |
| `librispeech-en-full` | `05d4aad4c2a91c6ce4bb8cf250854bc39993082f7829f13047a124ee14f627d1` |
| `voxpopuli-en-nonnative-full` | `118864ac4084d3da24e766a068545e96118334b5708534654c2c26e84dea0584` |
| `servicenow-codeswitched-en-fr-full` | `8d8770519115f19da693dcabbbf2820592c31cc14f1795d9d4d72f91689edbb9` |
| `servicenow-codeswitched-en-fr-ca-full` | `b12d0daafd098279fc74b8a48780fdbcb890e3c7383eddbc357aa6b6ba563695` |
| `cs-fleurs-fr-en-full` | `630741842543cee995c1b35b427f95d5f705ffba0378c3c1b60f75b854641e5b` |
| `nonspeech-full` | `ce2071ce3920b4ee3527f72b5c9cab346c8009e532473da65645479b3d4eb59c` |
| `nonspeech-music-full` | `318da2acaaa8edfadf686054de40c51e2aa2987ef2170f9f532555dc7a1b5e30` |
| `african-accented-fr-test-full` | `81e07f805488560e1f533a05a95e16518fdd86ff8c8dcc42997c830221250329` |
| `notsofar-en-eval-full` | `e341ab9b1e7114cf722d220aea2ef53c944651f8e371453b7f79885ae9c133b4` |
| `noisy-fleurs-fr-full` | `07614d3c12da2773ca77d3e2c4ccec6e6afcdd977a37ca4f9b3190d32eef7d18` |
| `noisy-fleurs-en-full` | `f2a95c58d11222da4a79ae37740301e89121aee36b2420d8a301e2be3eb60b99` |
| `noisy-servicenow-codeswitched-en-fr-full` | `20c4554211de381c4e590ea8028e4307df8f65adba7fd3d3b6dad701e84066d1` |
| `nasa-en-test-full` | `36a2cb1ddd40b4e702c187414a64d4681fd6dc57d3b17cd952b8ef94bf1c018d` |
| `peertube-edu-captions-fr-full` | `3862413b611a4294ba7d68f4c3f325a2812cd9a78ad9b7cc387ea343cd897069` |
| `peertube-edu-captions-fr-v2-full` | `a2fe1d849d42bb9b5bf9d370fb6eeaa812bfaca3c626fd4966ec4633ff2407e3` |

`fast` panels:

| panel | sha256 of `configs/panels/<panel>.txt` |
|---|---|
| `fleurs-fr-fast` | `c9ed56b6dc20f1054d3d71482e6c0ec923c7fd5c8db221031e341228b6291276` |
| `mls-fr-fast` | `5953577da19aaeffbb5329f3926da4cfd66453c42cd6cdacaa5d58ca76e6dc08` |
| `fleurs-en-fast` | `1c85c0cf4ebd089eebb3e66323b6c0747588cf76fffcf707a9c1bab613ed7df8` |
| `librispeech-en-fast` | `accec31d2dec07c325d4ac69a4a5cfb39d76810b2e96b7e9c610360ae23f11c8` |
| `voxpopuli-en-nonnative-fast` | `acb0701609fc2ce48c0fcb66aaed4799da4ac37c7b5417880dd1085e0d125894` |
| `servicenow-codeswitched-en-fr-fast` | `ab471e21996618eeb574dd78b79903e44744a1aeaf608f3ae2d1213a45bd7eb0` |
| `servicenow-codeswitched-en-fr-ca-fast` | `eef152389f692ebe9707254f42bc4ed21f680cee28ad15f849b346be5edf22d7` |
| `cs-fleurs-fr-en-fast` | `75e673f1baf72092fe539ab9bc6e6b899e43ec35e26eaf93422b16ee5d6b9c68` |
| `nonspeech-fast` | `bd998d082bcb13fdbafe8c1a2092e2702674fbb0a074355225a6dc4e2b5cdf01` |
| `nonspeech-music-fast` | `403e91803798ab59a2b72431c9f9ba46f5a2c9e2d2d9257523c9f95f8991c2d4` |
| `african-accented-fr-test-fast` | `0a8abe1ab89b590228e06134369b630c0682b364f3b78b2f88a9a89a51159221` |
| `notsofar-en-eval-fast` | `09470409d86938886a84eb808636c68b3fdb2cd16ec370f916845a33bf92ba7e` |
| `noisy-fleurs-fr-fast` | `628c04a97900031c401e5d23c08d900872e5735b101185126bfadd98fda1b1ac` |
| `noisy-fleurs-en-fast` | `9a8ed2d9edc47bc7b3ebdef62da6cac741bc89125132b995ef2cc86629be19e5` |
| `noisy-servicenow-codeswitched-en-fr-fast` | `3c08bf678cd79ab001a961316207c00be13a16f400a824da463da2a33e09ae16` |
| `nasa-en-test-fast` | `185dcee27e23f36e7bd68d1390b652e85057b72c155f424078e9e585f76b85d7` |
| `peertube-edu-captions-fr-fast` | `02a9147061137ec5c39de60a1892f8247962dd76df7d48aa515292a812d23557` |
| `peertube-edu-captions-fr-v2-fast` | `3890a1a4c69cd9f1be72e5a4775fda22bb17279f7115958ecd49ae07c561eabc` |

Both suites are scored for every campaign run. The `full` panels decide the
numbers the campaign acts on (§ 4); the `fast` panels are the cheap view taken
at the same time and reported alongside, because their 100 clips per panel give
paired intervals too wide to separate the differences this selection is about
(`configs/suites/full.yaml` calls itself the place where a small difference
becomes visible).

The base model's two runs of 2026-11-21 (§ 2) scored exactly these
files: their `run.json` panel hashes equal the hashes above, panel for panel,
and the panel sets are the same 18 (`full`: 18 panels, 8 312 items, 0 failed;
`fast`: 18 panels, 1 690 items, 0 failed). `audiolab compare` refuses two runs
whose recorded panel hashes differ (`compare_runs` in
`src/audiolab/eval/report.py`), so only runs scored on these panels compare at
all.

### 2. The scoring rules

Every campaign number is scored under:

1. **Normalizer version 2**, the contractions, context-`'s` and elision-spacing
   rules of [proposed decision 0013](0013-normalizer-v2-contractions.md),
   implemented in `e2aefe1` (`src/audiolab/eval/normalize.py`, version 2, the
   default) and unchanged at `9843311`. That commit is the one
   [0013](0013-normalizer-v2-contractions.md) refers to. Version 1 stays
   reproducible (`audiolab score --normalizer 1`, `audiolab compare
   --normalizer 1`) and no published v1 number is rewritten.
2. **WER**, the corpus word error rate of `src/audiolab/eval/metrics.py`, in the
   `normalized` view the comparison reports (`wer_normalized` in its JSON,
   [`docs/stt/evaluation.md` § 6](../../stt/evaluation.md#6-what-each-number-means)).
3. **The paired confidence intervals of `audiolab compare`** as implemented at
   `9843311`: a 95 % percentile bootstrap that resamples whole groups (a
   speaker, a talk, a sentence), not single utterances, and draws both runs
   together so the difference `candidate − base` is paired
   (`src/audiolab/eval/stats.py`). A panel whose interval stays away from zero
   is reported as separable; the comparison pairs only the clips both runs
   answered and lists the ids it left out.

The base model's runs of 2026-11-21 were written under version 1 (`full`) and
version 2 (`fast`); the comparison re-scores the stored records under the
version it is asked for, so `audiolab compare --normalizer 2` compares both
under version 2 without touching either directory (`_paired_items` in
`src/audiolab/eval/report.py`). Where a campaign document writes the base's
per-panel numbers out under version 2 instead, they come from a `cp -a` copy
re-scored with `audiolab score --normalizer 2`, the published directory
untouched — the recommended practice of
[`docs/stt/evaluation.md` § 5](../../stt/evaluation.md#5-text-normalization-three-views-two-versions).

### 3. The sealed set

The campaign's sealed set is the sealed counterpart of every dataset of
`full.yaml`, file by file. The list was verified against the prepared data
under `$AUDIOLAB_DATA` on 2026-11-21: every id in every file is present in its
dataset's manifest, no id is repeated, and each file carries the count below.
The files are committed at `9843311`, so their content is pinned by that
commit.

| sealed file | dataset | clips |
|---|---|---|
| `african-accented-fr-test-sealed.txt` | `african-accented-fr-test` | 119 |
| `cs-fleurs-fr-en-sealed.txt` | `cs-fleurs-fr-en` | 207 |
| `fleurs-en-sealed.txt` | `fleurs-en` | 132 |
| `fleurs-fr-sealed.txt` | `fleurs-fr` | 143 |
| `librispeech-en-sealed.txt` | `librispeech-en` | 492 |
| `mls-fr-sealed.txt` | `mls-fr` | 468 |
| `nasa-en-test-v2-sealed.txt` | `nasa-en-test-v2` | 294 |
| `noisy-fleurs-en-sealed.txt` | `noisy-fleurs-en` | 132 |
| `noisy-fleurs-fr-sealed.txt` | `noisy-fleurs-fr` | 143 |
| `noisy-servicenow-codeswitched-en-fr-sealed.txt` | `noisy-servicenow-codeswitched-en-fr` | 60 |
| `nonspeech-sealed.txt` | `nonspeech` | 20 |
| `nonspeech-music-sealed.txt` | `nonspeech-music` | 147 |
| `notsofar-en-eval-sealed.txt` | `notsofar-en-eval` | 4 223 |
| `peertube-edu-captions-fr-sealed.txt` | `peertube-edu-captions-fr` | 920 |
| `peertube-edu-captions-fr-v2-sealed.txt` | `peertube-edu-captions-fr-v2` | 934 |
| `servicenow-codeswitched-en-fr-sealed.txt` | `servicenow-codeswitched-en-fr` | 60 |
| `servicenow-codeswitched-en-fr-ca-sealed.txt` | `servicenow-codeswitched-en-fr-ca` | 38 |
| `voxpopuli-en-nonnative-sealed.txt` | `voxpopuli-en-nonnative` | 50 |
| **total** | | **8 582** |

`nasa-en-test-sealed.txt` is **not** in the set. Its panel scores 26 clips of
the episode NASA re-released under another slug, and the re-release sat in
every `nasa-en-train` preparation made before the duplicate-class rule — a
clip a model is scored on may not be trained on, so future sealed scoring uses
`nasa-en-test-v2-sealed`, the same set without that episode
([its card](../../stt/datasets.md#nasa-en-test-v2--the-same-set-without-the-re-released-episode)).

`peertube-edu-captions-fr-sealed` is read under the owner's answer to
[O-12](../../backlog.md#owner-review-needed) (a) of 2026-11-16: the panel holds
920 clips in 23 talks, two of which are the sealed halves of the two re-upload
classes whose twins are scored in the full panel (40 clips each, 80 in total);
its primary number is pre-registered on the other **840 clips / 21 talks**, the
920 figure reported alongside.

The first and the decoded captions sets seal **the same talks**: the v2 sealed
panel holds the 21 talks the v1 one seals outside the two split classes, plus 3
talks new in v2 (24 talks, 934 clips), because v2 keeps the source's talks
sealed by construction ([the v2 card](../../stt/datasets.md#peertube-edu-captions-fr-v2--the-same-talks-with-the-caption-text-decoded));
measured on 2026-11-21, 831 of the v1 panel's 840 clips outside the split
classes have an audio `sha256` that is also in the v2 sealed panel (the decoded
preparation re-cut its clips, so the boundaries differ, the talks do not). The
sealed set is therefore read panel by panel, like every other score here: no
number is pooled across panels, and the two captions panels — which would count
those 21 talks twice — are never added together.

No agent scores the sealed set: no campaign run passes `--sealed`. Once the
campaign has named one candidate, the owner scores the sealed set **once**: one
run on that candidate and one on the base model, for the paired delta, reported
panel by panel under the rules of § 2.

### 4. The candidate selection rule

Written before any candidate number exists.

**Candidate set, fixed now:** `mix-v1` (`configs/train/mix-v1.yaml`) and its
noise arm `mix-v1-n40` — the same recipe with MUSAN noise on about 40 % of the
clips, whose config is committed before that run starts. No candidate is added
to this campaign (the seed replicate `mix-v1-s2` is not a candidate).

**Eligibility.** A candidate is eligible when all three hold:

1. it reached its configured `updates` with no unresolved anomaly — no crash
   left unrecovered, no NaN loss, no GPU or systemd-oomd OOM, no stall, and no
   refused resume;
2. no panel of `full.yaml` shows a **separable loss** against the base: for a
   speech panel, the candidate − base delta's paired interval lies wholly above
   zero **and** its point estimate is above +0.5 WER points; for `nonspeech`
   and `nonspeech-music`, the hallucination-rate delta's paired interval lies
   wholly above zero **and** its point estimate is above 0.71 points, the
   run-to-run spread measured between two runs of the same recipe
   ([screening.md](../../stt/screening.md#noise-augmentation));
3. its unweighted mean normalized WER over the seven primary panels is below
   the base's.

**Primary panels** (the target use of
[D-009](../../owner/directives.md): French first, code-switched dictation):
`servicenow-codeswitched-en-fr-full`, `servicenow-codeswitched-en-fr-ca-full`,
`cs-fleurs-fr-en-full`, `fleurs-fr-full`, `mls-fr-full`,
`peertube-edu-captions-fr-v2-full`, `african-accented-fr-test-full`. **Secondary
panels:** the two nonspeech panels, `nonspeech-full` and
`nonspeech-music-full`. English and accented English are not primary, because
D-009 makes French the priority and keeps English where it is simple; they are
covered by the eligibility guard, which reads every panel of `full.yaml`.

**Comparison.** Each candidate is compared with the base by `audiolab compare
--normalizer 2 BASE CANDIDATE` under the rules of § 2, `--json` for the
per-panel numbers. The eligibility guard reads all 18 `full` panels; the
ranking reads the seven primary ones. `full` decides; the `fast` comparison is
reported alongside and changes nothing.

**Selection.** Among the eligible candidates, the one with the lowest
unweighted mean normalized WER over the seven primary panels is the campaign's
candidate. The paired interval of every primary panel is reported with the
choice; where no primary panel separates a candidate from the base, the lowest
mean is still taken, and the decision record says in words that the difference
is within the paired intervals rather than separable. A candidate that improves
the primary mean but regresses on either secondary panel by more than that
panel's paired interval must have the regression explained in the decision
record before it can be chosen.

**Ties and no eligible candidate.** If the two candidates are compared
directly (`audiolab compare mix-v1 mix-v1-n40`) and no primary panel separates
them, they are tied and the simpler recipe, `mix-v1`, is the campaign's
candidate; the record says the two are not separable. If no candidate is
eligible, the campaign names none — that is a result, recorded as such — and
the thresholds are never relaxed once the numbers exist. The number of
candidates compared is reported with the result.

The candidate is chosen before the sealed set is opened
([decision 0002](../accepted/0002-frozen-evaluation-and-sealed-test-set.md)):
the sealed set confirms the choice, it never makes it.

**Applied once.** The rule is applied once, at the end of the campaign's first
selection round, when both candidates have been scored on the frozen suites.
Changing it after any candidate number exists requires the owner.

### 5. What this decision does not decide

Publication of any model (D-006, the owner's explicit request, after the owner
has tested it); the final acceptance of [0013](0013-normalizer-v2-contractions.md),
[0015](0015-noise-augmentation-in-the-baseline-recipe.md) and
[0016](0016-training-mix-v1.md), which stay proposed and which the campaign's
runs treat as validated for testing (as `configs/train/mix-v1.yaml` records).

## Why

[Decision 0002](../accepted/0002-frozen-evaluation-and-sealed-test-set.md)
exists because a test set reused to compare many checkpoints ends up selecting
the winner for that test set. Campaign 1 trains candidates of about 30 000
updates each — tens of hours of one GPU — and scores them repeatedly while they
run, so its panels, its scoring rules and its candidate rule are fixed here,
before the first candidate exists, and its sealed set stays unopened until one
candidate is named. The sealed set is the honest final number; the frozen
panels are what the campaign may look at while it works.

## Alternatives considered

- **Letting the `fast` suite decide the selection.** It is cheaper and was
  designed for iteration, but its panels hold 100 clips each (30 for
  `nonspeech-music`) — an order of magnitude below the `full` panels the
  comparison is about — so its paired intervals are widest exactly where the
  ranking has to separate two close candidates. The `fast` numbers are reported
  with every choice; they do not decide it.
- **Using the sealed set to break a tie between two close candidates.** This
  was proposed and is not taken: decision 0002 fixes the candidate before the
  sealed set is opened, precisely so that the final number is not the best of
  several measurements. A tie is resolved by the simpler recipe, in the record.
- **Keeping `nasa-en-test-sealed`.** Rejected: 26 of its clips are the
  re-released episode whose duplicate was in every pre-rule NASA training
  preparation, so scoring it would score training speech. `nasa-en-test-v2`
  exists for exactly this and its sealed panel is the same set minus that
  episode (294 clips of 11 of the 12 groups).
- **Freezing the panels but not the scoring rules** (or keeping normalizer
  v1). Rejected: a change of normalizer changes scores, and decision 0002 fixes
  the scoring rules with the test sets. The campaign is scored under version 2
  as decided for it; v1 stays reproducible for the published numbers.
- **Adding panels to a suite during the campaign** (for example a panel that
  turns out to be missing). Rejected: DS-01 gives that to the owner, and
  `audiolab compare` refuses runs scored on different panel hashes, so a
  changed suite would split the campaign's numbers into incomparable sets.

## Checks made before writing (2026-11-21)

- **The base runs scored exactly these panels.** For each suite, the sha256 of
  every committed `configs/panels/<panel>.txt` equals the hash recorded for
  that panel in `$AUDIOLAB_RUNS/eval/parakeet-tdt-0.6b-v3-nemo/<suite>/run.json`,
  and the two panel sets are identical; the records hold 8 312 items (0 failed)
  for `full` and 1 690 (0 failed) for `fast`. `git diff ce46aba 9843311 --
  configs/suites configs/panels` is empty, so the suite files did not change
  between that commit and this freeze.
- **The sealed list.** For each of the 18 files above, every id below its
  header is present in that dataset's `manifest.jsonl` under `$AUDIOLAB_DATA`,
  with no repeated id (0 absent, 0 repeated).
- **O-12 (a).** The two split-class talks hold 40 clips each in
  `peertube-edu-captions-fr-sealed.txt` (80 of its 920), which leaves 840 clips
  in 21 talks; the same two talks hold 7 and 6 clips in
  `peertube-edu-captions-fr-full.txt` (2 and 1 of them also in `fast`), the
  straddle O-12 records.
- **The captions sealed panels are not talk-disjoint.** 21 of the v1 panel's
  23 talks are also in the v2 sealed panel, and 831 of the 840 v1 clips outside
  the split classes match a v2 sealed clip by audio `sha256`; the v2 panel adds
  3 talks new in the decoded set. Hence § 3's panel-by-panel reading.
