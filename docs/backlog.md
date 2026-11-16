# Backlog

Updated: 2026-11-15.

The durable list of open work. The [roadmap](roadmap.md) holds milestones and
completion criteria; this page holds concrete tasks and unanswered questions.
Results, source statuses and architectural rules stay in their own documents,
linked below. Remove completed items once their outcome has a durable home.

An entry is not a directive, an accepted decision, a priority order or permission
to start a long job. Unless stated otherwise, approaches below are proposals.
The [owner directives](owner/directives.md) and
[accepted decisions](decisions/README.md) still govern the work. In particular,
record a data-source status change in the [source register](stt/sources.md)
before acquiring or rejecting it (D-018).

## Resumed

The owner resumed these items on 2026-11-10. The priority is the training and
evaluation data (DS-01 to DS-03 below), not a first full training run. Each GPU run
and each long labelling job on the STT server still needs the owner's go.

| ID | Work or question | Reference / next step before execution |
|---|---|---|
| P-02 | Label more of the talk pools and test a training mix. | The "label more" part is **done** (2026-11-14): nine larger preparations, 300 840 clips in total, each leaving the evaluation talks out — the [cards](stt/datasets.md) and the [register rows](stt/sources.md) hold the numbers. What remains is with the owner: choosing the mix and the labelling budget, after C-01 removes the evaluation duplicates found on 2026-11-15; an agent proposal of a mix v1 with the real numbers is awaiting the owner's answer. Mix support and the leakage guard are implemented in the [training interface](stt/training.md#mixing-datasets). Granary as an anchor is a [hypothesis](stt/screening.md#what-the-base-model-already-saw), not a selected recipe. |
| P-03 | Separate the value of talk audio from Qwen's label style. | **Done** (2026-11-14, plan 06): the controlled comparison — `nasa-en-train-qwen`, the same 12 k NASA audio with every clip's label Qwen's alone, screened against `screen-nasa-en-plain` on the identical 10 000-clip draw — shows **no separable difference on the code-switched panels**, so the pre-registered reading (b) holds: the label style does not explain the talk gain; where the gain is real, it is the audio's. Outside the pre-registered reading, the model labels are worse on the human-reference panels (nasa-en-test +1.46, peertube-edu-captions-fr +2.17) and better on notsofar-en-eval and voxpopuli-en-nonnative — reported to the owner as an observation. The full table and the pre-registered reading are in the [screening](stt/screening.md#label-style-same-audio). The optional Whisper arm (plan 06 Part C) was not started: it needs a Part B difference and the owner's go. |
| P-04 | Review contraction and apostrophe-spacing normalization. | Diagnostic done (2026-11-10, all 16 screening runs, official paired intervals): [experiments/normalization-review](../experiments/normalization-review/README.md). One screening conclusion moves — NASA's harm on servicenow en-fr becomes not separable under the diagnostic. The decision on whether contraction and elision equivalence becomes the official scoring rule of the next campaign is the owner's: he answered "probably yes" on 2026-11-10 and deferred it the same day, so [proposed decision 0013](decisions/proposed/0013-normalizer-v2-contractions.md) remains proposed and is open for the next campaign. Do not silently rewrite a campaign's scoring rules or historical results. |
| P-05 | Review the NASA source filters. | Done 2026-11-13: the owner's 2026-11-10 decision is implemented — NASA's Curious Universe is out of the source entirely (`EXCLUDED_SHOWS` in `src/audiolab/data/nasa.py`), the listing is re-pinned (768 episodes, 2026-11-13), the episode-level music and archive filters are unchanged. Still open: whether the edge-word gate's selection bias is acceptable; the plan keeps it for training and never for evaluation sets. The [NASA card](stt/datasets.md#nasa-en-train--nasas-podcasts-labelled-from-nasas-own-transcripts) records the current filters and counts. |

## Data campaign

Before any training campaign, the evaluation sets are fixed
([decision 0002](decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)),
so DS-01 comes before DS-03.

| ID | Work | Reference / limits |
|---|---|---|
| DS-01 | New evaluation sets for what the panels do not measure: other French accents, real meetings, noise and reverberation, music, technical talks with human references. | [Evaluation limits](stt/evaluation.md). Adding panels to the suites needs the owner's agreement; existing panels never change. Covers N-03 and part of N-04 and N-05. |
| DS-02 | Noise and reverberation augmentation in training, and a controlled test of it. | **Done** (2026-11-15): the augmentation is in [training](stt/training.md#noise-and-reverberation) (noise and reverb, built only from the train half of a bank — the eval half is what the noisy panels score), and the controlled test is in [screening.md, "Noise augmentation"](stt/screening.md#noise-augmentation): decision 0004's 40 % dose was measured at screening scale against a matched control (the original control was not trained with the candidates' clip order, micro-batch, code or card), both arms are **rejected** by the rule pre-committed before the numbers, and the rule's stronger-dose (n80) run was not selected. Open: the owner's review of the proposal (O-09). |
| DS-03 | Larger training preparations than 12 000 clips, and French technical talks as their own source. | **Done** (2026-11-14): the nine larger preparations (each in its own `-40k`/`-30k` directory, [cards](stt/datasets.md), [register rows](stt/sources.md)) and [the tech talks as their own source](stt/datasets.md#peertube-tech-fr-train--the-tech-talks-of-the-two-french-peertube-listings); DS-01's evaluation talks are excluded **by id only**, which misses re-uploads: three `peertube-edu-captions-fr` talks re-uploaded under other ids are in `peertube-tech-fr-train-40k` (`cf513c6d…`, 200 clips) and `peertube-edu-fr-train-40k` (`73b9957a…`, 31; `d8fdaf19…`, 172) — same title, author and duration as `a2c7bd63…`, `20f11958…`, `bc3ba643…`. Do not train on these two sets until they are rebuilt (C-01). The 12 000-clip sets stay as the [screening](stt/screening.md) used them. The mix choice and the labelling budget remain with the owner (P-02). |

## Owner review needed

| ID | Question | Reference |
|---|---|---|
| O-01 | Review the STT evaluation-core proposal. | [Proposed decision 0010](decisions/proposed/0010-stt-evaluation-core.md). |
| O-03 | Review a locked, optional heavy-model stack and the hygiene-check false positive that blocks it. | [Proposed decision 0012](decisions/proposed/0012-heavy-model-stack-as-a-locked-extra.md). Neither dependency packaging nor CI exceptions are approved by this backlog. |
| O-04 | Add Nemotron to the evaluated models? | A [model config](../configs/models/nemotron-3.5-transcribe-vllm.yaml) exists. Confirm access and a suitable evaluation budget before using it. |
| O-05 | Review the AMI per-group training cap. | Check the current [training-source preparation](stt/datasets.md#training-sources) and AMI card before proposing a different sampling rule. |
| O-06 | Pursue the remaining weak source candidates? | VOA Afrique and the European Commission Audiovisual Service remain documented in the [source register](stt/sources.md); their reasons and license constraints live there. |
| O-07 | Mix guard for training: refuse clips whose text holds a character `unwritable()` reports unless the recipe sets `allow_unwritable_text: true`. | Undecided (owner, 2026-11-11); the rule the guard would enforce is in the [training-source rules](stt/datasets.md#training-sources) and the mix support in [mixing datasets](stt/training.md#mixing-datasets). |
| O-09 | Review the noise-augmentation proposal: the measured 40 % dose of decision 0004 is not supported at screening scale (both arms rejected under the matched-control rule), and the question is whether the baseline recipe keeps or drops the noise clause. | [Proposed decision 0015](decisions/proposed/0015-noise-augmentation-in-the-baseline-recipe.md) (rewritten from the matched-control evidence); the test and the numbers are in [screening.md, "Noise augmentation"](stt/screening.md#noise-augmentation). |
| O-10 | Run the stronger noise dose the inconclusive branch names — 80 % of the clips, `noise_prob: 0.8`, same banks and range — or stop at 40 %? | **Answered 2026-11-15:** the matched control was trained and the rule applied — both arms reject, so the rule's n80 clause did not fire and n80 was not run. The result is in [screening.md, "Noise augmentation"](stt/screening.md#noise-augmentation); the follow-up options are in [proposed decision 0015](decisions/proposed/0015-noise-augmentation-in-the-baseline-recipe.md). |
| O-11 | Label quality before the mix: in the [label-style test](stt/screening.md#label-style-same-audio), model labels cost +1.46 points on `nasa-en-test` and +2.17 on `peertube-edu-captions-fr` against human labels. Tighten the agreement threshold, raise the human-labelled share, or accept? | The owner decides it with the mix (P-02). Wording of the test's readings is corrected in C-03. |
| O-12 | The `peertube-edu-captions-fr` sealed panel's two re-uploaded talks also score in the full (and fast) panel — 13 clips of sealed content are in the full panel (3 of them also in fast), and every run scored on the full panel has scored them, the reference runs and the screening and noise runs among them; the set is frozen. Owner's call: (a) how those scores should be weighted; (b) for the decoded set's panels, whether a talk whose duplicate class touches a first-set-sealed talk goes to the decoded set's sealed panel (over-sealing — the safe direction of the panel rule), or whether the rule stays talk-by-talk as written. | Recommendation pending. The straddling fact is recorded, not repaired, in the [dataset card](stt/datasets.md#peertube-edu-captions-fr--the-education-ministrys-french-talks-with-human-captions). |

## Corrections (2026-11-15)

An independent review of the data campaign found the problems below. Each is
checkable from the repository and the prepared data; the work is in progress.

| ID | Work | Reference / limits |
|---|---|---|
| C-01 | Exclude evaluation duplicates from training, then rebuild the affected training sets, including the 12 000-clip screening sets (owner, 2026-11-15). | Duplicates found: the three PeerTube re-uploads (DS-03); `nasa-en-test`'s sealed episode `terry-fong-…-future-of-robotics` republished as `…-future-of-robotics-2`, which is in every NASA training set; the `2017-end-of-year-podcast` test episode replays a training episode; 447 LibriSpeech utterances mixed as babble into the noisy panels are in `librispeech-pc-en-train*` — the training guard refusing any new training run (mix or single dataset) holding an utterance of `babble_utterances.json` is part of this work; SLR57 test speaker `cameroon_m_012` likely equals train speaker `012` (the overlap check compared two id formats and never matched); the VoxPopuli shared sessions ([decision 0014](decisions/accepted/0014-voxpopuli-shared-sessions.md)). New evaluation datasets: `peertube-edu-captions-fr-v2` (an HTML-entity bug, `&nbsp;`, removed 13 of 17 talks from v1) and `nasa-en-test-v2` (sealed panel without the republished episode); the v1 panels stay. Mix v1 (P-02) waits for this. |
| C-03 | Documentation and small-code corrections: wrong readings (`notsofar-en-eval` "empty answers" are fillers; the label error figures are a likely lower bound), non-reproducible analyses (`experiments/nasa-filters`), stale gap lists (`src/audiolab/eval/report.py`), cache keys that miss an input (D-017: the noise-bank manifests in the resume check; the request fields in the answer caches), `audiolab data fetch <name>-<k>k --force` preparing 12 000 clips into the sized directory. | — |

## Not started / deferred

| ID | Work | Reference / limits |
|---|---|---|
| N-01 | Train and evaluate the full baseline. | [Decision 0004](decisions/accepted/0004-parakeet-fr-baseline-recipe.md) defines the baseline; the [roadmap](roadmap.md) tracks its completion. Short screening runs are not that baseline. |
| N-02 | Consider a general multi-run `audiolab table` command. | Deferred. The [source-screening helper](../experiments/source-screening/README.md) is a narrowly scoped reporting script, not the implementation or approval of a general core command. |
| N-04 | Review further English evaluation coverage. | Mostly done by plan 01: real meetings (`notsofar-en-eval`, 2026-11-13) and conversational technical English (`nasa-en-test`, 2026-11-13) join the read-speech, non-native-English and code-switched panels. What remains unmeasured is dictation itself (a person thinking out loud — personal audio, not permitted) and long-form audio; both are named in [evaluation.md §8](stt/evaluation.md#8-what-this-does-not-measure). Revisit only with a permissible source for those two. |
| N-06 | Explore GGUF export. | Deferred; no exporter, target runtime or acceptance criterion is selected. |
| N-07 | Begin a TTS project. | TTS is within the repository's scope (D-003); its concrete project and protocol remain to be defined. It is not excluded by the STT-first roadmap. |
| N-08 | Review and select a low-bit audio pilot. | [QAT/QAD feasibility study](research/low-bit-training.md): literature reviewed; no implementation or run. Before execution, select the checkpoint, deployment GPU/runtime, quality margins and budget; prove the export path, then compare INT8 and mostly-4-bit PTQ before considering QAT/QAD. This does not resume paused training or select a TTS project. |
