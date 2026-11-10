# Backlog

Updated: 2026-11-10.

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
| P-01 | Re-screen NASA after its training labels were cleaned. | The [NASA screening analysis](stt/screening.md#what-each-source-does-alone) describes the text defect; the [dataset card](stt/datasets.md#nasa-en-train--nasas-podcasts-labelled-from-nasas-own-transcripts) describes the prepared replacement. Do not delete the earlier preparation before the comparison is checked. |
| P-02 | Label more of the talk pools and test a training mix. | Mix support and the leakage guard are implemented in the [training interface](stt/training.md#mixing-datasets); choosing the mix and the labelling budget still need the owner. Granary as an anchor is a [hypothesis](stt/screening.md#what-the-base-model-already-saw), not a selected recipe. |
| P-03 | Separate the value of talk audio from Qwen's label style. | Design a controlled comparison that changes only the labels of the same human-transcribed audio, for example LibriSpeech-PC or ICSI. The [screening hypothesis](stt/screening.md#what-each-source-does-alone) is not a result of this test. |
| P-04 | Review contraction and apostrophe-spacing normalization. | Diagnostic done (2026-11-10, all 16 screening runs, official paired intervals): [experiments/normalization-review](../experiments/normalization-review/README.md). One screening conclusion moves — NASA's harm on servicenow en-fr becomes not separable under the diagnostic; proposed decision 0013 awaits the owner's call on (whether contraction and elision equivalence becomes the official scoring rule of the next campaign; answered "probably yes" on 2026-11-10, **deferred by the owner the same day — decision open for the next campaign**; 0013 remains proposed). Do not silently rewrite a campaign's scoring rules or historical results. |
| P-05 | Review the NASA source filters. | The owner decided on 2026-11-10 to keep the episode-level music and archive filters and to leave NASA's Curious Universe out entirely; this is not implemented yet. Still open: whether the edge-word gate's selection bias is acceptable. The [NASA card](stt/datasets.md#nasa-en-train--nasas-podcasts-labelled-from-nasas-own-transcripts) records the current filters and counts. |
| P-06 | Re-prepare talk sets made before label-character cleanup. | Review the [training-source text rules](stt/datasets.md#training-sources) before a full training run. The existing screens do not need to be repeated solely to organize this repository. |

## Data campaign

Before any training campaign, the evaluation sets are fixed
([decision 0002](decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)),
so DS-01 comes before DS-03.

| ID | Work | Reference / limits |
|---|---|---|
| DS-01 | New evaluation sets for what the panels do not measure: other French accents, real meetings, noise and reverberation, music, technical talks with human references. | [Evaluation limits](stt/evaluation.md). Adding panels to the suites needs the owner's agreement; existing panels never change. Covers N-03 and part of N-04 and N-05. |
| DS-02 | Noise and reverberation augmentation in training, and a controlled test of it. | [Decision 0004](decisions/accepted/0004-parakeet-fr-baseline-recipe.md) plans noise on about 40 % of samples; [training](stt/training.md) has none today. The training half of the noise must never be the half used by evaluation. |
| DS-03 | Larger training preparations than 12 000 clips, and French technical talks as their own source. | The 12 000-clip sets stay as the [screening](stt/screening.md) used them. Needs DS-01's evaluation talks excluded first. Part of P-02. |

## Owner review needed

| ID | Question | Reference |
|---|---|---|
| O-01 | Review the STT evaluation-core proposal. | [Proposed decision 0010](decisions/proposed/0010-stt-evaluation-core.md). |
| O-03 | Review a locked, optional heavy-model stack and the hygiene-check false positive that blocks it. | [Proposed decision 0012](decisions/proposed/0012-heavy-model-stack-as-a-locked-extra.md). Neither dependency packaging nor CI exceptions are approved by this backlog. |
| O-04 | Add Nemotron to the evaluated models? | A [model config](../configs/models/nemotron-3.5-transcribe-vllm.yaml) exists. Confirm access and a suitable evaluation budget before using it. |
| O-05 | Review the AMI per-group training cap. | Check the current [training-source preparation](stt/datasets.md#training-sources) and AMI card before proposing a different sampling rule. |
| O-06 | Pursue the remaining weak source candidates? | VOA Afrique and the European Commission Audiovisual Service remain documented in the [source register](stt/sources.md); their reasons and license constraints live there. |

## Not started / deferred

| ID | Work | Reference / limits |
|---|---|---|
| N-01 | Train and evaluate the full baseline. | [Decision 0004](decisions/accepted/0004-parakeet-fr-baseline-recipe.md) defines the baseline; the [roadmap](roadmap.md) tracks its completion. Short screening runs are not that baseline. |
| N-02 | Consider a general multi-run `audiolab table` command. | Deferred. The [source-screening helper](../experiments/source-screening/README.md) is a narrowly scoped reporting script, not the implementation or approval of a general core command. |
| N-03 | Add music to non-speech evaluation coverage. | The [evaluation limits](stt/evaluation.md) explain the current gap. Select a permissible source and review the frozen-panel protocol before adding coverage. |
| N-04 | Review further English evaluation coverage. | Start from the existing [evaluation coverage](stt/evaluation.md), including English and non-native-English panels; identify what remains missing rather than assuming FLEURS is the only English set. |
| N-05 | Investigate train/test source overlap with speaker and session exclusion. | Compare concrete corpus cases, such as MLS and VoxPopuli, against the [leakage rules](stt/datasets.md). This is a question to test, not permission to weaken the audit. |
| N-06 | Explore GGUF export. | Deferred; no exporter, target runtime or acceptance criterion is selected. |
| N-07 | Begin a TTS project. | TTS is within the repository's scope (D-003); its concrete project and protocol remain to be defined. It is not excluded by the STT-first roadmap. |
| N-08 | Review and select a low-bit audio pilot. | [QAT/QAD feasibility study](research/low-bit-training.md): literature reviewed; no implementation or run. Before execution, select the checkpoint, deployment GPU/runtime, quality margins and budget; prove the export path, then compare INT8 and mostly-4-bit PTQ before considering QAT/QAD. This does not resume paused training or select a TTS project. |
