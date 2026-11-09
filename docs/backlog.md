# Backlog

Updated: 2026-11-09.

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

## Paused

These items remain paused by the owner. Organizing the repository does not
resume them.

| ID | Work or question | Reference / next step before execution |
|---|---|---|
| P-01 | Re-screen NASA after its training labels were cleaned. | The [NASA screening analysis](stt/screening.md#what-each-source-does-alone) describes the text defect; the [dataset card](stt/datasets.md#nasa-en-train--nasas-podcasts-labelled-from-nasas-own-transcripts) describes the prepared replacement. Do not delete the earlier preparation before the comparison is checked. |
| P-02 | Label more of the talk pools and test a training mix. | Choose the mix and budget first. The [training interface](stt/training.md) currently takes one dataset; mix support is not implemented. Granary as an anchor is a [hypothesis](stt/screening.md#what-the-base-model-already-saw), not a selected recipe. |
| P-03 | Separate the value of talk audio from Qwen's label style. | Design a controlled comparison that changes only the labels of the same human-transcribed audio, for example LibriSpeech-PC or ICSI. The [screening hypothesis](stt/screening.md#what-each-source-does-alone) is not a result of this test. |
| P-04 | Review contraction and apostrophe-spacing normalization. | Candidate equivalences include `it's` / `it is` and `d' accord` / `d'accord`. Review the [style diagnostic](stt/screening.md#what-the-voxpopuli-en-nonnative-gains-are) and [frozen-scoring rule](decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md) before changing metrics. Do not silently rewrite a campaign's scoring rules or historical results. |
| P-05 | Review the NASA source filters. | Decide whether to exclude Curious Universe entirely and whether the edge-word gate's selection bias is acceptable. The [NASA card](stt/datasets.md#nasa-en-train--nasas-podcasts-labelled-from-nasas-own-transcripts) records the current filters and counts. |
| P-06 | Re-prepare talk sets made before label-character cleanup. | Review the [training-source text rules](stt/datasets.md#training-sources) before a full training run. The existing screens do not need to be repeated solely to organize this repository. |
| P-07 | Review cache and screening-checkpoint retention. | Inventory the local files and which reports need them before deleting anything. Machine-specific paths and estimates belong in ignored local notes, not this backlog. |

## Owner review needed

| ID | Question | Reference |
|---|---|---|
| O-01 | Review the STT evaluation-core proposal. | [Proposed decision 0010](decisions/proposed/0010-stt-evaluation-core.md). |
| O-02 | Resolve the status and scope of unlicensed evaluation datasets. | [Proposed decision 0011](decisions/proposed/0011-unlicensed-datasets-for-private-evaluation.md). It remains proposed; do not infer a general permission from a particular dataset's use. |
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
| N-08 | Research low-bit training for audio models. | [Idea note](research/low-bit-training.md). No literature review, implementation or run has started. |
