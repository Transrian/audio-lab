# Research

Questions, literature reviews and experimental findings that are useful beyond
one working session. This is not a second roadmap or a collection of accepted
architectural decisions: open work lives in the [backlog](../backlog.md), and
binding choices live in [decisions](../decisions/README.md).

## Topics

| Topic | Status | Home |
|---|---|---|
| Low-bit training for audio models | Idea; not investigated | [Question and open points](low-bit-training.md) |
| Per-source STT screening | Initial screens reported; follow-up work paused | [Method, results and limits](../stt/screening.md); [reporting script](../../experiments/source-screening/README.md) |

A result can stay in a domain document, as the screening does. Link to its home
instead of copying it here.

## Start small

A documentary investigation needs only a topic note. Add code under
[`experiments/<topic>/`](../../experiments/README.md) only when there is something
to run; an existing `audiolab` command and configuration may already suffice.
Standalone experiments do not have to become maintained core features
([decision 0007](../decisions/accepted/0007-single-package-single-command.md)).
There is no required framework, plugin registry or new package per topic.

A useful note records:

- **Status and question**: idea, investigation, running, concluded or abandoned;
  distinguish a hypothesis from an observation or accepted rule.
- **Evidence**: sources with links and access dates for a literature review;
  what they do and do not establish. Say when no research has been done.
- **Protocol, when ready**: what changes, what stays fixed, metrics and stopping
  criteria. An unselected protocol is a proposal, not a completed experiment.
- **Reproduction, when code exists**: link the script, configuration, environment,
  code revision and artifact identifiers needed to repeat the work. Keep paths
  portable and machine-specific setup local.
- **Outcome and limits**: report negative and inconclusive results too. Keep
  measurements distinct from their interpretation and do not generalize beyond
  the conditions tested.

No result or implementation is needed to record an idea. A closed experiment
need not be integrated into the core; archive its conclusion and reproduction
instructions instead of turning it into an ongoing feature commitment.

## Boundaries

The [agent rules](../../AGENTS.md) and applicable directives still apply, including
privacy, data licenses and external storage of datasets, weights and run outputs.
Commit research prose, code, configurations and curated aggregate findings, not
raw run artifacts or transcripts. Local hand-off notes remain in git-ignored
`plans/`; durable research belongs here or in its linked domain document.

Official STT comparisons use the [common evaluation](../stt/evaluation.md), not a
new scorer invented for an experiment. Keep the
[sealed set and campaign scoring rules](../decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)
intact. Exploratory diagnostics must be labelled as such and must not be quoted
as official evaluation results
([decision 0008](../decisions/accepted/0008-training-monitoring-vs-official-evaluation.md)).
