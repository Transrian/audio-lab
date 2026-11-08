# 0011 — A dataset with no license may be used for evaluation only

- Scope: stt
- Derived from: D-009, D-013, decision 0009

## Decision

A public dataset whose card declares **no license** may be used to evaluate
models and to publish the resulting scores, under the same conditions as a
non-commercial dataset in [decision 0009](../accepted/0009-non-commercial-datasets-for-testing-only.md),
and one more:

- it is never used to train any model, published or not;
- the repository never redistributes it: it contains only a download script and
  scores;
- its dataset card says that no license was found, on which date, and links to
  this decision.

The first dataset concerned is `ServiceNow-AI/asr_codeswitched` (French–English
code-switching), the only test set found for the main target use of
[D-009](../../owner/directives.md): sentences switching between French and English
with technical vocabulary.

## Why

No permissively licensed French–English code-switching test set existed when the
sources were searched ([sources](../../stt/sources.md)). Without one, the main use
named in D-009 cannot be measured at all. A dataset with no license is all rights
reserved, so it is never trained on and never copied into the repository;
reporting the error rate a model gets on it is the narrowest possible use.

The owner approved using this set for evaluation only on 2026-11-08; this file
records that approval as a general rule so the next case is not decided silently.

## Alternatives considered

- Using only clearly licensed code-switching data (VoxPopuli accented English,
  and CS-FLEURS, which is non-commercial and read speech). Offered to the owner
  on 2026-11-08 as the alternative; he chose to use this set for evaluation as
  well.
