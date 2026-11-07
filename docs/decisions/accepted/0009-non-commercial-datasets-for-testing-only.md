# 0009 — Non-commercial datasets may be used for testing only

- Scope: stt
- Derived from: D-005, D-006
- Decided by: owner, 2026-11-07

## Decision

A dataset whose license restricts commercial use (for example CC BY-NC or
CC BY-NC-SA) may be used to **evaluate** models and to publish the resulting
scores, provided that:

- it is never used to train a model meant for publication
  ([decision 0001](0001-parakeet-fr-model-license.md));
- the repository never redistributes it: it contains only download scripts and
  scores;
- the license of each such dataset is checked and recorded in its dataset card
  before first use.

## Why

Without these datasets, French test coverage has almost no real spontaneous
speech: most interview and conversation corpora (CFPP2000, TCOF, ESLO, ORFEO)
are non-commercial. Evaluating on them and publishing scores is non-commercial
research use.

## Alternatives considered

None discussed.
