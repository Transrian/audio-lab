# 0002 — Frozen evaluation and a sealed final test set

- Scope: stt
- Derived from: D-005
- Decided by: owner, 2026-11-07

## Decision

1. Before a training campaign starts, its test sets and its scoring rules (text
   normalization, WER computation, confidence intervals) are fixed. They do not
   change during the campaign.
2. Each campaign also has a **sealed** final test set. It is scored once, on the
   single candidate chosen beforehand using the other test sets.

## Why

When the same test set is used to compare dozens of checkpoints, the winner ends
up tuned to that test set and its score is optimistic. In the owner's earlier
work, a model looked clearly better on a heavily reused test pool, while on a
fresh, never-used set the difference was not statistically significant. Fixing
the rules first and keeping one set unseen gives an honest final number.

## Alternatives considered

None discussed.
