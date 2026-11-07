# 0004 — Start from the known recipe, without personal data

- Scope: stt/parakeet-fr
- Derived from: D-005, D-013
- Decided by: owner, 2026-11-07

## Decision

The first training run reproduces the recipe that worked in the owner's earlier
private work, using public data only. It is the baseline every later change is
compared against.

Starting point, to be re-measured in this repository:

- base model: NVIDIA Parakeet TDT 0.6B v3, full encoder fine-tuned;
- constant learning rate, no scheduler: 1.5e-5 for the decoder heads, 5e-6 for the
  encoder; about 30,000 updates;
- noise added to about 40 % of training samples;
- public French data (Granary, VoxPopuli) plus a small share of English.

## Why

These settings gave a real gain on public French test sets before. Starting from
them saves weeks of search, and reproducing them on public data alone tells us
how much of the earlier gain came from personal data, which is excluded here
(D-013).

## Alternatives considered

None discussed.
