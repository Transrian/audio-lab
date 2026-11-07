# 0006 — Task tiers, never model names

- Scope: global
- Derived from: D-002
- Decided by: owner, 2026-11-07

## Decision

Agent instructions describe tasks by tier, never by model name:

- **routine**: any capable agent may do it, as long as all checks pass;
- **strong**: needs the most capable agent available.

The list of strong tasks lives in [`AGENTS.md`](../../../AGENTS.md). Which model
fills which tier is decided outside the repository.

## Why

Model names change every few months and mean nothing to a reader. Tiers describe
the risk of a task, which stays stable.

## Alternatives considered

- Naming the models in the repository. Rejected by the owner: it would not make
  a clean repository.
