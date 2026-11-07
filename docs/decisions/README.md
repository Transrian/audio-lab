# Decisions

A decision records something that is **not** a direct statement of the owner: an
inference, a design choice, or a trade-off. Owner statements live in
[`../owner/directives.md`](../owner/directives.md).

## Status

The folder a decision lives in is its status:

| Folder | Meaning | Binding? |
|---|---|---|
| `proposed/` | Written by an agent, not yet reviewed by the owner. | No. Treat it as a hypothesis and say so. |
| `accepted/` | The owner accepted it. | Yes, within its scope. |
| `rejected/` | The owner rejected it. Kept only when the reason is useful later. | No. |

Any agent may add a file to `proposed/`. Only the owner moves a decision to
`accepted/` or `rejected/`, through a commit carrying an `Owner-Approved:` trailer
([decision 0005](accepted/0005-owner-approved-trailer.md)).

An accepted decision is never rewritten. To change it, write a new decision that
supersedes it and link the two.

## Format

File name: `NNNN-short-slug.md`, numbered in creation order across all folders.

```markdown
# NNNN — Title

- Scope: global | stt | stt/<project> | ...
- Derived from: D-xxx, D-yyy
- Decided by: owner, YYYY-MM-DD   (absent while proposed)

## Decision
What is decided, in a few sentences.

## Why
The reasoning, in plain words.

## Alternatives considered
Only alternatives that were actually discussed. If none were, write "None discussed."
Never invent alternatives after the fact.
```

## Index

| ID | Scope | Status | Title |
|---|---|---|---|
| [0001](accepted/0001-parakeet-fr-model-license.md) | stt/parakeet-fr | accepted | A published Parakeet model is CC-BY-4.0 |
| [0002](accepted/0002-frozen-evaluation-and-sealed-test-set.md) | stt | accepted | Frozen evaluation and a sealed final test set |
| [0003](accepted/0003-ci-on-every-push.md) | global | accepted | CI checks run on every push |
| [0004](accepted/0004-parakeet-fr-baseline-recipe.md) | stt/parakeet-fr | accepted | Start from the known recipe, without personal data |
| [0005](accepted/0005-owner-approved-trailer.md) | global | accepted | `Owner-Approved:` trailer on protected files |
| [0006](accepted/0006-task-tiers-not-model-names.md) | global | accepted | Task tiers, never model names |
| [0007](accepted/0007-single-package-single-command.md) | global | accepted | One package, one command, one config per experiment |
| [0008](accepted/0008-training-monitoring-vs-official-evaluation.md) | stt | accepted | Training monitoring is direct; official evaluation goes through the API |
