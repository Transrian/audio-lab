# 0008 — Training monitoring is direct; official evaluation goes through the API

- Scope: stt
- Derived from: D-005, D-015, D-016
- Decided by: owner, 2026-11-07

## Decision

There are two kinds of measurement, kept separate:

- **Training monitoring** (loss curves, a validation WER during training, early
  warnings) uses the model directly, inside the training process, and stays as
  simple as possible.
- **Official evaluation** (any result in a report, any comparison between
  models, choosing the winning checkpoint) always goes through the
  OpenAI-compatible API, as D-015 and D-016 require.

A monitoring number is never quoted as an evaluation result.

## Why

Starting a server at every validation step would make training slow and
complicated for no benefit. Official results, on the other hand, must all be
measured the same way, so that a local checkpoint and a remote model are
comparable.

## Alternatives considered

- Running every measurement, including training-time validation, through the
  API. Rejected: needless complexity during training.
