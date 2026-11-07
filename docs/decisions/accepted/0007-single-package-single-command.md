# 0007 — One package, one command, one config per experiment

- Scope: global
- Derived from: D-010
- Decided by: owner, 2026-11-07

## Decision

- All code lives in one Python package, `audiolab`, managed with
  [uv](https://docs.astral.sh/uv/).
- Everything is run through one command, `audiolab`, with subcommands per area
  (for example `audiolab eval`, `audiolab train`).
- An experiment is described by one small configuration file in `configs/`, not
  by a new script.

## Why

A reader should be able to use the repository after reading the README, without
copying files around. Shared code (configuration, GPU selection, run folders,
scoring) is written once and reused by every problem family.

## Alternatives considered

None discussed.
