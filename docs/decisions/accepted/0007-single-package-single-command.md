# 0007 — A shared core and standalone exploratory experiments

- Scope: global
- Derived from: D-010
- Decided by: owner, 2026-11-07; amended in place at the owner's request, 2026-11-09

## Decision

### Maintained core

- Shared, maintained code lives in one Python package, `audiolab`, managed with
  [uv](https://docs.astral.sh/uv/).
- Its entry point is one command, `audiolab`, with subcommands per area
  (for example `audiolab eval`, `audiolab train`).
- Each run supported by the core is described by one small configuration file
  in `configs/`, not by a new script.

### Exploratory experiments

- Research and hypothesis-testing code may live in a versioned
  `experiments/<topic>/` directory, with its own script entry point,
  configuration and isolated dependencies when needed. It does not have to
  become an `audiolab` subcommand or part of the package.
- Each experiment documents its question, the environment and commands needed
  to reproduce it, and its conclusions and limits, including negative or
  inconclusive results. Datasets, weights and run outputs stay outside the
  repository; the documentation references them without committing them.
- An experiment may depend on the core; the core must not depend on an
  experiment. Reuse existing components where they fit, but do not add core
  options or abstractions solely to accommodate one exploration. Small local
  duplication is preferable to premature generalization.
- Official STT comparisons still use the common evaluation protocol
  ([decision 0008](0008-training-monitoring-vs-official-evaluation.md)); a
  standalone script is not a separate source of official scoring rules. All
  applicable directives and safeguards still apply.
- An experiment may remain experimental indefinitely. Move useful code into
  the maintained core only when there is a concrete reason to share and
  maintain it.

## Why

A reader should be able to use the maintained core after reading the README,
without copying files around. Common capabilities stay in one place, while
exploratory work can test a different approach without adding lasting complexity
or heavy dependencies to the core. An unsuccessful experiment is still useful
research and does not need to become a supported feature.

## Alternatives considered

- Requiring every exploration to use the package, command and configuration
  structure of the core. Rejected: a short-lived or abandoned experiment could
  force lasting options and abstractions into otherwise simple code.
- Keeping experiment-specific modules inside `src/audiolab/`. Standalone topic
  directories provide a clearer boundary when that code is not intended to be
  maintained as part of the core.
