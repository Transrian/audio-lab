# AGENTS.md

Rules for LLM agents working in this repository. Humans should start with the
[README](README.md).

## Read first

1. [`docs/owner/directives.md`](docs/owner/directives.md): what the owner asked for.
2. [`docs/decisions/`](docs/decisions/README.md): accepted decisions, then proposed ones.
3. [`docs/roadmap.md`](docs/roadmap.md): the current phase.
4. `AGENTS.local.md`, if it exists (git-ignored): the facts of this machine,
   such as servers, paths and models. Facts that hold only on one machine go
   there, never into a committed file.

## Authority

Owner directives > accepted decisions > everything else.

- A directive or decision applies only within its scope. Do not extend it to
  another scope; write a proposed decision instead.
- A proposed decision is a hypothesis. When you rely on one, say so.
- When you state what the project requires, cite the directive or decision. If
  you cannot cite one, it is your inference: present it as such.
- If something you need is not decided, ask the owner or write a proposed
  decision. Do not decide silently.

## Protected files

Files listed in [`.github/protected-paths.txt`](.github/protected-paths.txt)
change only when the owner explicitly asked for that change. The commit then ends
with a trailer summarizing the request:

```
Owner-Approved: owner asked to <what>, YYYY-MM-DD
```

Never write this trailer without such a request.

## Task tiers

Tasks are **routine** unless listed here. **Strong** tasks need the most capable
agent available; if one was handed to you as routine, stop and say so.

Strong tasks:

- anything under `src/audiolab/eval/` or `src/audiolab/data/`;
- training recipes and anything that decides which checkpoint wins;
- writing or changing decisions;
- CI and `scripts/check_repo.py`.

## Never commit

- audio of any kind, dictations, personal transcripts (D-013);
- datasets, model weights, run outputs (they live outside the repository);
- secrets, tokens, private hostnames or private IP addresses.

## Workflow

- Commit directly to `main` (D-014). Rebase on `origin/main` before pushing.
- Small commits; message format `area: what changed` (for example `eval: add WER`).
- Before every push, run `./scripts/check.sh`. It must pass.
- If CI on `main` is red, fixing it comes before any other work.
- Keep working notes in `plans/` (git-ignored, local only; start with
  `plans/README.md` if it exists). Write there anything a later session would
  need: findings, open questions, plans in progress. Notes are never a source of
  truth: what lasts moves to `docs/` and is committed.

## Code and docs

- The maintained core uses one package (`src/audiolab/`), one command (`audiolab`)
  and configs in `configs/`. Explorations may use standalone scripts in
  `experiments/<topic>/` ([decision 0007](docs/decisions/accepted/0007-single-package-single-command.md)).
  Reuse `audiolab.core` where it fits; do not complicate the core for an exploration.
- New behavior comes with a CPU test in `tests/`.
- Cache expensive results (D-017). The cache key holds everything that can
  change the result and nothing else; a wrong reuse is a bug, a test proves it.
- Before acquiring, dropping or rejecting a data source, update
  [`docs/stt/sources.md`](docs/stt/sources.md) with its status and the reason (D-018).
- Write documentation for a human who has no access to your session: no
  references to conversations, no "as discussed", no history of how you got
  there. Every claim must be checkable from the repository.
- Each fact has one home. Link to it instead of repeating it.
- Keep this file short (CI enforces a word limit). Details belong in `docs/`.
