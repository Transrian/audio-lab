# 0003 — CI checks run on every push

- Scope: global
- Derived from: D-005, D-013
- Decided by: owner, 2026-11-07

## Decision

GitHub Actions runs on every push to `main`:

- lint and formatting checks;
- the CPU test suite;
- repository hygiene: no audio files, no secrets, no private network addresses,
  and the `Owner-Approved:` trailer on protected files
  ([decision 0005](0005-owner-approved-trailer.md));
- once datasets exist: a train/test leakage audit on the dataset manifests.

CI never needs a GPU, datasets or model weights. The full leakage audit on audio
content runs locally before each training run.

## Why

Agents of varying capability commit directly to `main` (D-002, D-014). Automatic
checks catch mistakes that a less capable agent would not notice, before they
spread.

## Alternatives considered

None discussed.
