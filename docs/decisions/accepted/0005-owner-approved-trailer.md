# 0005 — `Owner-Approved:` trailer on protected files

- Scope: global
- Derived from: D-011, D-014
- Decided by: owner, 2026-11-07

## Decision

Some files may only change at the owner's explicit request. They are listed in
[`.github/protected-paths.txt`](../../../.github/protected-paths.txt) and include
the owner directives, accepted and rejected decisions, the CI checks themselves
(so that a check cannot be weakened without the owner knowing), and, once they
exist, frozen test sets and scoring rules.

A commit that touches a protected file ends with a trailer summarizing the
owner's request:

```
Owner-Approved: owner asked to accept decision 0008, 2026-11-20
```

CI rejects pushes where such a commit has no trailer.

## Why

Without pull requests (D-014), there is no approval button. The trailer keeps a
visible record that the owner asked for the change. It catches forgotten
approvals; it cannot stop an agent that writes a false trailer, so agents are
instructed never to do so.

## Alternatives considered

- Pull requests reviewed by the owner, with agents using a separate GitHub
  account or a GitHub App. Rejected by the owner as too complex.
