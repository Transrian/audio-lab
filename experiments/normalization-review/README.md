# Normalization review: contractions and apostrophe spacing (P-04)

Diagnostic only: the official score is not changed. `src/audiolab/eval/` and
`NORMALIZER_VERSION` stay as they are ([decision 0002](../../docs/decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)
freezes a campaign's scoring rules), and the extra rules live only in
`rescore.py`.

## Question

Do the screening conclusions change if scoring treats English contractions
(`it's` / `it is`) and the French elision spacing (`j' ai` / `j'ai`) as
equivalent? The base model expands contractions like the Parliament's edited
report while the verbatim references do not, and the first African-accented
French run learned a space after the apostrophe
([screening](../../docs/stt/screening.md)). The answer feeds Q6: a
normalizer version 2 for the next campaign
([proposed decision 0013](../../docs/decisions/proposed/0013-normalizer-v2-contractions.md)).

## What the script does

`rescore.py` reads the per-clip answer files (`records.jsonl`) of finished
evaluation runs under `$AUDIOLAB_RUNS/eval` and, for every panel of the full
suite, reports candidate minus base for the 16 screening runs against the
base model served the same way (`parakeet-tdt-0.6b-v3-nemo/full`):

- the **official** delta — the frozen normalizer v1 applied to the stored
  records; the script cross-checks every candidate's official column against
  the official `audiolab.eval.report.compare_runs` and stops if the two
  diverge;
- the **diagnostic** delta — the same pipeline with two extra rules applied
  in the script, to both sides (reference and hypothesis), before the
  normalizer:
  - English contractions mapped to a canonical expanded form (table below);
  - the French elision spacing: `x' y` → `x'y` for `x` in `c d j l m n s t`,
    `qu`, `jusqu`, `lorsqu`, `puisqu`.

Tokenization, edit counts, the per-group bootstrap and the paired difference
intervals are all the official code (`audiolab.eval.metrics`,
`audiolab.eval.stats`) with the frozen settings (10 000 draws, seed
20261107). No model, no audio, no network: only run files and dataset
manifests are read.

## The contraction table

Fixed mappings, both sides, case-insensitive:

| | | | |
|---|---|---|---|
| `i'm` → `i am` | `i've` → `i have` | `i'll` → `i will` | `you're` → `you are` |
| `you've` → `you have` | `you'll` → `you will` | `we're` → `we are` | `we've` → `we have` |
| `we'll` → `we will` | `they're` → `they are` | `they've` → `they have` | `they'll` → `they will` |
| `isn't` → `is not` | `aren't` → `are not` | `wasn't` → `was not` | `weren't` → `were not` |
| `haven't` → `have not` | `hasn't` → `has not` | `hadn't` → `had not` | `don't` → `do not` |
| `doesn't` → `does not` | `didn't` → `did not` | `can't` → `cannot` | `couldn't` → `could not` |
| `won't` → `will not` | `wouldn't` → `would not` | `shouldn't` → `should not` | `mustn't` → `must not` |
| `let's` → `let us` | `it's` → `it is` | `that's` → `that is` | `there's` → `there is` |
| `here's` → `here is` | `what's` → `what is` | `who's` → `who is` | `he's` → `he is` |
| `she's` → `she is` | `where's` → `where is` | `how's` → `how is` | |

Plus two suffix rules on any stem: `'ll` → ` will`, `'ve` → ` have` (they
cover the combinations the table does not list, such as `it'll`).

## Ambiguities (what the table deliberately does not do)

- `'d` is **never** expanded: `he'd` is had or would, and there is no single
  canonical form.
- `'s` is expanded only after `it that there here what who he she where how`,
  always to `is`. Possessives (`john's`, `tomorrow's`) are left alone. The
  `has` reading before a participle (`it's been`) is not handled; it is rare
  in speech and both sides of a score carry the same bias.
- Out of scope, left as written: `y'all`, `ain't`, `'cause`, `'em`.
- The elision rule only removes a space after an apostrophe that is already
  there. A missing apostrophe (`l homme`) is a different defect and is not
  fixed.
- Typographic apostrophes (`’`) are mapped to the ASCII one before the rules
  run, as the official normalizer would; no other character is touched.

## Reproduce

```bash
set -a; . ./.env; set +a
uv run python experiments/normalization-review/rescore.py
```

Base: `$AUDIOLAB_RUNS/eval/parakeet-tdt-0.6b-v3-nemo/full`; candidates: all
`$AUDIOLAB_RUNS/eval/screen-*/full`. `--rules elision` / `--rules contractions`
give the ablations, `--rules none` recomputes the official view as a sanity
check, and `--base` / `--candidate` / `--runs-dir` / `--panels-dir` /
`--data-dir` override the defaults. The unit tests for the rules and the
pipeline are in `tests/test_normalization_rescore.py` (synthetic runs, CPU
only).

## Results (2026-11-10, the 16 screening runs)

The full tables are in the plan file's `## Status` (git-ignored) and were
written there by this script's output. The official column reproduces
[docs/stt/screening.md](../../docs/stt/screening.md) exactly. What moves:

- The uncleaned African-accented run's elision damage is largely removed
  (fleurs-fr +11.45 → +1.19, mls-fr +12.86 → +0.51, cs-fleurs-fr-en
  +10.52 → +2.19, servicenow en-fr +23.74 → +12.55, en-fr-ca +20.59 → +7.77
  points vs base), the same order as the earlier apostrophe-only diagnostic;
  it stays separably harmful, so its conclusion does not change.
- The non-native-English gains shrink by 0.4 to 1.0 point and stay separable
  (ICSI −6.16 → −5.47, NOTSOFAR −5.13 → −4.77, LibriSpeech-PC −1.61 → −1.37,
  AMI −3.66 → −2.93, the talk sources −6.01 to −7.41 → −5.17 to −6.48,
  NASA −3.94 → −3.62): most of those gains is deletions and edge style, not
  contractions.
- One conclusion moves: **NASA's harm on servicenow en-fr, separable at
  +1.93 [+0.52, +3.38] under v1, is not separable at +1.35 [0.00, +2.74]
  under the diagnostic**. That flip is a knife-edge: the lower bound is
  exactly 0.0 (243 of the 10 000 bootstrap draws fall below zero, 15 land
  exactly on it, and the 2.5th percentile sits on that atom) — it says the
  diagnostic removes NASA's separability on that panel, not that there is a
  large effect.

## Limits

- The rules apply to both sides: a model that rewrites the user's words into
  another style is no longer counted against. That is the intended trade-off,
  and it is the owner's to accept (Q6).
- It is a re-reading of frozen runs, not a re-evaluation: latencies, empty
  outputs, truncations and the formatted view are untouched.
- The non-speech panel is unaffected by design: the rules never turn an
  answer into nothing or nothing into an answer.
- The rule tables are a fixed, reviewable set (decision 0002's spirit): they
  are written out above and in the proposed decision, not learned from the
  data.
