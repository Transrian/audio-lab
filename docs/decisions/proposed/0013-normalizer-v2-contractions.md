# 0013 — Normalizer v2: contractions and elision spacing, for the next campaign

- Scope: stt
- Derived from: D-005, decisions 0002, 0007
- Decided by: proposed by an agent, 2026-11-10; awaiting owner review

## Proposal

Scoring today treats "it's" and "it is" as different words, and "j' ai" as
two words where "j'ai" is one. The screening shows both artifacts in the
data: the base model expands contractions the way the European Parliament's
edited report does, while the verbatim references and several training
sources do not; and the first African-accented French run learned a space
after the apostrophe from its labels ([screening](../../stt/screening.md)).

This proposes a version 2 of the normalizer that adds two rules on top of
the v1 rules, applied to both sides of a score (reference and hypothesis):

1. **English contractions map to a canonical expanded form.** A fixed table:
   `i'm`→`i am`, `i've`→`i have`, `i'll`→`i will`, `you're`→`you are`,
   `you've`→`you have`, `you'll`→`you will`, `we're`→`we are`, `we've`→
   `we have`, `we'll`→`we will`, `they're`→`they are`, `they've`→`they
   have`, `they'll`→`they will`, `isn't`→`is not`, `aren't`→`are not`,
   `wasn't`→`was not`, `weren't`→`were not`, `haven't`→`have not`,
   `hasn't`→`has not`, `hadn't`→`had not`, `don't`→`do not`, `doesn't`→
   `does not`, `didn't`→`did not`, `can't`→`cannot`, `couldn't`→`could
   not`, `won't`→`will not`, `wouldn't`→`would not`, `shouldn't`→`should
   not`, `mustn't`→`must not`, `let's`→`let us`; `'s` expands to ` is`
   only after `it that there here what who he she where how` (never for
   possessives such as `john's`); `'ll` → ` will` and `'ve` → ` have` on
   any stem; `'d` is **never** expanded, because `he'd` is had or would and
   has no single canonical form.
2. **The French elision spacing is closed:** `x' y` → `x'y` for `x` in
   `c d j l m n s t`, `qu`, `jusqu`, `lorsqu`, `puisqu` (`j' ai` → `j'ai`,
   `d' accord` → `d'accord`). A missing apostrophe (`l homme`) is a
   different defect and is not fixed.

Everything else stays v1. The normalizer version becomes
`NORMALIZER_VERSION = 2` (`src/audiolab/eval/normalize.py`) for runs of the
**next** campaign, recorded in every summary as v1 is recorded; it does not
apply to the running or past campaigns ([decision 0002](../accepted/0002-frozen-evaluation-and-sealed-test-set.md)).
Every v1 result — including the whole screening — stays as published; no
historical number is rewritten. The rule set is fixed and reviewable in
[`experiments/normalization-review`](../../experiments/normalization-review/README.md),
where the diagnostic that motivates it was run.

## Why

A dictation user says "it's" or "j'ai", and the two spellings mean the same
thing; counting them as errors scores style, not recognition. The diagnostic
re-scoring of the 16 screening runs against the base model (2026-11-10, the
official pipeline plus the two rules above, with the official paired
intervals) shows what the rules would do:

- they remove most of the damage the spaced elision did in the first
  African-accented run: fleurs-fr +11.45 → +1.19, mls-fr +12.86 → +0.51,
  cs-fleurs-fr-en +10.52 → +2.19, servicenow en-fr +23.74 → +12.55 and
  en-fr-ca +20.59 → +7.77 points vs base, the same order as the earlier
  apostrophe-only diagnostic. The run stays separably harmful, so its
  screening conclusion is unchanged;
- the non-native-English "gains" shrink by 0.4 to 1.0 point and stay
  separable: ICSI −6.16 → −5.47, NOTSOFAR −5.13 → −4.77, LibriSpeech-PC
  −1.61 → −1.37, AMI −3.66 → −2.93, the talk sources −6.01 to −7.41 → −5.17
  to −6.48, NASA −3.94 → −3.62. Most of those gains is deletions and edge
  style, not contractions, so a narrower rule set was chosen on purpose;
- one screening conclusion moves: NASA's harm on servicenow en-fr, separable
  at +1.93 [+0.52, +3.38] under v1, is not separable at +1.35 [0.00, +2.74]
  under the diagnostic. That flip is a knife-edge: the diagnostic lower
  bound is exactly 0.0 — of the 10 000 bootstrap draws, 243 fall below zero,
  15 land exactly on it, and the 2.5th percentile sits on that atom — so it
  should be read as "the diagnostic removes NASA's separability on that
  panel", not as a large effect.

The cost is the intended one: a model that rewrites the user's words into
another style ("it's" → "it is", or the other way round) is no longer
counted against. That trade-off is the owner's to take: whether contraction
and elision equivalence becomes the official scoring rule of the next
campaign. The owner answered "probably yes" to that question on 2026-11-10;
the answer is recorded here as an answer, not an acceptance — this proposal
is the formal step before acceptance.

## Alternatives considered

- **Keep v1 for the next campaign as well.** Every screening conclusion
  stays exactly as published, at the cost of scoring style as errors on
  panels whose references are verbatim, and of re-measuring the same elision
  artifact in any future run trained on spaced-apostrophe text.
- **Report both views, v1 and v2, in every future summary.** The headline
  doubles, and the reader still has to be told which view decides which
  question; the campaign's winning rule is one, per decision 0002.
- **Wider equivalence classes** (courtesy formulas, near repeats, runs of
  words deleted at the edges — the earlier style diagnostic that left ICSI's
  gain at −1.41). Those hide more than style: the deletions that carry the
  verbatim gains would disappear from the score, so they are left out of v2.
