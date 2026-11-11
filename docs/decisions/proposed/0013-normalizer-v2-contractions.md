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

This proposes a version 2 of the normalizer that adds three rules on top of
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
2. **That `'s` is expanded by its context**, because `it's been`, `he's got`
   and `that's gotten` mean `has`: `'s` expands to `has` before `been`, `got`
   or `gotten`, and to `is` everywhere else the pronoun rule applies. The
   table above names the `is` reading only; this rule chooses between the two
   readings.
3. **The French elision spacing is closed:** `x' y` → `x'y` for `x` in
   `c d j l m n s t`, `qu`, `jusqu`, `lorsqu`, `puisqu` (`j' ai` → `j'ai`,
   `d' accord` → `d'accord`). A missing apostrophe (`l homme`) is a
   different defect and is not fixed.

Rule 2 is the limitation the diagnostic measured: the experiment applied the
simpler reading (`'s` always `is`), which turns a verbatim `it's been` into
`it is been`. That is wrong in both directions. It counts a **correct reading as
an error**: a hypothesis saying "it has been" — the base model's style — no
longer matches the transformed reference, which now says `is` where the speaker
said `has`. And it counts an **incorrect reading as correct**: against the
reference `it's been` (the verbatim spelling of the same words), a hypothesis
that read `is` is scored right, because the transformed reference says `it is`
too. The `has` pattern is common in meeting speech (the ICSI and NOTSOFAR panels
are meetings), so both directions have weight. Rule 2 is therefore the
recommended implementation, not an alternative: a v2 that expands `'s` to `is`
everywhere would score style correctly on `it's` and incorrectly on
`it's been`.

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
re-scoring of the 16 screening runs against the base model (the official
pipeline plus the two rules of 1 and 3, with the official paired intervals)
shows what the rules would do. Its numbers are in
[screening.md, "What the voxpopuli-en-nonnative gains are"](../../stt/screening.md#what-the-voxpopuli-en-nonnative-gains-are);
that section is their home, and this proposal does not repeat its tables. What
matters for the decision: the run that learned the spaced elision stays
separably harmful, the non-native-English gains shrink but stay separable, and
exactly one screening conclusion moves — NASA's harm on servicenow en-fr, from
separable to not separable, on an interval that ends on zero: a knife-edge
rather than a large effect. The diagnostic used the simpler `'s` reading (always
`is`), which is the limitation rule 2 fixes: the reported numbers therefore
measure rules 1 and 3, and the `'s`-before-a-participle cases are the ones a v2
must still get right.

The cost is the intended one: a model that rewrites the user's words into
another style ("it's" → "it is", or the other way round) is no longer
counted against. That trade-off is the owner's to take: whether contraction
and elision equivalence becomes the official scoring rule. The owner answered
"probably yes" to that question on 2026-11-10 and then deferred the decision the
same day, leaving it open for the next campaign; the answer is recorded here as
an answer, not an acceptance — this proposal stays proposed, and nothing under
`src/audiolab/eval/` implements it.

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
