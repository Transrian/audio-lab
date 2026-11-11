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
`audiolab.eval.stats`) with its frozen settings (`stats.DRAWS`, `stats.SEED`).
No model, no audio, no network: only run files and dataset
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
  `has` reading before a participle is not handled either: `it's been`, `he's
  got` and `that's gotten` become `it is been`, `he is got`, `that is gotten`,
  so a correct `has` reading against a verbatim `'s` is still counted as an
  error. That pattern is common in meeting speech (ICSI, NOTSOFAR), so the
  proposed decision recommends a context rule instead: `'s` before
  `been|got|gotten` expands to `has`, otherwise to `is`.
- Out of scope, left as written: `y'all`, `ain't`, `'cause`, `'em`.
- The elision rule only removes a space after an apostrophe that is already
  there, and only the apostrophe of an elision: `l'homme` is untouched, a
  missing apostrophe (`l homme`) is a different defect and is not fixed, and an
  apostrophe before the letter (`rock 'n' roll`) is left alone. Every kind of
  space counts (`j'  ai` with two spaces, a no-break space), because the
  elision is the same in all of them.
- The same lookbehind leaves an elision at the very start of a single-quoted
  phrase unjoined (`il a dit 'j' ai faim'`), so it stays two tokens and the two
  spellings score differently there. The alternative would join a quoted `'n'`
  as well, which is the defect the lookbehind exists to avoid; keeping the
  unjoined reading is the chosen behaviour, pinned by a test, and no hypothesis
  of the base or candidate runs contains such a phrase, so no number moves.
- Typographic apostrophes (`’`, `ʼ`) are mapped to the ASCII one before the
  rules run, as the official normalizer would; no other character is touched, so
  a spacing this script joins can never be an artifact of a dash or a quotation
  mark being rewritten here.

## Reproduce

```bash
set -a; . ./.env; set +a
uv run python experiments/normalization-review/rescore.py
```

Base: `$AUDIOLAB_RUNS/eval/parakeet-tdt-0.6b-v3-nemo/full`; candidates: all
`$AUDIOLAB_RUNS/eval/screen-*/full`. The **full tables are this command's
stdout**: one table per panel, plus the changed conclusions and, for each of
them, how the `stats.DRAWS` bootstrap draws sit around zero. Nothing has to be
copied out of a note to read them.

`--rules elision` / `--rules contractions` give the ablations, `--rules none`
recomputes the official view as a sanity check, and `--base` / `--candidate` /
`--runs-dir` / `--panels-dir` / `--data-dir` override the defaults. A candidate
may be given as `LABEL=PATH` to name its row, as in
[experiments/source-screening/table.py](../source-screening/table.py). The unit
tests for the rules and the pipeline are in `tests/test_normalization_rescore.py`
(synthetic runs, CPU only).

The script refuses a run whose recorded normalizer version is not the version of
the scoring code (`audiolab.eval.normalize.NORMALIZER_VERSION`), and the report
header says which version produced the numbers: this diagnostic only means
anything for runs scored under the rules it applies.

## Results (the 16 screening runs)

The numbers are in
[docs/stt/screening.md](../../docs/stt/screening.md#what-the-voxpopuli-en-nonnative-gains-are),
with the proposed decision [0013](../../docs/decisions/proposed/0013-normalizer-v2-contractions.md);
the tables themselves are the reproduce command's stdout. In short: the
uncleaned African-accented run's elision damage is largely removed but stays
separably harmful, the non-native-English gains shrink and stay separable, and
one conclusion moves — NASA's harm on servicenow en-fr, separable officially,
is not separable under the diagnostic, on an interval that ends on zero.

## Limits

- The rules apply to both sides: a model that rewrites the user's words into
  another style is no longer counted against. That is the intended trade-off,
  and it is the owner's to accept (the question above).
- It is a re-reading of frozen runs, not a re-evaluation: latencies, empty
  outputs, truncations and the formatted view are untouched.
- The non-speech panel is unaffected: the rules never turn an answer into
  nothing or nothing into an answer, so its clips stay hallucinated or not.
- The rule tables are a fixed, reviewable set (decision 0002's spirit): they
  are written out above and in the proposed decision, not learned from the
  data.
