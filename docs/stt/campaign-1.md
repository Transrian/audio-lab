# Campaign 1

Campaign 1 trains and selects the first candidate models of the baseline recipe
of [decision 0004](../decisions/accepted/0004-parakeet-fr-baseline-recipe.md) on
the training mix of [proposed decision
0016](../decisions/proposed/0016-training-mix-v1.md) option 1 (its label-quality
question O-11 answered "accept"), without decision 0004's noise clause
([proposed decision
0015](../decisions/proposed/0015-noise-augmentation-in-the-baseline-recipe.md)
option 1), and scores every run under normalizer v2 ([proposed decision
0013](../decisions/proposed/0013-normalizer-v2-contractions.md)); the panels, the
scoring rules and the sealed set are frozen for the whole campaign by [proposed
decision 0017](../decisions/proposed/0017-campaign-1-test-freeze.md), which also
fixes the candidate set and the rule that names the winner before any candidate
number exists. The pending decisions the campaign assumes — 0013, 0015 and 0016
— are treated as validated for testing while it runs: that working assumption is
proposed decision 0017's, and `configs/train/mix-v1.yaml` records it, so nothing
here decides them and no part of any of them is moved. This page records what
the campaign measures: its candidates, the decision rule of each experiment
(committed before that experiment's numbers exist), and the reference runs every
number is read against. Task numbers (T6, T7, T8 …) come from the campaign's
plan under `plans/campaign-1/`, which is local and not committed.

## Candidates

The candidate set is fixed by [proposed decision
0017](../decisions/proposed/0017-campaign-1-test-freeze.md) § 4: `mix-v1` and its
noise arm `mix-v1-n40`. No candidate is added later, and the seed replicate
`mix-v1-s2` is not a candidate. A row is filled in when its run has been trained
and scored (`full` decides, `fast` is reported alongside, per 0017 § 4).

| candidate | config | training | `full` scored | `fast` scored | status |
| --- | --- | --- | --- | --- | --- |
| `mix-v1` | `configs/train/mix-v1.yaml` | — | — | — | — |
| `mix-v1-n40` | `configs/train/mix-v1-n40.yaml` | — | — | — | — |

## Experiment rules

A rule is committed here **before** the numbers it judges exist; the numbers and
the verdict are written next to it when the run finishes. Every comparison is
paired, uses the intervals of proposed decision 0017 § 2, and scores under
normalizer v2.

### T6 — noise at the recipe's scale (`mix-v1-n40`)

Committed 2026-11-21, before the run started. `mix-v1-n40`
(`configs/train/mix-v1-n40.yaml`) is `configs/train/mix-v1.yaml` plus MUSAN noise
mixed into 40 % of the clips at 0–20 dB — the clause of decision 0004 that
[proposed decision
0015](../decisions/proposed/0015-noise-augmentation-in-the-baseline-recipe.md)
option 1 drops from the baseline recipe and leaves to "a later, targeted test",
and that option 2 names "a test at the recipe's scale". The screening measured
the clause at 2 000 updates on one source ([screening.md, "Noise
augmentation"](screening.md#noise-augmentation)); this run measures it at the
recipe's scale, on mix v1's data.

**The comparison.** `audiolab compare --normalizer 2 BASE mix-v1-n40` and
`audiolab compare --normalizer 2 mix-v1 mix-v1-n40`, `--json` for the per-panel
numbers, on `full.yaml`, where BASE is the base run of proposed decision 0017
§ 4, `$AUDIOLAB_RUNS/eval/parakeet-tdt-0.6b-v3-nemo/full`. `mix-v1-n40` minus the
base says where the arm sits; the noise's effect is the direct one, `mix-v1`
versus `mix-v1-n40`. A panel's difference is *separable* when its paired 95 %
interval stays away from zero.

**The rule.** The noise **adds value** only if both hold:

1. **A qualifying noisy gain.** At least one `noisy-*-full` panel improves
   separably by more than that panel's screening run-to-run spread — its
   `mix-v1-n40` minus `mix-v1` interval lies wholly below zero **and** its point
   estimate is larger, in magnitude, than the spread in the table below. The
   largest such qualifying improvement, in magnitude, is *the best qualifying
   noisy gain*.
2. **No larger non-noisy loss.** No non-noisy panel of `full.yaml` — the 15
   others, the two `nonspeech*` hallucination rates included — shows a separable
   loss: a `mix-v1-n40` minus `mix-v1` interval wholly above zero whose point
   estimate is larger than the best qualifying noisy gain of clause 1.

Magnitudes are compared in points, whatever the metric — WER points on a speech
panel, hallucination-rate points on a `nonspeech*` one (`audiolab compare --json`
gives fractions: ×100) — as the screening's own rule compares its `B − A`
differences ([screening.md, "Matched control: the rule, before the
numbers"](screening.md#matched-control-the-rule-before-the-numbers)).

Otherwise the result is **"not shown at recipe scale"**; that result does not
decide 0015 by itself (the owner does).

**The screening run-to-run spread on a panel** is the bound farthest from zero
of the 95 % interval of the matched-control − original-control difference that
panel records in
[screening.md, "Noise augmentation"](screening.md#noise-augmentation) (its
run-to-run table), read as the section itself reads it: the observed `n40` gain
on `noisy-fleurs-en` is "within the run-to-run interval on that panel (up to
0.42)". The three noisy panels carried:

| noisy panel | matched − original control | spread |
| --- | --- | --- |
| `noisy-fleurs-en-full` | +0.15 [−0.10, 0.42] | 0.42 |
| `noisy-fleurs-fr-full` | +0.03 [−0.37, 0.43] | 0.43 |
| `noisy-servicenow-codeswitched-en-fr-full` | +0.32 [−0.58, 1.16] | 1.16 |

A paired interval excluding zero is therefore not by itself enough: the gain
must also be larger than the run-to-run noise the screening measured on that
panel.

**What the rule does not do.** Its verdict is about proposed decision 0015 only:
whether the recipe keeps decision 0004's noise clause. It never chooses the
campaign's candidate — proposed decision 0017 § 4 does that, and `mix-v1-n40` is
one of the two candidates it fixes — and its result never changes a threshold of
that rule.

### T11 — seed replicate of mix v1 (`mix-v1-s2`)

Committed 2026-11-21, before the run started and before any `mix-v1-s2` number
exists. `mix-v1-s2` (`configs/train/mix-v1-s2.yaml`) is
`configs/train/mix-v1.yaml` with one field changed, `seed: 20261108` against
`mix-v1`'s `seed: 20261107`; every other field — the mix's rows and their
shares, the updates, the rates and the batch — is identical. The seed is not the
trainer's alone: each row of a mix is drawn with a seed derived from the run's
seed and the row's name alone ([training.md, "Mixing
datasets"](training.md#mixing-datasets)), so a new seed redraws every row — the
replicate differs from `mix-v1` in the seed **and** in the clips drawn. It trains
about 21 h on one RTX 3090, 3090 A, as soon as `mix-v1` is trained and its T8
evaluation has run. It is **not a candidate**.

**The comparison.**

```bash
audiolab compare --normalizer 2 --json \
  "$AUDIOLAB_RUNS/eval/mix-v1/full" "$AUDIOLAB_RUNS/eval/mix-v1-s2/full"
```

On `full.yaml`, paired, normalizer v2 (proposed decision 0017 § 2): the two
directories are the `full` directories T8's evaluation writes for the two runs,
`a` = `mix-v1` and `b` = `mix-v1-s2`. For every panel of the suite, Δ_T11 =
`mix-v1-s2` minus `mix-v1` is that comparison's `difference`, and together with
its paired 95 % interval it is **the recipe-scale run-to-run spread** on that
panel: the seed and the draw are the only things this replicate changes, so what
falls inside it is what repeating the recipe reproduces.

**What the report says.** Per panel of `full.yaml`, the report written next to
this rule states whether T6's difference, Δ_T6 = `mix-v1-n40` minus `mix-v1`
(T6's pre-registered comparison above), is **larger** than that panel's spread.
"Larger" is read mechanically on the point estimates, in points of the panel's
metric: on a panel, Δ_T6 is larger than the spread when **|Δ_T6| > |Δ_T11|**
there.

Magnitudes are compared in points, whatever the metric — WER points on a
speech panel, hallucination-rate points on a `nonspeech*` one (`audiolab compare
--json` gives fractions: ×100) — as the T6 rule above compares its own
differences. This is a **reporting** statement only: it does not change T6's
pre-registered verdict, which stands as written, and it relaxes no threshold of
proposed decision 0017 § 4; it informs the owner's later reading. Panels are
reported one by one, and no panel's numbers are pooled or averaged with another's.

**What the rule does not do.** `mix-v1-s2` is not a candidate: [proposed
decision 0017 § 4](../decisions/proposed/0017-campaign-1-test-freeze.md) fixes the
candidate set at `mix-v1` and `mix-v1-n40`, each candidate its run's final
`model.nemo`, and records that the seed replicate is not added to it. This run
enters no candidate comparison, makes no candidate eligible or ineligible, and
never replaces the campaign's candidate.

## References

Every campaign number is read against these runs. Three of the four columns are
scored under normalizer v2, the campaign's scoring rule (proposed decision 0017
§ 2); the base v1 column records the base run as it was written, under v1, so
the change the v2 rules make on it stays visible. The table gives the
`normalized` corpus WER in percent, read from each run's `summary.json`; the two
`nonspeech*` panels have no WER and are read as their hallucination rate
([evaluation.md § 6](evaluation.md#6-what-each-number-means)).

| panel | base v1 | base v2 | crispasr v2 | qwen3 v2 |
| --- | --- | --- | --- | --- |
| `fleurs-fr-full` | 4.96 | 4.89 | 4.92 | 3.96 |
| `mls-fr-full` | 5.61 | 5.61 | 5.51 | 5.54 |
| `fleurs-en-full` | 5.24 | 5.16 | 5.16 | 3.43 |
| `librispeech-en-full` | 2.04 | 2.00 | 2.00 | 1.64 |
| `voxpopuli-en-nonnative-full` | 20.44 | 19.33 | 19.05 | 18.46 |
| `servicenow-codeswitched-en-fr-full` | 13.26 | 13.16 | 13.02 | 12.92 |
| `servicenow-codeswitched-en-fr-ca-full` | 15.18 | 15.15 | 15.15 | 13.51 |
| `cs-fleurs-fr-en-full` | 16.27 | 16.25 | 16.10 | 16.50 |
| `nonspeech-full` \* | 5.00 | 5.00 | 5.36 | 16.43 |
| `nonspeech-music-full` \* | 2.63 | 2.63 | 1.75 | 25.44 |
| `african-accented-fr-test-full` | 6.57 | 6.57 | 6.57 | 4.93 |
| `notsofar-en-eval-full` | 7.38 | 6.80 | 6.71 | 6.77 |
| `noisy-fleurs-fr-full` | 16.44 | 16.38 | 16.26 | 13.73 |
| `noisy-fleurs-en-full` | 10.28 | 10.20 | 10.06 | 6.84 |
| `noisy-servicenow-codeswitched-en-fr-full` | 19.91 | 19.89 | 19.16 | 20.21 |
| `nasa-en-test-full` | 5.19 | 5.12 | 5.03 | 4.10 |
| `peertube-edu-captions-fr-full` | 15.25 | 15.31 | 14.01 | 7.77 |
| `peertube-edu-captions-fr-v2-full` | 15.54 | 15.59 | 14.43 | 8.07 |

\* The two `nonspeech*` panels are the hallucination battery: the number is the
hallucination rate in percent, not a WER.

**The runs.** Every column covers the same 18 panels of proposed decision 0017
§ 1 (`full.yaml`). The base's run scores all 18; the two served models were
evaluated before the 18th panel existed, so their row for it comes from a
separate run.

- **base** — `parakeet-tdt-0.6b-v3-nemo`
  (`configs/models/parakeet-tdt-0.6b-v3-nemo.yaml`, the local NeMo server),
  `$AUDIOLAB_RUNS/eval/parakeet-tdt-0.6b-v3-nemo/full`: 18 panels, 8 312 items,
  0 failed, written under normalizer v1. The v1 column is that directory's
  `summary.json`; the v2 column is its `full-v2` copy.
- **crispasr** — `parakeet-tdt-0.6b-v3-crispasr`
  (`configs/models/parakeet-tdt-0.6b-v3-crispasr.yaml`, the model served by
  CrispASR on the owner's home server),
  `$AUDIOLAB_RUNS/eval/parakeet-tdt-0.6b-v3-crispasr/full`: 17 panels, 7 712
  items; its v2 column is its `full-v2` copy.
- **qwen3** — `qwen3-asr-1.7b-vllm` (`configs/models/qwen3-asr-1.7b-vllm.yaml`,
  the same server's vLLM), `$AUDIOLAB_RUNS/eval/qwen3-asr-1.7b-vllm/full`: 17
  panels, 7 712 items; its v2 column is its `full-v2` copy.
- `peertube-edu-captions-fr-v2-full` was added to `full.yaml` after the two
  served models' runs, so their row comes from
  `$AUDIOLAB_RUNS/eval/<model>/captions-v2-2026-11-15` (2 panels, 700 items,
  the run of [datasets.md, "Reference runs of the v2 captions
  panels"](datasets.md#reference-runs-of-the-v2-captions-panels-2026-11-17))
  and, under v2, from its `captions-v2-2026-11-15-v2` copy. The base's 18-panel
  run scores that panel itself.

The base v2 column is the base that proposed decision 0017 § 4 compares against:
`audiolab compare --normalizer 2` on the base's `full` run reproduces it panel
for panel. Its unweighted mean over 0017's seven primary panels is 11.03 %.

The two served models' v1 numbers for the new panels are the reference runs of
[datasets.md, "Reference runs of the new
panels"](datasets.md#reference-runs-of-the-new-panels-2026-11-13); this page
holds their v2 reading and does not repeat the v1 tables. The base's v1 numbers
are the phase 1 base column reproduced in the
[screening](screening.md#results), for the nine panels that column carries. Every
v1 number of every reference is in the cited directory's `summary.json`.

**How a v2 column is made.** `audiolab score` rewrites a run directory in place,
so each v2 column comes from a `cp -a` copy of the cited directory, re-scored
inside the copy:

```bash
cp -a "$AUDIOLAB_RUNS/eval/<run>" "$AUDIOLAB_RUNS/eval/<run>-v2"
audiolab score --normalizer 2 "$AUDIOLAB_RUNS/eval/<run>-v2"
```

`score` re-scores the stored records with the normalizer frozen at `9843311`
(proposed decision 0017 § 2) and sends no request; the copy's `summary.json`
records the version it actually used (2), and the cited original directory stays
byte-identical. `score` rewrites `summary.json` and `report.md` only; the copy's
`run.json` still records the eval's version (1). `audiolab eval` is never used
for this: it would send the clips to a server again and rewrite the directory
whose numbers are cited. This is the recommended practice of [evaluation.md
§ 5](evaluation.md#5-text-normalization-three-views-two-versions), applied to
every reference above.

**No Whisper reference.** No `whisper-large-v3` run on the full suite exists, so
the campaign has no Whisper column. Producing one means transcribing all 8 312
items through the model of `configs/models/whisper-large-v3-vllm.yaml` (the
owner's home server, which the campaign's training runs also need); it is out of
scope for this campaign unless a window opens in which no training run holds
that server.
