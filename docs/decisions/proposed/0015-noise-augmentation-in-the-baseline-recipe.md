# 0015 — Noise augmentation in the baseline recipe

- Scope: stt/parakeet-fr
- Derived from: decision 0004 (the "noise added to about 40 % of training samples" clause, never implemented until now)
- Decided by: proposed by an agent, 2026-11-14; rewritten 2026-11-15 from the matched-control evidence; awaiting owner review

## Finding

Decision 0004's recipe always included noise on about 40 % of the training
samples; every run in this repository so far omitted it. The test measures
that clause at screening scale: 2 000-update fine-tunes of the base model on
10 000 clips of `peertube-fr-train` at 1e-5 / 3e-5 — (a) MUSAN noise on 40 %
of the clips at 0–20 dB (`noise-peertube-fr-n40`) and (b) that plus room
reverb on 20 % of the clips (`noise-peertube-fr-n40-r20`) — plus a **matched
control** (`noise-peertube-fr-control-1x16`): the original `screen-peertube-fr`
control's recipe retrained with the candidates' micro-batch (1 × 16), card
class (RTX 3090) and code, without augmentation. The original control was
trained with a different clip order, a different micro-batch (not loss-neutral
under the model's `mean_volume` loss reduction), a different code version and
a different card, so its differences against the candidates mix the
augmentation with run-to-run variance; the matched control carries everything
the candidates carry except the augmentation.

Result (paired, normalized WER; full tables in
[screening.md](../../stt/screening.md), "Noise augmentation"):

- **no noisy panel improved with a 95 % interval excluding zero for either
  dose** against the matched control (the noisy-panel deltas run from −0.31
  to +0.01);
- **n40 = reject** under the rule committed before the numbers: its only
  separable clean loss is `servicenow-codeswitched-en-fr-ca` at +0.75
  [0.03, 1.63]; its largest noisy gain (0.31, `noisy-fleurs-en`) is smaller
  than that loss;
- **n40r20 = reject**: its separable clean losses are the `nonspeech`
  hallucination rate at +1.43 [0.36, 2.86] and `mls-fr` at +0.07 [0.01,
  0.13]; its largest noisy gain (0.24) is smaller than the larger of the two.
  This reject rests entirely on the hallucination-rate difference: 2 of the
  panel's 280 clips hallucinate for the matched control, 6 for n40r20 — a
  four-clip gap (the two control runs of the same recipe differ by two clips
  on that panel). Count WER only and the arm is **inconclusive** — its only
  separable clean loss is then the 0.07-point `mls-fr` one, which its 0.24
  noisy gain exceeds — and the rule's n80 clause would have run an 80 % copy
  of it. The rule pre-committed this case, so the verdict stands, but it is
  not robust to whether the hallucination-rate panel counts;
- **no n80**: both arms reject, so the committed n80 clause does not fire.

The rule written before the numbers was applied as written. The multiplicity
is reported with its caveats: of the 61 pre-registered comparisons, 7 are
separable, against the 3.05 expected from test-set sampling alone — the
comparisons are not independent and the intervals cover test-set sampling
only, not training variance, so the count is a benchmark, not a p-value.
Run-to-run, the two control runs of the same recipe agree within about
0.1–0.4 points on the 17 panels, with one separable difference
(`voxpopuli-en-nonnative` +0.23 [0.04, 0.50]) — the same scale as the
candidate effects.

The earlier, unmatched comparison (against the original control) showed the
same pattern — no separable noisy gain, and a +0.22 [0.02, 0.45] loss on
`voxpopuli-en-nonnative` for n40. That panel holds 600 clips from only 22
speaker groups and rewards the base model's verbatim style; its loss is one
of many uncorrected comparisons on single runs, and against the matched
control the same difference is −0.01 [−0.17, 0.20], not separable. It is
recorded as history, not as evidence.

The 40 % dose is measured at screening scale and, as is, not supported there.
The screens do not measure decision 0004's ~30 000-update recipe, so this is
not a verdict on the recipe itself.

## Question

Does the baseline recipe keep or drop noise augmentation?

## Options

1. **Drop the noise clause** from the baseline recipe (0004's other settings
   stand). Cheapest; leaves the robustness question for a later, targeted
   test.
2. **Test a different noise before deciding**: a different SNR range, babble
   noise (speech noise) instead of MUSAN, or a test at the recipe's scale
   (about 30 000 updates on the recipe's data, where the dose sits in the
   full recipe). Each is a matched-control cycle of its own: the first two
   cost a screening-scale retrain and a full-suite evaluation (the two control
   evaluations above took 8 and 24 min); the third is a run of the recipe
   itself — at the screening's measured 0.41 updates/s that is about 20 h of
   one RTX 3090, plus evaluation — and only the third answers the recipe's
   question.
3. **Keep the clause as a default** anyway, because the earlier project's
   recipe used the noise. The measured 40 % dose at screening scale does not
   support it: no separable noisy-panel gain, and separable clean losses on
   both candidates (the larger one resting on a four-clip hallucination gap).
   Not recommended.

Whichever way this is decided, decision 0004 is a record of the starting
recipe and is not edited.
