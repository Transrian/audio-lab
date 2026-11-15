# 0015 — Noise augmentation in the baseline recipe

- Scope: stt/parakeet-fr
- Derived from: decision 0004 (the "noise added to about 40 % of training samples" clause, never implemented until now)
- Decided by: proposed by an agent, 2026-11-14; rewritten 2026-11-15 from the matched-control evidence; awaiting owner review

## Finding

Decision 0004's recipe always included noise on about 40 % of the training
samples; every run in this repository before this test omitted it. The test
measures
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

- **one of the six noisy-panel comparisons against the matched control is
  separable — n40 on `noisy-fleurs-en`, −0.31 [−0.61, −0.03] — and the other
  five are not** (deltas from −0.31 to +0.01); the single gain is within the
  run-to-run interval on that panel (up to 0.42), so it is not a consistent
  gain from the noise;
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
Run-to-run, the two control runs of the same recipe differ by at most 0.43
points on the 15 WER panels, the `nonspeech` hallucination rate differs by
0.71 (two clips), and the only separable difference is
`voxpopuli-en-nonnative` at +0.23 [0.04, 0.50] — the same scale as the
candidate effects.

The earlier, unmatched comparison (against the original control) had no
separable noisy gain, and a +0.22 [0.02, 0.45] loss on `voxpopuli-en-nonnative`
for n40. That panel holds 600 clips from only 22 speaker groups and rewards a
verbatim transcription style — its references are raw verbatim while the base
model writes like an edited report ([screening.md](../../stt/screening.md#what-these-numbers-do-not-say),
"A panel can reward a style"); its loss is one of many uncorrected comparisons
on single runs, and against the matched control the same difference is
−0.01 [−0.17, 0.20], not separable. It is recorded as history, not as
evidence.

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
   noise (speech noise) instead of MUSAN (a prepared babble bank first), or a
   test at the recipe's scale (about 30 000 updates on the recipe's data,
   where the dose sits in the full recipe). A new screening-scale arm costs
   one training run plus a full-suite evaluation — about 81 min of one RTX
   3090 (the arms' training wall times) and about 24 min of transcription (a
   full suite from a cold cache) — judged against the existing matched
   control, which stays valid while the path a run without augmentation takes
   is unchanged — true at the time of writing: the two later training commits,
   `dadd610` and `b57587b`, act only on runs that augment (bank hashes in the
   resume payload, short noise windows dropped); a new matched control is
   needed only if that path changes, at the cost of another such cycle. A new
   arm also already differs from n40's augmentation: the short tail windows of
   the bank manifests are now dropped, so its noise bank is not n40's. The
   recipe-scale test is a run of the recipe itself — about
   20 h of one RTX 3090 at the screening's measured 0.41 updates/s, measured
   with the screen's 16 clips/update on peertube-length clips, plus
   evaluation — and only it answers the recipe's question.
3. **Keep the clause as a default** anyway, because the earlier project's
   recipe used the noise. The measured 40 % dose at screening scale does not
   support it: the single separable noisy-panel gain (n40 on `noisy-fleurs-en`,
   −0.31) is within the run-to-run spread between the two controls on that
   panel and does not pay for the separable clean losses, which both candidates
   show (the larger one resting on a four-clip hallucination gap). Not
   recommended.

Whichever way this is decided, decision 0004 is a record of the starting
recipe and is not edited.
