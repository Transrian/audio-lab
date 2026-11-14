# 0015 — Noise augmentation in the baseline recipe

- Scope: stt/parakeet-fr
- Derived from: decision 0004 (the "noise added to about 40 % of training samples" clause, never implemented until now), plan 02
- Decided by: proposed by an agent, 2026-11-14; awaiting owner review

## Finding

Decision 0004's recipe always included noise on about 40 % of the training
samples; every run in this repository so far omitted it. Plan 02 measured that
clause directly: two runs, identical to the `screen-peertube-fr` control in
base model, clips (same seed, same order — the sampled-clip hashes match),
updates and optimizer, adding (a) MUSAN noise on 40 % of the clips at 0–20 dB
(`noise-peertube-fr-n40`) and (b) that plus room reverb on 20 % of the clips
(`noise-peertube-fr-n40-r20`), each evaluated against the control on the three
noisy panels and the control's nine clean panels.

Result (paired, normalized WER; full tables in [screening.md](../../stt/screening.md),
"Noise augmentation"):

- **no noisy panel improved with a 95 % interval excluding zero** for either
  dose (deltas from −0.15 to +0.14, all intervals containing zero);
- **n40 worsened non-native English** (voxpopuli-en-nonnative) by
  +0.22 [0.02, 0.45] — the plan's decision rule therefore classifies it
  **reject**;
- n40r20 is classified **inconclusive**: no significant noisy gain, no
  significant clean loss (its only significant clean effect is an
  improvement, cs-fleurs-fr-en, −0.40 [−0.63, −0.19]).

The rule written before the numbers was applied as written. The 40 % dose is
measured and, as is, not supported.

## Question

Does the baseline recipe keep or drop noise augmentation?

## Options

1. **Drop the noise clause** from the baseline recipe (0004's other settings
   stand). Cheapest; leaves the robustness question for a later, targeted
   test (stronger dose, different SNR range, or babble noise).
2. **Run the follow-up the inconclusive branch names first**: 80 % of the
   clips (`noise_prob: 0.8`), same banks and range — one more train+eval cycle
   (~3 h of 3090 GPU) — and decide with the full dose-response in hand.
3. **Keep the clause as a default** anyway, because the control's non-native
   English cost (0.22 points) is small and the earlier project's recipe used
   the noise. This is against the measured evidence and is not recommended.

The follow-up run (option 2) is staged but not started: it needs the owner's
go. Whichever way this is decided, decision 0004 is a record of the starting
recipe and is not edited.
