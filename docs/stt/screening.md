# Per-source screening

Before mixing training data, each source is tried **on its own**: one short
fine-tune of the base model per source, all with the same budget and the same
recipe, then every final checkpoint is scored on the full suite against the base
model served the same way. The question is what one source brings by itself,
measured against the target use of
[D-009](../owner/directives.md): better French, with English and
French-accented English kept.

This is a screening, not a recipe search. It gives a *feel* for each source; it
does not choose the final mix ([roadmap](../roadmap.md), phase 4) and it is not
the baseline of [decision 0004](../decisions/accepted/0004-parakeet-fr-baseline-recipe.md).

## The runs

One config per source, identical except for the dataset and the GPU:
[`configs/train/screen-*.yaml`](../../configs/train/). Run on 2026-11-08, two
RTX 5090 in parallel, code at commit `7aede77`.

| Setting | Value | Why |
|---|---|---|
| Base model | `nvidia/parakeet-tdt-0.6b-v3` | the model to improve (D-005) |
| Clips per source | 10 000 | the same for every source; the smallest prepared source has 11 435 |
| Updates | 2 000 | the same for every source |
| Clips per update | 16 (2 per step × 8 accumulated) | 4 clips of up to 20 s of dense read speech already overflow 32 GB |
| Clip length | 0.5–20 s | the window the sources were prepared with |
| Learning rate | 1e-5 encoder, 3e-5 decoder and joint, constant | the rates of a per-source screening of this model in the earlier private project; constant as in decision 0004 |
| Optimizer | AdamW, weight decay 1e-3, gradient clip 1.0 | |
| Precision | bf16-mixed | |

**Same clips, same updates.** Every source gets 32 000 clip presentations
(2 000 updates × 16 clips, 3.2 passes over its 10 000 clips) and the same number
of optimizer steps, so a difference in the scores comes from the data. As a rule
of thumb, Adam's step is close to the learning rate whatever the batch, so 2 000
updates at these rates should move the weights about as far as 2 000 updates of
the earlier screening.

**The rates were not searched.** They are a reasonable fine-tuning rate for this
model, not a tuned one; a source that looks harmful here could look different at
another rate or length.

## Results

Normalized WER on the full suite (hallucination rate for `nonspeech`). The base
column is the base model served by `audiolab serve`, exactly like the five
fine-tunes, in %; each other cell is the change against it in points, with its
95 % confidence interval, and **bold** when the interval excludes zero. Negative
is better. "servicenow en-fr" and "en-fr-ca" are the panels
`servicenow-codeswitched-en-fr` and `servicenow-codeswitched-en-fr-ca`. The
mls-fr intervals rest on fewer than 20 groups, and `audiolab compare` marks them
approximate.

| panel | base | Granary | VoxPopuli FR | MLS FR | African-accented FR | LibriSpeech EN |
|---|---|---|---|---|---|---|
| fleurs-fr | 4.96 | +0.10 [−0.12, +0.31] | −0.21 [−0.45, +0.02] | **+0.97** [+0.55, +1.44] | **+11.45** [+10.23, +12.73] | **+7.32** [+6.00, +8.75] |
| mls-fr | 5.61 | **−0.26** [−0.54, −0.02] | −0.15 [−0.37, +0.05] | −0.14 [−0.50, +0.34] | **+12.86** [+11.80, +13.72] | **+51.54** [+49.48, +53.30] |
| servicenow en-fr | 13.26 | −1.10 [−2.32, +0.09] | −1.05 [−2.23, +0.11] | +0.90 [−1.03, +3.05] | **+23.74** [+20.77, +26.82] | **+24.82** [+22.14, +27.42] |
| servicenow en-fr-ca | 15.18 | **−1.71** [−3.35, −0.25] | −0.66 [−2.15, +1.06] | −0.51 [−2.53, +1.44] | **+20.59** [+17.69, +23.66] | **+32.89** [+29.79, +35.92] |
| cs-fleurs-fr-en | 16.27 | −0.90 [−2.03, +0.13] | **−0.98** [−1.96, −0.07] | −0.37 [−1.57, +0.71] | **+10.52** [+8.87, +12.11] | **+27.98** [+25.96, +30.05] |
| voxpopuli-en-french-accent | 20.44 | +0.09 [−0.52, +0.98] | **+4.73** [+3.37, +6.61] | **+2.59** [+0.95, +4.98] | **+5.02** [+3.16, +8.04] | **+3.10** [+2.23, +4.34] |
| fleurs-en | 5.24 | −0.03 [−0.22, +0.17] | +0.11 [−0.08, +0.30] | +0.35 [−0.03, +0.80] | **+0.43** [+0.09, +0.78] | **+13.20** [+12.18, +14.23] |
| librispeech-en | 2.04 | −0.04 [−0.23, +0.13] | −0.03 [−0.15, +0.10] | −0.01 [−0.24, +0.21] | +0.10 [−0.10, +0.31] | **+2.27** [+1.83, +2.72] |
| nonspeech (hallucination) | 5.00 | **−3.93** [−6.79, −1.43] | **−2.50** [−5.00, −0.36] | **−3.93** [−6.43, −1.79] | **−3.93** [−6.79, −1.43] | −0.36 [−2.86, +2.14] |

Formatted WER (casing and punctuation kept), absolute, on the panels whose
references carry them:

| panel | base | Granary | VoxPopuli FR | MLS FR | African-accented FR | LibriSpeech EN |
|---|---|---|---|---|---|---|
| fleurs-fr | 8.85 | 8.68 | 8.63 | 19.03 | 29.41 | 23.34 |
| servicenow en-fr | 16.15 | 15.81 | 15.21 | 27.89 | 48.80 | 83.00 |
| servicenow en-fr-ca | 18.25 | 17.50 | 18.28 | 26.24 | 44.55 | 84.24 |
| cs-fleurs-fr-en | 20.87 | 19.84 | 19.55 | 29.74 | 39.45 | 72.64 |
| fleurs-en | 11.21 | 11.01 | 11.33 | 15.52 | 18.58 | 93.89 |

Each run trained 2 000 updates in 38 to 54 minutes, at a peak of 16 to 24 GiB.

### What each source does alone

- **Granary** is the only source with no measurable harm anywhere. It gains a
  little on the Québec code-switched panel and, by an approximate interval, on
  read French (MLS), and keeps English and formatting intact.
- **VoxPopuli FR** is neutral to slightly better on French, including the
  code-switched panels, but costs **4.7 points on English spoken with a French
  accent**, the panel closest to it (same Parliament, other speakers). It does
  not answer in French there; it drops words: 22 of the 600 clips come back
  empty, against none for the base model, and more answers are cut short.
- **MLS FR** hurts the main French panel (+0.97) and accented English, and does
  not even improve its own corpus's panel. Its lowercase, unpunctuated text
  doubles the formatted WER on fleurs-fr. This matches what the earlier project
  found about MLS.
- **African-accented FR**: most of its damage is orthographic. Its transcripts
  write a space after an apostrophe (`j' ai`, `d' accord`), the model learns it,
  and the scorer counts two words. Re-scoring its answers with that space
  removed (a diagnostic, not the official score) leaves fleurs-fr at +1.03
  [+0.55, +1.54], mls-fr at +0.46 [−0.05, +0.91], fleurs-en at +0.46
  [+0.12, +0.82], and still a real cost on code-switched speech
  (servicenow en-fr-ca +7.66, en-fr +12.25, cs-fleurs-fr-en +2.06) and on accented
  English (+5.03, with 46 empty answers out of 600). The source would need its
  text fixed before any mix.
- **LibriSpeech EN** is unusable as it is prepared: its transcripts are all
  uppercase, the model learns to write in capitals (`HE HOPED THERE WOULD BE
  STEW FOR DINNER`), which this tokenizer spells poorly, and it gets worse even
  on LibriSpeech. Alone it also loses French, as an English-only fine-tune
  should. A cased and punctuated version of the same audio would be the one to
  test.
- **Hallucinations**: every French source cut the hallucination rate on
  non-speech from 5.0 % to 1.1–2.5 %. Whether that comes from the data or from
  fine-tuning at all, this screening cannot tell.

## The batch normalization bug

The first two runs (Granary, LibriSpeech) came back broken, and the way they
were broken pointed at the cause:

| Normalized WER | base | Granary run | LibriSpeech run |
|---|---|---|---|
| fleurs-en (full) | 5.24 % | 98.61 % (mostly empty answers) | 20.15 % |
| librispeech-en (full) | 2.04 % | 83.98 % | 6.34 % |
| fleurs-fr (full) | 4.96 % | 21.71 % | 35.09 % |

A run trained on LibriSpeech got *worse* on LibriSpeech while its training loss
kept falling: the model was fine in training mode and wrong in evaluation mode.
The data was checked first: the base model transcribes six Granary training
clips close to their reference texts. The weights themselves had moved very
little (0.17 % on an attention matrix). The 24 batch normalization layers of the
encoder had not: their running variance had dropped by up to a third. In
training mode these layers update the statistics that inference uses, from each
batch. The likely reason they drifted is the batch itself: two clips padded to
the longer one, so the padding frames count as audio.

The test was to put the base model's statistics back into the broken Granary
checkpoint, weights unchanged, and score it on the fast suite:

| Normalized WER, fast suite | base | broken Granary run, base statistics restored |
|---|---|---|
| fleurs-en | 5.94 % | 11.62 % |
| librispeech-en | 2.30 % | 2.53 % |
| fleurs-fr | 6.31 % | 6.31 % |
| mls-fr | 5.73 % | 5.65 % |

The statistics explain most of the damage, not all of it: fleurs-en stayed twice
as bad, so the weights trained against the drifting statistics had moved too.
The fix keeps those layers in evaluation mode for the whole run
(`FrozenBatchNormStatistics` in
[`finetune.py`](../../src/audiolab/train/finetune.py)); their scale and shift
still train. A 300-update Granary run with the fix landed on or below the base
model on the five fast panels the base was scored on (fleurs-en 5.69 % against
5.94 %, mls-fr 5.20 % against 5.73 %), and every run in [Results](#results) was
trained with it.

## What these numbers do not say

- **Same clips is not same hours.** Granary clips are short (8.3 h for 12 000
  clips), MLS clips long (49.8 h): at the same clip count, MLS trains on six
  times more audio.
- **Text style travels.** MLS and the African-accented French set are lowercase
  without punctuation, LibriSpeech is uppercase: a model trained on them alone
  can unlearn casing and punctuation. Read the formatted WER, not only the
  normalized one.
- **In-domain advantage.** A source scored on a panel of the same corpus (MLS
  train and the MLS panel, LibriSpeech train and the LibriSpeech panel) gains
  from the domain match, not only from the data quality; no speaker is shared.
- **One run per source.** No seed variance is measured; small differences are
  not a ranking.
