# Per-source screening

Before mixing training data, each source is tried **on its own**: one short
fine-tune of the base model per source, all with the same budget and the same
recipe, then every final checkpoint is scored on the full suite against the base
model served the same way. The question is what one source brings by itself,
measured against the goal of the project (D-005): better French, with English
and French-accented English kept.

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
| Learning rate | 1e-5 encoder, 3e-5 decoder and joint, constant | the rates of an earlier per-source screening of this model; constant as in decision 0004 |
| Optimizer | AdamW, weight decay 1e-3, gradient clip 1.0 | |
| Precision | bf16-mixed | |

**Same clips, same updates.** Every source is seen 32 000 times (3.2 passes over
its 10 000 clips) and its weights move by the same number of optimizer steps, so
a difference in the scores comes from the data. Adam's step is close to the
learning rate whatever the batch, so 2 000 updates at these rates move the
weights about as far as 2 000 updates of the earlier screening.

**The rates were not searched.** They are a reasonable fine-tuning rate for this
model, not a tuned one; a source that looks harmful here could look different at
another rate or length.

## Results

To be filled from the evaluation runs.

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
The data was checked first: the base model transcribes the Granary training
clips close to their reference texts. The weights themselves had moved very
little (0.17 % on an attention matrix). The 24 batch normalization layers of the
encoder had not: their running variance had dropped by up to a third. In
training mode these layers update the statistics that inference uses, from each
batch, and a batch of two clips padded to the longer one feeds them padding
frames as if they were audio.

The proof was to put the base model's statistics back into the broken Granary
checkpoint, weights unchanged: fleurs-en went from 98.6 % to 11.6 % and
LibriSpeech from 84.0 % to 2.5 % on the fast suite. The fix keeps those layers in
evaluation mode for the whole run (`FrozenBatchNormStatistics` in
[`finetune.py`](../../src/audiolab/train/finetune.py)); their scale and shift
still train. A 300-update Granary run with the fix landed on or below the base
model on every fast panel (fleurs-en 5.69 % against 5.94 %, mls-fr 5.20 %
against 5.73 %), and every run below was trained with it.

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
