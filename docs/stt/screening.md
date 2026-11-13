# Per-source screening

Before mixing training data, each source is tried **on its own**: one short
fine-tune of the base model per source, all with the same budget and the same
recipe, then every final checkpoint is scored on the full suite against the base
model served the same way. The question is what one source brings by itself,
measured against the target use of
[D-009](../owner/directives.md): better French, with English kept, including
English spoken with an accent.

This is a screening, not a recipe search. It gives a *feel* for each source; it
does not choose the final mix ([roadmap](../roadmap.md), phase 4) and it is not
the baseline of [decision 0004](../decisions/accepted/0004-parakeet-fr-baseline-recipe.md).

## Reporting helper

The [source-screening helper](../../experiments/source-screening/README.md)
formats retained evaluation runs using the common paired comparison. It takes
explicit baseline and candidate paths, without training a model or sending
transcription requests. Its README gives the environment and exact invocation.
Generated output does not automatically replace the historical tables below;
use the recorded code revisions and inputs when reproducing those results.

## The runs

One config per source, identical except for the dataset, the clip count of a
source smaller than 10 000 clips, and the GPU:
[`configs/train/screen-*.yaml`](../../configs/train/). Two batches, each on two
RTX 5090 in parallel: five sources on 2026-11-08, code at commit `7aede77`, and
eleven on 2026-11-09 ([second batch](#the-second-batch)). The training code is
the same at every commit used.

| Setting | Value | Why |
|---|---|---|
| Base model | `nvidia/parakeet-tdt-0.6b-v3` | the model to improve (D-005) |
| Clips per source | 10 000, or all the clips of a smaller source | the smallest source of the first batch has 11 435 |
| Updates | 2 000 | the same for every source |
| Clips per update | 16 (2 per step × 8 accumulated) | 4 clips of up to 20 s of dense read speech already overflow 32 GB |
| Clip length | 0.5–20 s | the window the sources were prepared with |
| Learning rate | 1e-5 encoder, 3e-5 decoder and joint, constant | the rates of a per-source screening of this model in the earlier private project; constant as in decision 0004 |
| Optimizer | AdamW, weight decay 1e-3, gradient clip 1.0 | |
| Precision | bf16-mixed | |

**Same clips, same updates.** Every source gets 32 000 clip presentations
(2 000 updates × 16 clips, 3.2 passes over 10 000 clips, more over a smaller
source) and the same number of optimizer steps, so a difference in the scores
comes from the data. As a rule of thumb, Adam's step is close to the learning
rate whatever the batch, so 2 000 updates at these rates should move the weights
about as far as 2 000 updates of the earlier screening.

**The rates were not searched.** They are a reasonable fine-tuning rate for this
model, not a tuned one; a source that looks harmful here could look different at
another rate or length.

### The second batch

Run on 2026-11-09. Four sources have fewer than 10 000 clips and train on all of
them, so each clip is seen more often:

| Dataset | Clips | Times each clip is seen | Training time | Peak memory | Commit |
|---|---|---|---|---|---|
| `icsi-en-train` | 8 752 | 3.7 | 53 min | 19.9 GiB | `4658150` |
| `notsofar-en-train` | 2 330 | 13.7 | 56 min | 18.1 GiB | `51871cc` |
| `librispeech-pc-en-train` | 10 000 | 3.2 | 53 min | 17.7 GiB | `4658150` |
| `pxcorpus-fr-train` | 1 912 | 16.7 | 60 min | 15.9 GiB | `51871cc` |
| `african-accented-fr-train` | 10 000 | 3.2 | 51 min | 16.2 GiB | `51871cc` |
| `ami-fr-speakers-en-train` | 3 971 | 8.1 | 51 min | 17.9 GiB | `51871cc` |
| `peertube-fr-train` | 10 000 | 3.2 | 43 min | 22.4 GiB | `77ceaa2` |
| `peertube-edu-fr-train` | 10 000 | 3.2 | 42 min | 21.7 GiB | `f587ed4` |
| `debconf-en-train` | 10 000 | 3.2 | 42 min | 20.4 GiB | `9dc5654` |
| `fosdem-en-train` | 10 000 | 3.2 | 42 min | 20.5 GiB | `fce23d7` |
| `nasa-en-train` | 10 000 | 3.2 | 42 min | 20.1 GiB | `fce23d7` |

`african-accented-fr-train` is screened a second time, on its cleaned text
([card](datasets.md#african-accented-fr-train--african-accented-french)); the
first batch trained on the text as published, and that run's evaluation is kept
as `$AUDIOLAB_RUNS/eval/screen-african-accented-fr-uncleaned-2026-11-08/`.

## Results

Normalized WER on the full suite (hallucination rate for `nonspeech`). The base
column is the base model served by `audiolab serve`, exactly like the
fine-tunes, in %; each other cell is the change against it in points, with its
95 % confidence interval, and **bold** when the interval excludes zero. Negative
is better. "servicenow en-fr" and "en-fr-ca" are the panels
`servicenow-codeswitched-en-fr` and `servicenow-codeswitched-en-fr-ca`. The
mls-fr intervals rest on fewer than 20 groups, and `audiolab compare` marks them
approximate.

### First batch (2026-11-08)

The African-accented FR column is the run on the uncleaned text.

| panel | base | Granary | VoxPopuli FR | MLS FR | African-accented FR | LibriSpeech EN |
|---|---|---|---|---|---|---|
| fleurs-fr | 4.96 | +0.10 [−0.12, +0.31] | −0.21 [−0.45, +0.02] | **+0.97** [+0.55, +1.44] | **+11.45** [+10.23, +12.73] | **+7.32** [+6.00, +8.75] |
| mls-fr | 5.61 | **−0.26** [−0.54, −0.02] | −0.15 [−0.37, +0.05] | −0.14 [−0.50, +0.34] | **+12.86** [+11.80, +13.72] | **+51.54** [+49.48, +53.30] |
| servicenow en-fr | 13.26 | −1.10 [−2.32, +0.09] | −1.05 [−2.23, +0.11] | +0.90 [−1.03, +3.05] | **+23.74** [+20.77, +26.82] | **+24.82** [+22.14, +27.42] |
| servicenow en-fr-ca | 15.18 | **−1.71** [−3.35, −0.25] | −0.66 [−2.15, +1.06] | −0.51 [−2.53, +1.44] | **+20.59** [+17.69, +23.66] | **+32.89** [+29.79, +35.92] |
| cs-fleurs-fr-en | 16.27 | −0.90 [−2.03, +0.13] | **−0.98** [−1.96, −0.07] | −0.37 [−1.57, +0.71] | **+10.52** [+8.87, +12.11] | **+27.98** [+25.96, +30.05] |
| voxpopuli-en-nonnative | 20.44 | +0.09 [−0.52, +0.98] | **+4.73** [+3.37, +6.61] | **+2.59** [+0.95, +4.98] | **+5.02** [+3.16, +8.04] | **+3.10** [+2.23, +4.34] |
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

### Second batch (2026-11-09)

The African-accented FR column is the run on the cleaned text.

| panel | base | ICSI EN | NOTSOFAR EN | LibriSpeech-PC EN | PxCorpus FR | African-accented FR | AMI FR speakers | PeerTube FR | PeerTube EDU FR | DebConf EN | FOSDEM EN | NASA EN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fleurs-fr | 4.96 | **+0.29** [+0.05, +0.55] | **+0.26** [+0.02, +0.51] | **−0.20** [−0.40, −0.02] | **+2.02** [+1.47, +2.61] | **+0.94** [+0.52, +1.39] | +0.08 [−0.17, +0.35] | −0.18 [−0.41, +0.06] | −0.18 [−0.44, +0.10] | −0.15 [−0.38, +0.07] | −0.15 [−0.35, +0.04] | **−0.37** [−0.59, −0.15] |
| mls-fr | 5.61 | −0.21 [−0.55, +0.05] | **+0.58** [+0.02, +1.20] | **−0.30** [−0.59, −0.02] | **+0.95** [+0.39, +1.46] | +0.28 [−0.17, +0.70] | **+2.73** [+1.41, +4.36] | **−0.57** [−1.12, −0.14] | **−0.59** [−1.13, −0.20] | **−0.35** [−0.82, −0.00] | **−0.43** [−0.97, −0.05] | −0.36 [−0.96, +0.09] |
| servicenow en-fr | 13.26 | +0.55 [−0.84, +1.94] | +0.62 [−0.63, +1.92] | −1.29 [−2.88, +0.20] | **+1.58** [+0.20, +3.04] | **+10.51** [+7.71, +13.39] | **+6.08** [+4.20, +8.06] | **−2.45** [−4.14, −0.93] | −0.98 [−2.42, +0.46] | **−1.37** [−2.43, −0.35] | −1.03 [−2.18, +0.05] | **+1.93** [+0.52, +3.38] |
| servicenow en-fr-ca | 15.18 | +0.30 [−1.31, +1.89] | −0.60 [−1.98, +0.69] | **−2.13** [−3.81, −0.53] | −1.26 [−3.13, +0.51] | **+5.33** [+2.26, +8.65] | **+12.87** [+8.31, +17.66] | **−3.65** [−5.63, −1.82] | **−3.14** [−5.00, −1.44] | **−2.51** [−4.39, −0.68] | **−2.75** [−4.67, −1.01] | −0.39 [−2.17, +1.23] |
| cs-fleurs-fr-en | 16.27 | **−2.13** [−3.06, −1.28] | −0.64 [−1.71, +0.35] | **−1.89** [−2.97, −0.95] | **−1.28** [−2.39, −0.28] | **+1.59** [+0.35, +2.75] | **+3.23** [+1.63, +4.99] | **−2.10** [−3.19, −1.13] | **−2.10** [−3.23, −1.12] | **−1.06** [−2.12, −0.11] | **−1.46** [−2.58, −0.43] | **−2.20** [−3.33, −1.17] |
| voxpopuli-en-nonnative | 20.44 | **−6.16** [−7.46, −5.34] | **−5.13** [−6.33, −4.36] | **−1.61** [−2.64, −1.03] | −0.16 [−1.83, +3.33] | **+5.14** [+3.50, +7.37] | **−3.66** [−4.82, −2.13] | **−6.63** [−7.69, −5.70] | **−6.01** [−7.02, −5.01] | **−7.41** [−8.71, −6.26] | **−7.37** [−8.77, −6.21] | **−3.94** [−5.48, −2.46] |
| fleurs-en | 5.24 | −0.20 [−0.51, +0.14] | +0.09 [−0.23, +0.41] | −0.09 [−0.29, +0.12] | **+0.66** [+0.20, +1.18] | +0.24 [−0.15, +0.63] | +0.13 [−0.21, +0.45] | **−0.31** [−0.54, −0.08] | **−0.38** [−0.64, −0.14] | **−0.42** [−0.67, −0.18] | **−0.38** [−0.66, −0.12] | **+0.61** [+0.11, +1.11] |
| librispeech-en | 2.04 | −0.01 [−0.20, +0.18] | +0.13 [−0.08, +0.32] | −0.16 [−0.35, +0.01] | +0.13 [−0.13, +0.38] | +0.10 [−0.12, +0.34] | +0.01 [−0.13, +0.15] | −0.09 [−0.25, +0.07] | −0.14 [−0.34, +0.04] | −0.04 [−0.23, +0.17] | −0.06 [−0.26, +0.14] | **+1.16** [+0.66, +1.76] |
| nonspeech (hallucination) | 5.00 | +0.71 [−1.79, +3.21] | +2.14 [−0.71, +5.00] | +0.71 [−1.07, +2.86] | −1.79 [−4.64, +1.07] | **−3.93** [−6.79, −1.43] | +0.36 [−2.50, +2.86] | **−3.57** [−6.07, −1.07] | **−2.86** [−5.36, −0.71] | −1.79 [−3.93, +0.00] | +0.00 [−2.50, +2.50] | **+5.36** [+2.50, +8.57] |

Formatted WER, absolute:

| panel | base | ICSI EN | NOTSOFAR EN | LibriSpeech-PC EN | PxCorpus FR | African-accented FR | AMI FR speakers | PeerTube FR | PeerTube EDU FR | DebConf EN | FOSDEM EN | NASA EN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fleurs-fr | 8.85 | 9.86 | 9.35 | 8.74 | 22.49 | 21.23 | 9.20 | 8.61 | 8.39 | 8.59 | 8.89 | 8.89 |
| servicenow en-fr | 16.15 | 18.17 | 16.15 | 15.22 | 28.76 | 37.36 | 19.63 | 14.40 | 15.96 | 14.98 | 15.34 | 20.45 |
| servicenow en-fr-ca | 18.25 | 23.50 | 17.42 | 16.19 | 19.76 | 32.67 | 29.27 | 15.62 | 16.13 | 17.10 | 16.62 | 21.79 |
| cs-fleurs-fr-en | 20.87 | 20.35 | 20.09 | 19.14 | 26.47 | 32.22 | 23.79 | 18.99 | 18.98 | 19.89 | 19.46 | 19.62 |
| fleurs-en | 11.21 | 11.17 | 11.17 | 11.10 | 23.16 | 19.85 | 11.42 | 10.40 | 10.33 | 10.29 | 10.42 | 11.91 |

### What each source does alone

- **Granary** is the only source with no measurable harm anywhere. It gains a
  little on the Québec code-switched panel and, by an approximate interval, on
  read French (MLS), and keeps English and formatting intact.
- **VoxPopuli FR** is neutral to slightly better on French, including the
  code-switched panels, but costs **4.7 points on English spoken by non-native
  speakers**, the panel closest to it (same Parliament, other speakers). It does
  not answer in French there; it drops words: 22 of the 600 clips come back
  empty, against none for the base model, and more answers are cut short.
  (Its speakers are non-native, but mostly not French: see the panel's
  [card](datasets.md#voxpopuli-en-nonnative--english-spoken-by-non-native-speakers-in-the-european-parliament).)
- **MLS FR** hurts the main French panel (+0.97) and non-native English, and does
  not even improve its own corpus's panel. Its lowercase, unpunctuated text
  doubles the formatted WER on fleurs-fr. This matches what the earlier project
  found about MLS.
- **African-accented FR**: most of its damage is orthographic. Its transcripts
  write a space after an apostrophe (`j' ai`, `d' accord`), the model learns it,
  and the scorer counts two words. Re-scoring its answers with that space
  removed (a diagnostic, not the official score) leaves fleurs-fr at +1.03
  [+0.55, +1.54], mls-fr at +0.46 [−0.05, +0.91], fleurs-en at +0.46
  [+0.12, +0.82], and still a real cost on code-switched speech
  (servicenow en-fr-ca +7.66, en-fr +12.25, cs-fleurs-fr-en +2.06) and on
  non-native English (+5.03, with 46 empty answers out of 600). Its text was
  cleaned after this run (apostrophes and spacing); the second batch trains on
  the cleaned text.
- **LibriSpeech EN** is unusable as it is prepared: its transcripts are all
  uppercase, the model learns to write in capitals (`HE HOPED THERE WOULD BE
  STEW FOR DINNER`), which this tokenizer spells poorly, and it gets worse even
  on LibriSpeech. Alone it also loses French, as an English-only fine-tune
  should. The cased and punctuated version of the same audio is
  LibriSpeech-PC EN, in the second batch.

Second batch:

- **LibriSpeech-PC EN**, the same audio with casing and punctuation, gains a
  little almost everywhere and harms nothing: cs-fleurs-fr-en −1.89, servicenow
  en-fr-ca −2.13, mls-fr −0.30, fleurs-fr −0.20, and a better formatted WER on
  all five panels. The uppercase text alone made LibriSpeech EN unusable.
- **ICSI EN** gains on non-native English (−6.16, mostly style:
  [below](#what-the-voxpopuli-en-nonnative-gains-are)) and on cs-fleurs-fr-en
  (−2.13), and costs a little on fleurs-fr (+0.29). Its formatted WER is worse on
  the French panels (servicenow en-fr-ca 18.25 → 23.50), which the normalized
  view does not show; the cause is not analysed.
- **NOTSOFAR EN** gains on non-native English (−5.13, mostly style) and costs a
  little on fleurs-fr (+0.26) and mls-fr (+0.58); nothing else changes
  measurably.
- **AMI FR speakers** gains on non-native English (−3.66, not analysed for
  style) but harms French and code-switching: servicenow en-fr-ca +12.87, en-fr
  +6.08, cs-fleurs-fr-en +3.23, mls-fr +2.73.
- **PxCorpus FR** harms French and English (fleurs-fr +2.02, mls-fr +0.95,
  fleurs-en +0.66, servicenow en-fr +1.58) and gains only on cs-fleurs-fr-en
  (−1.28). Its lowercase, unpunctuated text more than doubles the formatted WER
  of fleurs-fr (8.85 → 22.49) and fleurs-en.
- **NOTSOFAR, AMI and PxCorpus** see each clip 13.7, 8.1 and 16.7 times, against
  3.2 for a source of 10 000 clips: their harm may come from the repetition
  rather than from the corpus.
- **African-accented FR, cleaned text**: the cleaning removed most of the
  damage (fleurs-fr +11.45 → +0.94, mls-fr +12.86 → +0.28, cs-fleurs-fr-en
  +10.52 → +1.59), close to what the diagnostic above predicted. It still harms
  the code-switched panels (servicenow en-fr +10.51, en-fr-ca +5.33) and
  non-native English (+5.14), and its lowercase, unpunctuated text more than
  doubles the formatted WER of fleurs-fr (8.85 → 21.23).
- **PeerTube FR** (Capitole du Libre talks labelled by Qwen3-ASR where Whisper
  agrees: [card](datasets.md#peertube-fr-train--capitole-du-libre-talks-in-french))
  is the first source that improves every panel: separably on seven of the
  nine, by intervals that include zero on fleurs-fr (−0.18) and librispeech-en
  (−0.09). It and DebConf EN are the only sources separably better on all three
  code-switched panels (servicenow en-fr-ca −3.65, en-fr −2.45, cs-fleurs-fr-en
  −2.10), the panels closest to the target use of D-009, and its formatted WER
  is better on all five panels (servicenow en-fr-ca 18.25 → 15.62). Its −6.63 on
  non-native English is of the same size as ICSI's, and likely of the same kind.
- **PeerTube EDU FR, DebConf EN and FOSDEM EN**, the other three talk sources
  labelled by Qwen3-ASR where Whisper agrees, do what PeerTube FR does: no point
  estimate on any panel is worse than the base (FOSDEM leaves the hallucination
  rate unchanged, +0.00), so no harm shows anywhere. All four talk sources gain
  separably on cs-fleurs-fr-en (−1.06 to −2.10), servicenow en-fr-ca (−2.51 to
  −3.65), mls-fr (−0.35 to −0.59) and fleurs-en (−0.31 to −0.42), and PeerTube
  FR (−2.45) and DebConf (−1.37) on servicenow en-fr as well. The two
  English sources gain on mls-fr and on the code-switched panels too, by less
  on cs-fleurs-fr-en (−1.06 and −1.46 against −2.10 for both French ones): the
  gain does not follow the language of the source. Their formatted WER is better
  than the base on every panel but FOSDEM's fleurs-fr (8.85 → 8.89). Their −6.0
  to −7.4 on non-native English is likely style, as for ICSI
  ([below](#what-the-voxpopuli-en-nonnative-gains-are)).
- **NASA EN** (podcasts labelled from NASA's own near-verbatim transcripts:
  [card](datasets.md#nasa-en-train--nasas-podcasts-labelled-from-nasas-own-transcripts))
  is mixed. It gains on cs-fleurs-fr-en (−2.20), non-native English (−3.94) and
  fleurs-fr (−0.37 [−0.59, −0.15], the largest separable gain of any source on
  that panel), and harms librispeech-en (+1.16), fleurs-en (+0.61), servicenow
  en-fr (+1.93) and the hallucination rate on non-speech (5.0 % → 10.4 %); its
  formatted WER is worse on fleurs-en and on both servicenow panels (en-fr
  16.15 → 20.45). Most of its harm on the English and code-switched panels is
  a defect of its text. NASA writes every apostrophe as a typographic `’`, and
  its quotes and dashes the same way (`“ ” — –`); the tokenizer of this model has no piece for them and no
  byte fallback, so the model learns to write its unknown token, which the
  answers show as `⁇` (`I don ⁇ t know`) and the scorer reads as a word break.
  The run writes it in 96 of the 600 librispeech-en answers, 69 of the 515 of
  fleurs-en and 96 of the 238 of servicenow en-fr; the base model in 3 answers
  of the whole suite. A diagnostic re-scoring (not the official score) that reads
  `⁇` between two words as an apostrophe leaves librispeech-en at +0.00
  [−0.21, +0.22], fleurs-en at −0.48 [−0.79, −0.17], servicenow en-fr at −0.53
  [−1.79, +0.75] and en-fr-ca at −0.78 [−2.61, +0.83], with cs-fleurs-fr-en at
  −2.35, non-native English at −5.48 and fleurs-fr unchanged; the formatted WER
  of fleurs-en falls to 10.39, and both servicenow panels stay worse (en-fr
  17.12, en-fr-ca 21.22). The hallucinations do not move: the run's 29 answers
  on non-speech are short phrases ("Yeah.", "So they use the two microphones.") over animal,
  human and object sounds. Sound a podcast plays under the speech without the
  transcript noting it (music beds, jingles) is one possible cause, not
  analysed: the card's rules keep clips away from noted sounds only. The
  training labels are now written in the characters the base model can write
  ([Training sources](datasets.md#training-sources)) and `nasa-en-train` was
  prepared again. Its re-screen (`screen-nasa-en-plain`, 2026-11-13, the same
  budget and recipe, scored on the suite of 2026-11-13 against its base run)
  shows the text defect was its English harm: no answer of its purely English
  panels holds the unknown token (10 of its 7 712 answers hold one, all on
  French or code-switched panels; the base holds none, the run above 592 of
  4 116), librispeech-en is not separable from the base (−0.07 [−0.26, +0.11]),
  fleurs-en improves (−0.67 [−0.95, −0.39]) and servicenow en-fr is not
  separable (−1.05 [−2.37, +0.32]); the gains stay, cs-fleurs-fr-en −2.21
  [−3.25, −1.29], voxpopuli-en-nonnative −4.45 [−5.78, −3.36] and fleurs-fr
  −0.31 [−0.55, −0.08]. On panels the first screen did not score, it gains on
  `nasa-en-test` (5.10 → 2.83) and on peertube-edu-captions-fr (13.94 → 7.44),
  and it harms notsofar-en-eval (+3.69 [1.97, 5.65]). The hallucination rate
  is still above the base on non-speech (5.36 → 8.57, +3.21 [0.36, 6.07]; the
  run above was at 10.36): the label fix did not explain it, and its 24
  non-speech answers are still short conversational fragments ("Yeah.",
  "What do you think?"), now spelled with real apostrophes.
- **A hypothesis, not tested: part of the talks' gain may be the Qwen3-ASR
  label style**, the model learning to write like a stronger ASR model (a kind
  of distillation), since the human-labelled podcast does not give all of it:
  even through the diagnostic, NASA does not gain separably on servicenow
  en-fr-ca or mls-fr, and it raises the hallucination rate. NASA also differs
  from the talks in register, recording and editing, so this screening cannot
  tell the two apart; a test that changes only the labels would, for example the
  same NASA clips labelled as the talks are.

Both batches:

- **Hallucinations**: every French source of the first batch, the cleaned
  African-accented French and both PeerTube sources cut the hallucination rate on
  non-speech from 5.0 % to 1.1–2.5 %; PxCorpus moved it by −1.79, an interval
  that includes zero. No English source lowered it measurably (−1.79 to +2.14),
  and NASA raised it (+5.36). That points at the French data rather than at
  fine-tuning itself, which the first batch alone could not tell.

### What the voxpopuli-en-nonnative gains are

The gains of 5 to 7.4 points of ICSI, NOTSOFAR and the four talk sources on
[`voxpopuli-en-nonnative`](datasets.md#voxpopuli-en-nonnative--english-spoken-by-non-native-speakers-in-the-european-parliament)
are mostly transcription style, not recognition. The base model writes like the
European Parliament's edited report: it drops the chair's "thank you", writes
"Mr President", expands contractions ("it's" → "it is"), smooths repetitions
and drops words at the edges of a clip. The panel's references are raw
verbatim, and a fine-tune on verbatim transcripts unlearns that style.

A diagnostic re-scoring (not the official score, no confidence interval) that
also normalizes contractions, courtesy formulas, near repeats and runs of words
deleted at the edges leaves ICSI at −1.41, NOTSOFAR at −1.59 and
LibriSpeech-PC at −0.17 points: an upper bound of their real gain. Deletions
carry the official gain: in ICSI's run they fall from 8.05 to 3.31 points of
WER. Fillers play no part, since the normalizer drops `uh` and `um`. The runs of
the talk sources, NASA and AMI were not re-scored.

A narrower diagnostic covered every screening run instead, with the official
paired intervals and both sides of a score: the official normalizer plus only
two style rules — English contractions to their expanded form, and the French
elision spacing (`j' ai` → `j'ai`) — applied
([experiments/normalization-review](../../experiments/normalization-review/README.md),
reproducible without a server or a GPU; that README's reproduce command prints
the full tables). The non-native-English gains shrink by 0.4 to 1.0 point and
stay separable (ICSI −6.16 → −5.47, the talk sources −6.01 to −7.41 → −5.17 to
−6.48, AMI −3.66 → −2.93, NASA −3.94 → −3.62), the uncleaned African run's
elision damage is largely removed as the apostrophe diagnostic above predicted
(servicenow en-fr +23.74 → +12.57), and one conclusion moves: NASA's harm on
servicenow en-fr, separable at +1.93 [+0.52, +3.38] officially, is not separable
at +1.35 [0.00, +2.74] under the diagnostic. That flip is a knife-edge: the
diagnostic's lower bound is exactly 0.0, and of its 10 000 bootstrap draws 15
land exactly on zero and 243 fall below it, so the 2.5th percentile sits on that
atom. It says the diagnostic removes NASA's separability on that panel, not that
there is a large effect. Whether the two rules become the official normalizer of
the next campaign is
[proposed decision 0013](../decisions/proposed/0013-normalizer-v2-contractions.md);
every v1 result stays as published until it is accepted.

ICSI's −2.13 on cs-fleurs-fr-en is real but concentrated: the base model
collapses on some long code-switched clips (it translates them into English or
cuts them short), ICSI's run less often, and the top 5 % of items carry 90 % of
the gain.

### What the base model already saw

No source made the main French panel better by more than a hair: the intervals
on fleurs-fr that exclude zero are LibriSpeech-PC's −0.20 [−0.40, −0.02] and
NASA's −0.37 [−0.59, −0.15], two English sources, and the best French one is
VoxPopuli FR's −0.21 [−0.45, +0.02]. One likely reason is that the base model
has seen most of this data already. Its
[model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) says it was
trained on about 660 000 hours of Granary, its main training set, and 10 000
hours of human-transcribed speech, and cites among its sources YODAS and
YouTube-Commons (where `granary-fr-train` comes from), MOSEL (pseudo-labels over
VoxPopuli audio), MLS, FLEURS and CoVoST 2. The card does not list the exact
corpora per language, so the overlap is likely, not checked. Of the five sources
of the first batch, only African-accented French is not among those cited.

Two consequences:

- **Gains in French need data the base model has not seen**: the target use of
  D-009 (a developer dictating, French with English words and technical terms)
  rather than more of the same corpora. The candidates are in the
  [source register](sources.md). The four talk sources, the first of that kind
  screened and none of them among the corpora the card cites, all gain on the
  code-switched panels; four results agree with this, they do not prove it.
- **Granary as an anchor in a mix.** Because Granary is close to what the model
  learned, and harms nothing alone, it could hold the scores in a mix: a new
  source that improves one panel and degrades another would be trained together
  with a share of Granary, so that the model keeps what it knew. That is a
  hypothesis for the mix phase; this screening does not test it.

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
- **Fewer clips, more repetition.** A source smaller than 10 000 clips is seen
  more often (up to 16.7 times for PxCorpus, [second
  batch](#the-second-batch)): its score mixes what the corpus brings with what
  repeating it costs, and this screening cannot separate the two.
- **Text style travels.** MLS, the African-accented French set and PxCorpus are
  lowercase without punctuation, LibriSpeech is uppercase: a model trained on
  them alone can unlearn casing and punctuation. The two LibriSpeech arms
  isolate it: the same audio harms every speech panel with uppercase text and
  none with casing and punctuation. Read the formatted WER, not only the
  normalized one.
- **So do characters.** A character the tokenizer cannot write is learned as its
  unknown token: NASA's typographic apostrophes account for most of its harm on
  the English and code-switched panels ([above](#what-each-source-does-alone)), which says little about its
  speech or its transcripts.
- **Model labels or human labels.** The four talk sources are labelled by
  models, and how many of their labels are wrong is not measured. The one
  human-labelled source of the same register, NASA, differs from them in more
  than its labels, so the screening does not say what model labels cost or bring.
- **A panel can reward a style.** `voxpopuli-en-nonnative` has verbatim
  references and the base model writes like an edited report, so a source with
  verbatim transcripts gains there without recognizing more words
  ([above](#what-the-voxpopuli-en-nonnative-gains-are)). Its gains do not measure
  non-native English until the style is separated from the errors.
- **In-domain advantage.** A source scored on a panel of the same corpus (MLS
  train and the MLS panel, LibriSpeech train and the LibriSpeech panel) gains
  from the domain match, not only from the data quality; no speaker is shared.
- **One run per source.** No seed variance is measured; small differences are
  not a ranking.
