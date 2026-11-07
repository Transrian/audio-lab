# Datasets

One card per dataset `audiolab data fetch` can prepare. Everything is prepared
into `$AUDIOLAB_DATA` as `manifest.jsonl` plus 16 kHz mono 16-bit WAV files;
nothing here is committed, because datasets are not ours to redistribute
([D-013](../owner/directives.md)). `audiolab data list` prints the same names.

Prepared sizes below are what one run of `audiolab data fetch` produced on the
owner's machine, so a reader can tell how much disk a phase needs. Audio is
re-encoded to 16 kHz mono; the source archives are larger and stay in the cache
(`$AUDIOLAB_CACHE`, default `.cache/audiolab`).

| dataset | language | license | split | clips | groups | hours | prepared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fleurs-fr` | fr | CC-BY-4.0 | test | 676 | 332 | 2.0 | 217 MiB |
| `fleurs-en` | en | CC-BY-4.0 | test | 647 | 350 | 1.8 | 197 MiB |
| `mls-fr` | fr | CC-BY-4.0 | test | 2426 | 18 | 10.1 | 1.1 GiB |
| `librispeech-en` | en | CC-BY-4.0 | test-clean | 2620 | 40 | 5.4 | 601 MiB |
| `nonspeech` | none | see below | esc50 + generated | 300 | 300 | 0.6 | 63 MiB |

## Panels and their groups

A **group** is the unit the bootstrap resamples: utterances of one group share a
speaker, a recording or a text, so their errors are not independent. Groups are
also what the sealed panel holds back whole. How many of them each panel has is
part of how precise its numbers can be:

| dataset | `fast` | `full` | `sealed` |
| --- | --- | --- | --- |
| `fleurs-fr` | 100 clips / 100 groups | 533 / 266 | 143 / 66 |
| `fleurs-en` | 100 / 100 | 515 / 280 | 132 / 70 |
| `mls-fr` | 100 / 14 | 600 / 14 | 468 / 4 |
| `librispeech-en` | 100 / 32 | 600 / 32 | 492 / 8 |
| `nonspeech` | 60 / 60 | 240 / 240 | 60 / 60 |

Group counts are the ones the panels depend on, so they are checked against the
prepared data by ``audiolab data panels --check``; the clip counts are checked by
``audiolab data fetch``.

## Frozen sources

Every source is fetched at a pinned revision and verified against a checksum
that lives in the code, which is the authoritative home of these numbers: a
download that does not match fails loudly instead of quietly changing the
dataset under a frozen panel.

| dataset | source | pinned at | verified against |
| --- | --- | --- | --- |
| `fleurs-fr`, `fleurs-en` | `google/fleurs` | revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | sha256 of `test.tsv` and `test.tar.gz`, per language (`src/audiolab/data/fleurs.py`) |
| `mls-fr` | `facebook/multilingual_librispeech` | revision `2e83e61823b4c47dcbcb1980bb88601274127609` | sha256 published by the Hub for every `audio/*.tar.gz` (`src/audiolab/data/mls.py`) |
| `librispeech-en` | OpenSLR SLR12 | `test-clean.tar.gz` | sha256 of the archive (`src/audiolab/data/librispeech.py`) |
| `nonspeech` | generated here | fixed seed per clip | nothing to verify: the same seed rebuilds the same samples (`src/audiolab/data/nonspeech.py`) |
| `nonspeech` | `karolpiczak/ESC-50` | commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6` | sha256 of `meta/esc50.csv` (`src/audiolab/data/esc50.py`) |

A cached file is trusted once it has been verified: re-preparing a dataset does
not download it again, so the check happens when the file arrives.

## `fleurs-fr`, `fleurs-en` — read Wikipedia sentences

- **Source**: `google/fleurs`, `data/{fr_fr,en_us}/test.tsv` and
  `data/{fr_fr,en_us}/audio/test.tar.gz` on Hugging Face.
- **License**: CC-BY-4.0.
- **Text**: the raw transcription, which has casing and punctuation, so the
  `formatted` view is computed. The table has no header line: sentence id, file
  name, raw text, normalized text, character tokens, duration, gender. One
  sentence is read by several readers, so the sentence id repeats and the file
  name identifies the recording.
- **Group**: the sentence (`group_kind: sentence`): FLEURS publishes no speaker
  identity, and the recordings of one sentence share their text and difficulty.
  The file name is the clip id.
- **Why it is here**: short, clean, read speech in the two languages of phase 1,
  and the only French set with a half-decent size that is clearly licensed.

## `mls-fr` — read audiobook French

- **Source**: `facebook/multilingual_librispeech`,
  `data/mls_french/test/{audio_filenames.txt,segments.txt,transcripts.txt}` and
  `data/mls_french/test/audio/*.tar.gz`. The repositories also serve the split
  as Parquet; that is a generated copy of these files, so the files themselves
  are used.
- **License**: CC-BY-4.0.
- **Text**: the transcript, lowercase and unpunctuated, so only the `normalized`
  and `strict` views apply.
- **Group**: the speaker (`group_kind: speaker`), 18 of them in this split. The
  speaker is carried by the utterance id (`<speaker>_<book>_<utterance>`), not by
  `segments.txt`, whose first column is the archive.org URL of the book.
- **Why it is here**: the French set that looks most like the target use —
  read, continuous speech and several thousand utterances, with the speaker in
  the utterance id (see below).
- **Speakers, and what it costs**: the 2426 utterances belong to 18 speakers, so
  the panels do not hold the same number of them. The sealed panel takes whole
  speakers first: it is 4 speakers (468 clips, 4 groups), while `full` and `fast`
  use the other 14 (600 and 100 clips, 14 groups each). A French interval from
  this dataset is therefore built on 4 to 14 groups, not on thousands of
  independent utterances, and the report says so (`few groups (n < 20): interval
  approximate`).

## `librispeech-en` — read audiobook English

- **Source**: `test-clean.tar.gz` from OpenSLR (SLR12), the standard
  LibriSpeech test split.
- **License**: CC-BY-4.0.
- **Text**: the chapter transcript, uppercase and unpunctuated.
- **Group**: the speaker (`group_kind: speaker`), 40 of them here. The sealed
  panel takes 8 of them (492 clips) and `full` and `fast` the other 32 (600 and
  100 clips).

## `nonspeech` — the hallucination battery

Three kinds of clip that must stay silent, 300 in total:

| kind | clips | category | how it is made |
| --- | --- | --- | --- |
| digital silence | 50 | `silence` | zeros, 2 to 20 s drawn from a fixed seed |
| low-level noise | 50 | `noise` | white or pink noise with a random level and seed |
| everyday sounds | 200 | `objects/…`, `nature/…`, `animals/…`, `human/…` | four ESC-50 clips per class, 50 classes |

- **Source**: the first two kinds are generated by audiolab from a fixed seed, so
  they are reproducible; the third is ESC-50 (`meta/esc50.csv` and
  `audio/<file>.wav` from the ESC-50 repository).
- **License**: generated clips have no license to honour; ESC-50 is
  CC-BY-NC-3.0 and is used for testing only, which
  [decision 0009](../decisions/accepted/0009-non-commercial-datasets-for-testing-only.md)
  allows.
- **Reference text**: empty. What is measured is what a model invents.
- **Group**: each clip is its own group, so a bootstrap over groups is a
  bootstrap over clips.
- **Why it is here**: a model that hallucinates on silence or noise is unusable
  even when its WER looks good, and this is cheap to check.
- **Levels**: the noise is scaled to a target **RMS** of -60 to -30 dBFS, which
  is what a level in dBFS normally means; scaling by the peak instead would put
  the clips about 10 dB below the level their name claims.

## Known overlap with what models were trained on

These splits are test sets, but a test set is not automatically data the model
has never seen. **This project has not measured that for any model, and no number
in a report is corrected for it.** What follows is the list of questions to ask a
model card before trusting a difference of a few points, not a finding:

- **Was the model trained on the test split?** All four corpora are public and
  widely reused, so a model trained on "LibriSpeech" or "MLS" may have seen the
  very utterances scored here. Our panels come from the same public splits as
  every other evaluation of these models, so the question is not avoidable, only
  answerable: [FLEURS `test`][fleurs], [MLS `test`][mls] and
  [LibriSpeech `test-clean`][librispeech] say where the data comes from, and
  [arXiv:2505.22251][contamination] studies this overlap for the LibriSpeech
  test sets in a large language-model training corpus.
- **Is the number too good?** `librispeech-en` is the standard English test
  split, and a WER far below the published range of the same model family is a
  reason to check the model card rather than a result to report.
- **`mls-fr` is the likeliest to be memorised.** Its test utterances are
  audiobook French, and audiobooks are the most common source of read-speech
  training material, so a model is more likely to have met this text than the
  Wikipedia sentences of FLEURS.
- **`nonspeech` too.** ESC-50 is a standard audio-classification corpus and can
  be part of an audio model's training data like any other public set; the
  generated silence and noise cannot be.
- **No panel is a defence against contamination.** These points change no
  number: they say which differences are worth being careful about.

[fleurs]: https://huggingface.co/datasets/google/fleurs
[mls]: https://huggingface.co/datasets/facebook/multilingual_librispeech
[librispeech]: https://www.openslr.org/12
[contamination]: https://arxiv.org/abs/2505.22251

### Known gaps

- **Music is not covered.** The battery has silence, noise and ESC-50 sounds;
  music and singing are the cases where hallucinations are most likely. Named
  as a gap for phase 1a rather than half-measured.
- FLEURS and LibriSpeech are read speech. Spontaneous speech, telephone audio
  and overlapping speech are not covered by any dataset here.
- The panels are samples of these test splits and were chosen by us, so absolute
  numbers describe the panels, not the whole datasets.
