# Datasets

One card per dataset `audiolab data fetch` can prepare. Everything is prepared
into `$AUDIOLAB_DATA` as `manifest.jsonl` plus 16 kHz mono 16-bit WAV files;
nothing here is committed, because datasets are not ours to redistribute
([D-013](../owner/directives.md)). `audiolab data list` prints the same names.
Which of these are used for evaluation, and every other source considered with
its status and license, is in [the source register](sources.md) (D-018).

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
| `voxpopuli-en-french-accent` | en | CC0-1.0 | `en_accented` test, accent `en_fr` | 737 | 27 | 2.4 | 259 MiB |
| `servicenow-codeswitched-en-fr` | fr (en) | no license: evaluation only | test | 298 | 2 | 0.6 | 70 MiB |
| `servicenow-codeswitched-en-fr-ca` | fr (en) | no license: evaluation only | test | 188 | 2 | 0.4 | 42 MiB |
| `cs-fleurs-fr-en` | fr (en) | CC-BY-NC-4.0 | read/test + xtts/test1 | 983 | 332 | 2.9 | 319 MiB |

`fr (en)` means French is the language the model is told and that the normalizer
applies, while every clip of the set also holds English words: those are the two
code-switched datasets, which exist to measure exactly that.

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
| `nonspeech` | 60 / 60 | 280 / 280 | 20 / 20 |
| `voxpopuli-en-french-accent` | 100 / 22 | 600 / 22 | 50 / 5 |
| `servicenow-codeswitched-en-fr` | 100 / 1 | 151 / 1 | 147 / 1 |
| `servicenow-codeswitched-en-fr-ca` | 96 / 1 | 96 / 1 | 92 / 1 |
| `cs-fleurs-fr-en` | 100 / 100 | 600 / 266 | 207 / 66 |

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
| `voxpopuli-en-french-accent` | `facebook/voxpopuli` | revision `42f01879c780b4a2e90ec0b4f616c2ece526e4f1` | sha256 of the two `en_accented/test-*.parquet` shards (`src/audiolab/data/voxpopuli.py`) |
| `servicenow-codeswitched-en-fr`, `-en-fr-ca` | `ServiceNow-AI/asr_codeswitched` | revision `82bf4b49499bcaa6f4cf6a67bebfab3ee2affcbb` | sha256 of each configuration's Parquet file (`src/audiolab/data/codeswitched.py`) |
| `cs-fleurs-fr-en` | `byan/cs-fleurs` | revision `0cdbf166c5517ae4b6eb1c54248522eedec53017` | sha256 of each `metadata.jsonl` in the code, and of every WAV from the checksum the Hub publishes (`src/audiolab/data/csfleurs.py`) |

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
- **Panels**: drawn kind by kind, so every panel holds all three. `fast` has
  10 silences, 10 noises and 40 sounds; `full` 40, 40 and 200; `sealed` 10
  silences and 10 noises. ESC-50 is never sealed: it is public, so a held-back
  copy of it would prove little. The first panels, built on 2026-11-07, sealed
  all 50 silences, so no run could score a silence; they were rebuilt on
  2026-11-08 before any score was published, and the new sealed clips are all
  taken from the old sealed panel, so none of them was ever scored.
- **Why it is here**: a model that hallucinates on silence or noise is unusable
  even when its WER looks good, and this is cheap to check.
- **Levels**: the noise is scaled to a target **RMS** of -60 to -30 dBFS, which
  is what a level in dBFS normally means; scaling by the peak instead would put
  the clips about 10 dB below the level their name claims.

## `voxpopuli-en-french-accent` — English with a French accent

- **Source**: `facebook/voxpopuli`, config `en_accented`, its one `test` split
  (`test-0000{0,1}-of-00002.parquet`), narrowed to the rows whose `accent` is
  `en_fr`. The other fourteen accents of the config are downloaded with the same
  two shards and left out.
- **License**: CC0-1.0 for the data. The same repository publishes code and
  models under CC-BY-NC; neither is used here.
- **Text**: the corpus's own transcript of what was said in the European
  Parliament, e.g. `So this means the quite relevant change`. It has sentence
  case and proper nouns but **no punctuation** (seven of the 737 hold a mark of
  their own), so the report's `formatted` view is `n/a` and only `normalized`
  and `strict` are computed. Every kept clip is flagged as a gold transcript and
  preparation stops if one is not.
- **Language**: `en` for every clip, both for the API and for the normalizer.
- **Group**: the speaker (`group_kind: speaker`), 27 of them. One accent is
  spoken by several people: their errors are not independent, and the sizes are
  uneven — the three largest speakers hold 486 of the 737 clips, the smallest
  ones a single clip. The sealed panel keeps 5 whole speakers back, `full` and
  `fast` draw on the other 22.
- **Why it is here**: English with a French accent is one of the four cases of
  the target use ([D-009](../owner/directives.md)) and no other panel here says
  what an accent costs. It is the only set of that case acquired here; the
  register keeps [GLOBE V2](sources.md#english-technical-and-french-accented-english)
  as a further candidate.
- **Known limits**: 2.4 hours of read parliamentary speech, one accent out of
  many, and the corpus's own transcripts hold errors that no model can avoid
  repeating. Its accent label is a claim by the corpus, and this project has not
  checked that a speaker's French is their first language. The models here were
  served with `language: en`, so a model that switches language on its own is
  not measured by this panel.

## `servicenow-codeswitched-en-fr` and `servicenow-codeswitched-en-fr-ca` — synthetic IT-support code-switching

- **Source**: `ServiceNow-AI/asr_codeswitched`, one Parquet file per
  configuration, `en_fr` (298 clips) and `en_fr_ca` (188 clips), both `test`.
  The two configurations are prepared as two datasets: Québec French says `cell`
  where French says `portable`, and a reader should be able to compare the two
  numbers side by side.
- **License**: **the card carries none.** The owner approved the use of this set
  for evaluation on 2026-11-08 (proposed decision 0011): it is never trained on,
  the repository never redistributes it, and nothing about its text or audio
  leaves the machine. Its scores may be published.
- **Audio**: synthetic. The corpus writes a sentence and renders it with a
  text-to-speech voice; each configuration has two of them (`F-Marine` and
  `M-Denis Landrieu` for `en_fr`, `F-Amelie` and `M-Felix Tabarnak` for
  `en_fr_ca`). The point of the set is the words and the switch, not the voice.
- **Text**: the corpus's transcript of the sentence, casing and punctuation
  kept, so the `formatted` view applies: `J'ai terminé ma demande ID 9708297,
  but I still don't see the Admin role assigned.` The source labels every word
  with its language (`words`, `word_languages`), which is what places this set
  closest to what a French developer dictates; the panels score the sentence,
  they do not check the switches word by word.
- **Language**: `fr` for every clip, for the API and for the normalizer. The
  corpus frames every utterance in French and embeds English in it, but the
  frame is French and not always the majority: by the source's own word labels,
  about a third of the `en_fr` utterances hold more English words than French
  ones. The model is told `fr`, as a French dictation that happens to contain
  English words would be, and a model that answers in French here is doing what
  was asked.
- **Group**: the voice (`group_kind: speaker`), two per configuration. The
  sealed panel therefore takes one voice whole and `full` and `fast` score the
  other: `en_fr` 100 / 151 / 147 clips, `en_fr_ca` 96 / 96 / 92. `fast` cannot be
  shorter than the one voice it is drawn from, so on `en_fr_ca` it is as long as
  `full`. Every panel of both datasets carries the report's `few groups (n < 20)`
  caution: their intervals describe one voice.
- **Why it is here**: the vocabulary and the register of the target use — a
  ticket, a VPN, an update, a mailbox — with the switch inside the sentence,
  which no other panel has, and a Québec variety to compare against it.
- **Known limits**: synthetic voices, two per configuration, so a score here
  says nothing about a real speaker switching language; the sentences are
  written to be read rather than thought and dictated; and the intervals rest on
  one voice per panel.

## `cs-fleurs-fr-en` — code-switched FLEURS sentences

- **Source**: `byan/cs-fleurs`, the two test subsets that hold a French-English
  pair: `read/test` (307 clips, read by one person) and `xtts/test1` (676 clips,
  up to three voices of a generative text-to-speech model). Each subset is a
  `metadata.jsonl` plus one WAV per clip. The `mms/test` subset is left out: its
  references are the corpus's own word-level translations and read like machine
  output (`il aurait une 20s. d'années`), so a score on them would measure the
  corpus as much as the model.
- **License**: CC-BY-NC-4.0, so **testing only**
  ([decision 0009](../decisions/accepted/0009-non-commercial-datasets-for-testing-only.md)): the
  scores may be published, the dataset is never trained on and never
  redistributed.
- **Audio**: the read subset is a person reading, the `xtts` one is synthetic
  speech. Both are kept and `category` says which is which, because the same
  sentences read by a person and spoken by a voice are two different questions,
  and `source_split` names the subset.
- **Text**: the code-switched sentence, casing and punctuation kept: the switch
  happens inside the sentence rather than at its edge, at word and clause level.
  The corpus generates those sentences from FLEURS sentences (see *Known
  overlap*), so the switch is written, not spoken by someone who chose to switch.
- **Language**: `fr` for every clip, for the API and for the normalizer; the
  corpus's own language field for these rows is `fra-eng`.
- **Group**: the FLEURS sentence the clip was built from (`group_kind: sentence`),
  read out of the clip id: `fra_1660_MW` and `1660_fra-spk0_sample0` are both
  sentence 1660. 332 sentences over 983 clips. The subsets have one reader and
  three synthetic voices, too few groups to resample, so the group is the
  sentence, which is also what the two subsets share.
- **Why it is here**: the switch inside the sentence, with ordinary French
  around it, and a real voice next to a synthetic one for the same sentences.
- **Known limits**: the code-switched text is generated, so its French is
  translationese and its switch positions are not ones a speaker would have
  chosen; one reader for the real audio; the synthetic subset dominates the
  panel by size.

## Known overlap with what models were trained on

These splits are test sets, but a test set is not automatically data the model
has never seen. **This project has not measured that for any model, and no number
in a report is corrected for it.** What follows is the list of questions to ask a
model card before trusting a difference of a few points, not a finding:

- **Was the model trained on the test split?** The read-speech corpora are
  public and widely reused, so a model trained on "LibriSpeech" or "MLS" may
  have seen the very utterances scored here. Our panels come from the same
  public splits as every other evaluation of these models, so the question is
  not avoidable, only answerable: [FLEURS `test`][fleurs], [MLS `test`][mls] and
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
- **`cs-fleurs-fr-en` scores sentences `fleurs-fr` scores too.** Its 332
  sentences are FLEURS test sentences with part of their words replaced by
  English: the corpus builds its code-switched text out of FLEURS and keeps
  FLEURS's splits. The audio is not the same — the read subset is a fresh
  recording and the rest is synthetic speech — so no clip is counted twice and
  no group is shared (`cs-fleurs-fr-en:1660` and `fleurs-fr:1660` are different
  groups in different datasets). What is shared is the text, and a model that
  knows FLEURS's sentences is helped twice.
- **The ServiceNow sentences are machine-written and its voices synthetic.** No
  model can have been trained on the audio unless it was trained on this corpus
  itself, but its sentences were written by a language model, so their
  distribution is that of generated text and not of a paper or a bug report.
- **`voxpopuli-en-french-accent` is a test set by construction.** The accented
  English config has no train split; VoxPopuli's other audio is public and
  reused like the rest, so its English and French panels would carry the same
  question, not this one.
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
- FLEURS, MLS, LibriSpeech and the read subset of CS-FLEURS are read speech.
  Spontaneous speech, telephone audio and overlapping speech are not covered by
  any dataset here.
- **Dictation itself is still not covered.** The two code-switched sets put a
  switch inside a sentence, but their text is written first and read or spoken
  afterwards (ServiceNow's voices are synthetic, CS-FLEURS's sentences are
  generated from FLEURS ones). Someone thinking out loud, switching language
  mid-sentence and correcting themselves is not represented; the vocabulary is
  closer to the target use than the register is.
- **Accents are covered once, French ones only.** `voxpopuli-en-french-accent`
  is English with a French accent; English with another accent, and French with
  a foreign accent, are not covered, and the accent label is the corpus's own
  claim rather than something checked here.
- The panels are samples of these test splits and were chosen by us, so absolute
  numbers describe the panels, not the whole datasets.
