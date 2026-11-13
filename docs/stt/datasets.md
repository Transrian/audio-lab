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
| `voxpopuli-en-nonnative` | en | CC0-1.0 | `en_accented` test, accent `en_fr` | 737 | 27 | 2.4 | 259 MiB |
| `servicenow-codeswitched-en-fr` | fr (en) | no license: evaluation only | test | 298 | 298 | 0.6 | 70 MiB |
| `servicenow-codeswitched-en-fr-ca` | fr (en) | no license: evaluation only | test | 188 | 188 | 0.4 | 42 MiB |
| `cs-fleurs-fr-en` | fr (en) | CC-BY-NC-4.0 | read/test + xtts/test1 | 983 | 332 | 2.9 | 319 MiB |
| `african-accented-fr-test` | fr | Apache-2.0 | test | 515 | 12 | 0.3 | 36 MiB |
| `notsofar-en-eval` | en | CC-BY-4.0 | 240825.1_eval_full_with_GT | 20 454 | 12 | 16.7 | 1.9 GiB |
| `nasa-en-test` | en | public domain (US government work) | listing (60 reserved episodes) | 1505 | 60 | 4.3 | 479 MiB |
| `noisy-fleurs-fr` | fr | CC-BY-4.0 + material, per clip | derived: the source's panels | 676 | 332 | 2.0 | 214 MiB |
| `noisy-fleurs-en` | en | CC-BY-4.0 + material, per clip | derived: the source's panels | 647 | 350 | 1.8 | 195 MiB |
| `noisy-servicenow-codeswitched-en-fr` | fr (en) | no licence: evaluation only + material | derived: the source's panels | 298 | 298 | 0.6 | 70 MiB |
| `nonspeech-music` | none | PD or CC-BY per file, see below | 10 s excerpts of instrumental files | 261 | 39 | 0.7 | 80 MiB |
| `peertube-edu-captions-fr` | fr | CC-BY or CC0-1.0, per video | listing (the 140 talks with a human caption track) | 4 367 | 113 | 9.9 | 1.1 GiB |

`fr (en)` means French is the language the model is told and that the normalizer
applies, while every clip of the set also holds English words: those are the two
code-switched datasets, which exist to measure exactly that.

Datasets whose name ends in `-train` do not belong to that table: they hold
training data, they are never scored, and they have their own section below.

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
| `voxpopuli-en-nonnative` | 100 / 22 | 600 / 22 | 50 / 5 |
| `servicenow-codeswitched-en-fr` | 100 / 100 | 238 / 238 | 60 / 60 |
| `servicenow-codeswitched-en-fr-ca` | 100 / 100 | 150 / 150 | 38 / 38 |
| `cs-fleurs-fr-en` | 100 / 100 | 600 / 266 | 207 / 66 |
| `african-accented-fr-test` | 100 / 10 | 396 / 10 | 119 / 2 |
| `notsofar-en-eval` | 100 / 10 | 600 / 10 | 4223 / 2 |
| `nasa-en-test` | 100 / 48 | 600 / 48 | 320 / 12 |
| `noisy-fleurs-fr` | 100 / 100 | 533 / 266 | 143 / 66 |
| `noisy-fleurs-en` | 100 / 100 | 515 / 280 | 132 / 70 |
| `noisy-servicenow-codeswitched-en-fr` | 100 / 100 | 238 / 238 | 60 / 60 |
| `nonspeech-music` | 30 / 21 | 114 / 38 | 147 / 1 |
| `peertube-edu-captions-fr` | 100 / 90 | 600 / 90 | 920 / 23 |

Group counts are the ones the panels depend on, so they are checked against the
prepared data by ``audiolab data panels --check``; the clip counts are checked by
``audiolab data fetch``.

### Reference runs of the new panels (2026-11-13)

The first full-suite numbers of the campaign, one run per model:
``audiolab eval --model configs/models/<model>.yaml --suite configs/suites/full.yaml``
— 7 712 panel items per run, 0 failed (the Qwen run's records show 15 clips
answered only after a retry: 9 on the second attempt and 6 on the third). The
base model (``parakeet-tdt-0.6b-v3-crispasr``) is served by CrispASR on the
owner's home server, and ``qwen3-asr-1.7b-vllm`` by vLLM on the owner's home
server. Normalized WER in
percent; the two non-speech panels are hallucination rates. The panels of the
first phase 1 report keep their base column in the
[screening](screening.md#results); they reproduce it here (fleurs-fr 4.99 against
4.96, mls-fr 5.51 against 5.61, librispeech-en 2.04 against 2.04), so the two
runs are consistent with that report.

| panel | Parakeet | Qwen3 |
| --- | --- | --- |
| `african-accented-fr-test` | 6.57 | 4.93 |
| `nasa-en-test` | 5.10 | 4.17 |
| `notsofar-en-eval` | 7.29 | 7.40 |
| `noisy-fleurs-en` | 10.11 | 6.89 |
| `noisy-fleurs-fr` | 16.33 | 13.77 |
| `noisy-servicenow-codeswitched-en-fr` | 19.20 | 20.25 |
| `peertube-edu-captions-fr` | 13.94 | 7.78 |
| `nonspeech` (hallucination) | 5.36 % | 16.43 % |
| `nonspeech-music` (hallucination) | 1.75 % | 25.44 % |

Formatted WER, the panels whose references carry casing and punctuation:

| panel | Parakeet | Qwen3 |
| --- | --- | --- |
| `african-accented-fr-test` | 9.72 | 8.16 |
| `nasa-en-test` | 14.65 | 13.41 |
| `notsofar-en-eval` | 15.53 | 20.78 |
| `noisy-fleurs-fr` | 19.16 | 16.56 |
| `noisy-fleurs-en` | 15.62 | 12.34 |
| `noisy-servicenow-codeswitched-en-fr` | 22.31 | 23.76 |
| `peertube-edu-captions-fr` | 19.36 | 15.70 |

Reading of the new panels:

- Noise and reverberation cost 3 to 6 points of normalized WER on both models
  (fleurs-fr 4.99 → 16.33 Parakeet, 4.04 → 13.77 Qwen3; fleurs-en 5.24 → 10.11,
  3.51 → 6.89), and the code-switched set is the hardest condition of all for
  both (13.12 → 19.20, 13.00 → 20.25).
- Real meetings (`notsofar-en-eval`) are at the level of read speech for both
  models (7.29 / 7.40), but the base model answers **empty on 8.5 % of its
  clips** there (51 of 600), an empty-output rate it never shows above 2.4 % on
  any other panel; Qwen3's is 1.5 %.
- The base model stays silent on music (1.75 % hallucination) where Qwen3
  writes words on **25.44 %** of the clips; on the whole non-speech battery the
  gap is 5.36 % against 16.43 %.
- `peertube-edu-captions-fr` is the panel the base model does worst of the new
  speech sets (13.94); Qwen3 is much better on it (7.78) — part of that margin
  is expected, since the set is kept only where Whisper is right, and Qwen
  correlates with Whisper more than Parakeet does.
- `nasa-en-test` is at read-speech level for both (5.10 / 4.17).

The runs are reference numbers of the current suite, not a screening: they say
where each model stands on each new panel before campaign 1's fine-tunes are
trained.

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
| `pxcorpus-fr-train` | Zenodo record 10080490 | `PxCorpus.zip` | md5 published by Zenodo for the file (`src/audiolab/data/pxcorpus.py`) |
| `ami-fr-speakers-en-train` | AMI corpus | `ami_public_manual_1.6.2.zip` and the 42 of the corpus's headset WAVs this source reads | the corpus publishes no checksum: the sha256 of the annotations zip and of each of the 42 recordings is recorded in the code and checked on every download (`src/audiolab/data/ami.py`) |
| `nonspeech` | generated here | fixed seed per clip | nothing to verify: the same seed rebuilds the same samples (`src/audiolab/data/nonspeech.py`) |
| `nonspeech` | `karolpiczak/ESC-50` | commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6` | sha256 of `meta/esc50.csv` (`src/audiolab/data/esc50.py`) |
| `voxpopuli-en-nonnative` | `facebook/voxpopuli` | revision `42f01879c780b4a2e90ec0b4f616c2ece526e4f1` | sha256 of the two `en_accented/test-*.parquet` shards (`src/audiolab/data/voxpopuli.py`) |
| `servicenow-codeswitched-en-fr`, `-en-fr-ca` | `ServiceNow-AI/asr_codeswitched` | revision `82bf4b49499bcaa6f4cf6a67bebfab3ee2affcbb` | sha256 of each configuration's Parquet file (`src/audiolab/data/codeswitched.py`) |
| `cs-fleurs-fr-en` | `byan/cs-fleurs` | revision `0cdbf166c5517ae4b6eb1c54248522eedec53017` | sha256 of each `metadata.jsonl` in the code, and of every WAV from the checksum the Hub publishes (`src/audiolab/data/csfleurs.py`) |
| `granary-fr-train` | `espnet/yodas-granary` | revision `969944574ea3f37890beaf67ea651e160cfaf043` | sha256 published by the Hub for every shard of `data/fr000/asr_only` (`src/audiolab/data/granary.py`) |
| `voxpopuli-fr-train` | `facebook/voxpopuli` | revision `42f01879c780b4a2e90ec0b4f616c2ece526e4f1` | sha256 published by the Hub for every `fr/train-*.parquet` (`src/audiolab/data/voxpopuli_train.py`) |
| `mls-fr-train` | `facebook/multilingual_librispeech` | revision `2e83e61823b4c47dcbcb1980bb88601274127609` | sha256 published by the Hub for the train metadata and every `audio/*.tar.gz` (`src/audiolab/data/mls.py`) |
| `african-accented-fr-train` | OpenSLR SLR57 | `African_Accented_French.tar.gz` | sha256 of the archive, recorded in the code because OpenSLR publishes no checksum for it (`src/audiolab/data/african_french.py`) |
| `african-accented-fr-test` | OpenSLR SLR57 | the same `African_Accented_French.tar.gz` (the test part of it) | sha256 of that same archive (`src/audiolab/data/african_french.py`) |
| `notsofar-en-eval` | `microsoft/NOTSOFAR` | the same revision `ba8fd0f034ce185fe4d24f47e53b4b8194795f07` (`eval_set/240825.1_eval_full_with_GT`) | sha256 published by the Hub for every `close_talk` recording, git blob sha1 for each transcript (`src/audiolab/data/notsofar.py`) |
| `librispeech-en-train` | OpenSLR SLR12 | `train-clean-100.tar.gz` | sha256 of the archive (`src/audiolab/data/librispeech.py`) |
| `librispeech-pc-en-train` | OpenSLR SLR145 | `manifests.tar.gz` | sha256 of the archive, recorded in the code because OpenSLR publishes no checksum for it; the audio is `librispeech-en-train`'s own archive, checked against its sha256 (`src/audiolab/data/librispeech_pc.py`) |

A cached file is trusted once it has been verified: re-preparing a dataset does
not download it again, so the check happens when the file arrives.

## Stopping a preparation and going on

`audiolab data fetch`, `audiolab data fetch-audio` and the labelling of a talk
source can be stopped with Ctrl-C or SIGTERM. The task finishes the step in hand
— the clip it is writing or the row it is reading, the block of a download, the
requests a model pass has in flight (a talk Whisper is hearing ends after its
current 300 s chunk) —
prints `audiolab: <what> stopped; run the same command again to resume` and
exits with code 130; a second Ctrl-C or SIGTERM ends it at once. Running the same
command again goes on:

- A download is written to `<file>.part` in the cache and renamed once its
  checksum matches. A file pinned by a checksum continues from the bytes already
  there (an HTTP range request), and a part that already holds the whole file is
  only checked; an unpinned one, such as a talk's video, starts again from the
  first byte, because nothing would prove that the two halves are one file. A
  file that does not match its checksum is deleted, as before.
- A preparation writes each clip's WAV and then its manifest line to
  `manifest.partial.jsonl` in the dataset directory; `manifest.jsonl` appears
  only when the preparation is complete, so a stopped dataset is never taken
  for a prepared one. The next preparation takes a clip from the partial
  manifest, without decoding it again, only if its WAV is there, every field of
  its line is the one it would get now, and the partial manifest was written by
  the same data code and the same versions of numpy, soundfile and soxr
  (`preparation_key` in `src/audiolab/data/build.py`; the version of `ffmpeg`
  is not part of it); otherwise the clip, or the whole partial manifest, is
  prepared again.
- The labelling of a talk source caches every model answer per talk (see
  `peertube-fr-train` below), so the next run reads through what the stopped
  one did without asking a model again, and asks only for the rest.

On a terminal, a progress bar counts the clips of a preparation (out of the cap
for a training source), the talks read, and the bytes of a download; a log file
gets the usual lines only.

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
  Noisy versions of both panels: [`noisy-fleurs-fr`](#noisy-fleurs-fr--fleurs-french-in-noise-babble-and-reverb)
  and [`noisy-fleurs-en`](#noisy-fleurs-en--fleurs-english-in-noise-babble-and-reverb).

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
  even when its WER looks good, and this is cheap to check. The battery's
  music — a model must stay silent on that too — is the separate set
  [`nonspeech-music`](#nonspeech-music--music-for-the-hallucination-battery).
- **Levels**: the noise is scaled to a target **RMS** of -60 to -30 dBFS, which
  is what a level in dBFS normally means; scaling by the peak instead would put
  the clips about 10 dB below the level their name claims.

## `voxpopuli-en-nonnative` — English spoken by non-native speakers in the European Parliament

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
- **Group**: the speaker (`group_kind: speaker`), 27 of them. One corpus accent
  is spoken by several people: their errors are not independent, and the sizes
  are uneven — the three largest speakers hold 486 of the 737 clips, the smallest
  ones a single clip. The sealed panel keeps 5 whole speakers back, `full` and
  `fast` draw on the other 22.
- **Why it is here**: no other panel here says what reading English costs a
  speaker of another language, which is the accented English of the fourth case
  of the target use ([D-009](../owner/directives.md)); this is the only
  English-as-a-second-language set acquired here, and the
  [register](sources.md#english-technical-and-french-accented-english) lists the
  candidates for the French-accented part of that case.
- **Known limits**: 2.4 hours of read parliamentary speech, one corpus accent out
  of many, and the corpus's own transcripts hold errors that no model can avoid
  repeating. The name of the accent is a claim by the corpus, and this project
  has not checked the first language of any speaker.
- **Overlap with the prepared training source** (2026-11-13):
  `voxpopuli-fr-train` holds no clip of the 27 speakers of this set, but 206 of
  its 307 plenary sessions are in the source as well: 480 of the 737 clips here
  (1.6 h of the 2.4) and 1 956 of the source's 12 000 (4.67 h of 29.0) were
  recorded in the same sittings, in the same rooms and with the same
  microphones. A session is the clip id's prefix before the language tag, and
  the population is the prepared data — the two manifests —, reproducible with
  [`experiments/leakage-audit/sessions.py`](../../experiments/leakage-audit/README.md).
  That is not a text leak — different languages, different
  utterances — but it is the one overlap that shares acoustics with an
  evaluation set; whether it disqualifies those train clips is
  [proposed decision 0014](../decisions/proposed/0014-voxpopuli-shared-sessions.md).
- **Corpus label in doubt** (2026-11-08): matched to the Parliament's own list
  of members, only about 26 of the 737 clips come from French, Luxembourgish or
  French-speaking Belgian members; the largest speakers are from Latvia
  (208 clips), Flanders (171) and Portugal (107). In other accents of the same
  config the speakers do match their nationality. Either the `en_fr` label or
  the speaker ids are wrong for this accent, so until someone listens to a
  sample, the panel is read as English spoken by non-native speakers in the
  European Parliament, not as French-accented English. The models here were
  served with `language: en`, so a model that switches language on its own is
  not measured by this panel.
- **Renamed** (2026-11-08): this dataset and its panels were called
  `voxpopuli-en-french-accent`; the same 737 clips and the same selection rule
  (`accent == en_fr`) are kept under the new name.

## `servicenow-codeswitched-en-fr` and `servicenow-codeswitched-en-fr-ca` — synthetic IT-support code-switching

- **Source**: `ServiceNow-AI/asr_codeswitched`, one Parquet file per
  configuration, `en_fr` (298 clips) and `en_fr_ca` (188 clips), both `test`.
  The two configurations are prepared as two datasets: Québec French says `cell`
  where French says `portable`, and a reader should be able to compare the two
  numbers side by side.
- **License**: **the card carries none.** The owner approved the use of this set
  for evaluation on 2026-11-08 (decision 0011): it is never trained on,
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
- **Group**: the clip (`group_kind: clip`); every sentence is different. The
  voice is the clip's `category` (`voice/<name>`). Two voices are too few groups:
  an interval resampled from two values says nothing, and the sealed panel would
  hold one whole voice. Panels: `en_fr` 100 / 238 / 60 clips, `en_fr_ca`
  100 / 150 / 38 (fast / full / sealed), both voices in each. The intervals treat
  clips as independent, which is optimistic: clips of one voice share its
  rendering.
- **Why it is here**: the vocabulary and the register of the target use — a
  ticket, a VPN, an update, a mailbox — with the switch inside the sentence,
  which no other panel has, and a Québec variety to compare against it.
- **Known limits**: synthetic voices, two per configuration, so a score here
  says nothing about a real speaker switching language; the sentences are
  written to be read rather than thought and dictated; and the intervals ignore
  the voice, so they are narrower than they should be.
  Noisy version of the `en_fr` panel:
  [`noisy-servicenow-codeswitched-en-fr`](#noisy-servicenow-codeswitched-en-fr--code-switched-tts-in-noise-babble-and-reverb).

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

## `african-accented-fr-test` — African accented French, prompted sentences

- **Source**: the `test` part of OpenSLR SLR57, the same archive
  [`african-accented-fr-train`](#african-accented-fr-train--african-accented-french)
  trains on: 515 recordings under `speech/test/ca16/`, one directory per
  speaker, with the part's single transcript `transcripts/test/ca16/prompts.txt`
  (515 lines, one per recording). The part's speakers are Cameroonian, Chadian,
  Congolese and Gabonese. `dev` and `devtest` are in the same archive but are
  not read: they stay untouched as a reserve for a later campaign.
- **License**: Apache-2.0, as the train part.
- **Audio**: 515 prompted sentences, one recording per line of the transcript.
  12 speakers, 2 to 87 clips each; 0.31 h of audio.
- **Text**: the corpus's own sentences, casing and punctuation kept
  (`text_has_casing_punct: true`), with only the corpus's apostrophe spacing and
  its whitespace runs removed, the same cleanup the train text gets. `category`
  is `read` for every clip: the part's one transcript is its prompted sentences.
- **Group**: the speaker's directory (`group_kind: speaker`), 12 groups.
- **No overlap with the train part** (decision 0002): the preparation reads the
  prepared `african-accented-fr-train` manifest and drops a test speaker it
  already holds; 2026-11-12 that overlap was empty, so all 515 clips are in.
  More than a tenth of the part's speakers overlapping stops the preparation.
  The train fetcher keeps the check the other way: its `Take` takes its
  exclusions from this manifest (`evaluation_groups`).
- **Why it is here**: French spoken with an African accent, which no other
  panel measures; [D-009](../owner/directives.md) keeps French first and this is
  the only open-source set that holds it. The train part's known defect (the
  transcripts put a space after the apostrophe, which the model learned) is
  gone from the test text, so the set measures the accent, not the typing.
- **Known limits**: prompted sentences, not spontaneous speech; 12 speakers;
  the panel holds 10 of the 12 groups, so the interval is wide.

## `notsofar-en-eval` — recorded meetings, one close mic per participant

- **Source**: the `eval_set` of the same repository as
  [`notsofar-en-train`](#notsofar-en-train--recorded-meetings-with-one-close-mic-per-participant),
  the subset `240825.1_eval_full_with_GT` under
  `benchmark-datasets/eval_set/`: 129 meetings of recorded meetings with
  human word-by-word transcripts, each with `gt_transcription.json` and the
  close-talking recordings `close_talk/CT_<n>.wav` (16 kHz mono), every file
  checked against the checksum the Hub publishes for the pinned revision. The
  subset `240629.1_eval_small` (80 meetings) holds no transcripts and is not
  read; the `dev_set` is not read either.
- **License**: CC-BY-4.0.
- **Audio**: 129 meetings, 16.7 h. Twelve participants (the corpus's aliases
  Beth, Brad, Chris, Donald, Jenny, John, Linda, Rachel, Ron, Sally, Sarah and
  Tom) attend all of them; 20 454 clips, median 1.4 s, the longest 20 s.
- **Text**: the annotators' words, cased and punctuated
  (`text_has_casing_punct: true`), with the markup treated as in the train
  split: `<ST/>` stays in the row and nothing here reads it, `<FILL*/>` goes
  with its filler, `<PName>` keeps its name, and a row that holds only markup
  or a `<UNKNOWN/>` or `<ISSUE/>` stretch is left out. Every clip has text.
- **Group**: the speaker's alias (`group_kind: speaker`), 12 groups.
- **No overlap with the train split** (decision 0002): the preparation reads
  the prepared `notsofar-en-train` manifest and drops an eval alias it already
  holds — and stops, instead of dropping, if more than half of the eval
  aliases overlapped; on 2026-11-13 that overlap was empty, the eval meetings
  are a different twelve people. The train fetcher keeps the check the other
  way: its `Take` takes its exclusions from this manifest (`evaluation_groups`).
- **Why it is here**: a third corpus of meetings really held (AMI and ICSI
  are the other two), recorded after the base model's data and not in the
  training list its card cites; the close microphones make it a hard,
  spontaneous-speech set.
- **Known limits**: twelve speakers — the split holds no more, so the panel
  the plan expected (600 clips over at least fifteen speakers) cannot be
  drawn from it: the sealed panel holds two whole groups and the full panel
  spans the other ten, sixty clips of each. The challenge publishes a
  thousand simulated hours in the same repository; they are not read.

## `nasa-en-test` — NASA's podcasts, the human-transcribed evaluation set

- **Source**: the same pinned listing as
  [`nasa-en-train`](#nasa-en-train--nasas-podcasts-labelled-from-nasas-own-transcripts)
  (768 episodes, four shows, 2026-11-13), of which 60 are reserved for this set
  by a constant rule: the episodes whose groups a prepared `nasa-en-train`
  (including the `.curly-2026-11-09` re-screening) holds are out, NASA's
  Curious Universe is out, and of the rest the first 60 by
  `sha256("nasa-test-" + episode id)`, at most 15 of a show. The rule is a
  constant, so the reservation is the same on every machine and run. It comes
  out 15 of each of the four remaining shows — 30.2 h of episodes (median
  episode 26 minutes).
- **License**: public domain (US government work), on the same evidence as the
  training set's card.
- **Audio**: 1505 clips, 4.3 h; 3 to 30 clips per episode (median 26, two
  episodes end at the 30-clip share the fetcher gives each episode); median
  clip 9.1 s, the longest 20 s.
- **Text**: NASA's own transcript of each episode, cased and punctuated
  (`text_has_casing_punct: true`), placed in time the same way as the training
  set: Whisper segments the episode, and the transcript's words are aligned to
  them clip by clip (`audiolab.data.aligned_clips`). Speaker labels and notes
  are removed; a clip that would span a change of speaker or a note is left
  out.
- **The edge check is Whisper's, not the base model's** (where this set differs
  from the training set): a clip is kept only when **Whisper** hears the
  label's first and last words at the cut's edges — the model under test may
  not screen its own labels, so Parakeet, which does that for the training
  set, is not asked here. The consequence, stated plainly: **the set is
  biased toward clips Whisper large-v3 hears well at the edges**; a model that
  cuts words differently is not the one this gate favours. The gate was given
  1 705 of the 4 352 the transcripts labelled — at most 30 per episode, the
  share the fetcher allows an episode to take — and it left out 195 of those;
  the 5 clips with a character the base model cannot write fell after the
  gate, leaving the 1 505.
- **`base_wer`**: Parakeet (the base model of the fine-tune) transcribes every
  kept clip once, and its WER against the label is recorded, so a training mix
  can find the clips it gets wrong. It does not vote: the gate is Whisper's.
  The base model gets 942 of the 1505 kept clips wrong (62.6 %); the median
  `base_wer` is 0.038.
- **Group**: the episode page slug (`group_kind: recording`): the clips of one
  episode share its guests and its jargon.
- **No overlap with the training sets** (decision 0002): the reservation
  leaves the 265 episodes of the two prepared training sets out before it
  picks, and the training fetcher in turn reads this set's manifest
  (`evaluation_groups`) and skips the reserved episodes of every preparation
  it makes after the set exists — checked in `tests/test_data_nasa.py`. The
  set was prepared 2026-11-13, after both training preparations, so their 11956
  and 12000 clips hold none of its episodes. The listing of 2026-11-13
  holds no episode of NASA's Curious Universe: the source no longer reads the
  show, and the reservation leaves it out too
  (`EXCLUDED_SHOWS` in `src/audiolab/data/nasa.py`).
- **Why it is here**: conversational technical English with **human**
  transcripts, the closest evaluation side to the target use the campaign had:
  FLEURS and LibriSpeech are read speech, VoxPopuli a different register, and
  the meetings sets a different task.
- **Known limits**: the 43.5 % of candidate clips the transcripts label is the
  alignment's yield, not a defect of the episodes (the training set's was
  41.3 %); five clips were left out because their transcript holds a character
  the base model cannot write; the sets are 60 episodes of four shows, so a
  show's recording conditions are sampled by 15 of its episodes.

## `peertube-edu-captions-fr` — the education ministry's French talks with human captions

- **Source**: the eleven PeerTube instances of the French education ministry
  (`*.apps.education.fr`), the same pinned listing as
  [`peertube-edu-fr-train`](#peertube-edu-fr-train--the-education-ministrys-french-videos)
  (2 595 videos listed 2026-11-08). Each instance's API lists the caption
  tracks of a video; the scan of 2026-11-13 (cached, one request per second per
  instance) found a French track that is **not** machine-generated
  (`automaticallyGenerated: false`) on 140 talks (65.8 h) — 64 of them on
  numerique-educatif, 34 on sciences-technologies, 29 on
  arts-lettres-sciences-humaines, the remaining 13 one to six on six other
  instances (9 in all hold a human caption track; the other two hold none).
  The 1 930 machine-generated tracks are not usable as references and were left
  out.
- **License**: the video's license from the listing, CC-BY or CC0-1.0; the
  captions are assumed to share it. Every kept clip here is CC-BY.
- **Audio**: 4 367 clips, 9.9 h, from 113 of the 139 usable talks (one is in
  the training set and left out; twelve hold captions that yield no clip; the
  rest had every candidate clip rejected by the gates below). 107 talks hit
  the cap of 40 clips; the clip length is 3.0 to 15.0 s (median 7.6 s).
- **Text**: the human caption track, cut into 3–15 s clips that end at a cue
  whose text ends in `.`, `?`, `!` or `…` when possible, and never cross a gap
  longer than 1 s; cased and punctuated (`text_has_casing_punct: true`). The
  text is a reference written by people: no model wrote it, so the set has no
  `base_wer` and no label to second.
- **The gate is Whisper's, and the bias is stated plainly**: a clip is kept
  only when the normalized WER of Whisper large-v3 on the clip against the
  caption text is at most 0.35. That removes condensed subtitles and
  misplaced cues — and **biases the set toward the clips Whisper gets right**,
  so a score here is not independent of Whisper. The 20 490 candidate clips
  had WER deciles 0.00, 0.00, 0.00, 0.00, 0.05, 0.08, 0.12, 0.17, 0.25, 0.41;
  17 967 (87.7 %) met the threshold and 2 523 did not. Of the 17 967 that
  met it, 4 367 became clips; 2 077 hold a character the base model cannot
  write (after the plain-punctuation rule, almost all of them an `&`, a `+`,
  an `=` or a `_`) and 11 523 went beyond the 40-clip share of their talk.
- **Group**: the video's uuid (`group_kind: recording`): the clips of one talk
  share its teacher, its room and its jargon. `category` is `tech` when the
  talk's title matches the keyword list in the fetcher — 240 clips of 6 talks
  (the plan's 15-talk threshold for a separate `tech` report in T9 is not
  met) — and `general` otherwise: technical French is a subpopulation of the
  set, not a separate dataset.
- **No overlap with the training set** (decision 0002): the one talk of this
  set that [`peertube-edu-fr-train`](#peertube-edu-fr-train--the-education-ministrys-french-videos)
  already holds (73 of its clips are in that set) is left out at preparation,
  and the train fetcher in turn reads this set's manifest
  (`evaluation_groups`) and skips its talks in every preparation it makes
  after the set exists — checked in `tests/test_data_peertube_edu_captions.py`.
  The training set was prepared 2026-11-09, before this set, so only that one
  talk is shared, and it is the set's exclusion that removes it.
- **Why it is here**: real French speech close to the target use with
  **human** references — the register no other panel here covers: not read
  speech, not synthetic, not parliamentary, and not pseudo-labelled.
- **Known limits**: 113 talks of the ministry's instances — the digital-education
  catalog, including the five-part "Puissance 4" game-programming course and a
  web-radio series — the 26 talks that yielded no clip (12 with captions too
  short to cut, 14 whose candidate clips all failed a gate) are left out, so a
  talk's absence is not a random sample; the captions are of the teachers' own
  making, with their own timing errors the WER gate only partly removes.

## Noise banks — material for augmentation, not evaluation sets

These two datasets hold the files that the noisy evaluation sets are mixed
with (T4) and that the training augmentations take from (plan 02). They are
not scored: they have no text to measure against and no panels, by
deliberate refusal rather than by omission.

## `musan-noise` — MUSAN's noise files for augmentation

- **Source**: the `musan/noise/` part of
  [MUSAN (OpenSLR 17)](https://www.openslr.org/17/), served from the OpenSLR
  mirrors as `musan.tar.gz` (10.3 GiB; sha256 pinned in the fetcher). The
  `musan/speech/` and `musan/music/` parts are not read here; the music part
  is the material of `nonspeech-music`.
- **License**: CC-BY-4.0 for the corpus; the `free-sound` files are marked
  public domain in the archive, and the `sound-bible` files carry a licence
  per block in `noise/sound-bible/LICENSE` — a block marked anything other
  than public domain or CC BY 3.0 is dropped and counted, not kept.
- **Prepared**: 929 files, 6.2 h, 684 MiB, one recording per group
  (`group_kind: recording`). The 843 `free-sound` files are public domain; the
  86 kept `sound-bible` files hold the block licence the corpus writes for
  them (66 `Attribution 3.0`, i.e. CC-BY-3.0, and 20 `Public Domain`). One
  `sound-bible` file is absent from the corpus's own licence file and is
  dropped and counted, not kept.
- **The split rule**: one constant hash rule decides forever which half of
  the files is for evaluation and which for training
  (`musan_split` in the fetcher): the earlier private project trained on the
  odd half, so the even half — the one no model has seen — is the eval half
  (466 files). A model augmented with training-half noise is then scored
  against eval-half noise.
- **Why it has no panels**: it is material, not an evaluation set; nothing is
  scored on it, and a panel would imply a campaign to run. The noisy sets
  that are scored use it as their noise source.

## `rirs-noises` — room impulse responses: real rooms evaluate, simulated rooms train

- **Source**: [RIRS_NOISES (OpenSLR 28)](https://www.openslr.org/28/),
  `rirs_noises.zip` (1.3 GiB; sha256 pinned in the fetcher), the RIRs named by
  the archive's `rir_list` files. Its `pointsource_noises/` are copies of MUSAN
  files and are never read; the isotropic noises of the real directory, named
  by a `noise_list`, are not RIRs and are never read either.
- **License**: Apache-2.0.
- **Audio**: 60 325 room impulse responses, 19.6 h, one recording per group
  (`group_kind: recording`), no speech, empty text. The 325 real-room
  recordings are eight-microphone measurements kept as their first channel;
  the 60 000 simulated ones come from the three room sets, 20 000 each.
- **The split rule**: the real-room RIRs are the evaluation half, the
  simulated ones the training half, so a model augmented in simulated rooms is
  scored in real rooms, which is the honest direction for a
  noise-robustness claim.
- **Why it has no panels**: as for `musan-noise`: material for the noisy sets
  and for training augmentation, never scored on its own.

## Noisy sets — the same panel clips, degraded acoustically

A derived evaluation set: for every clip of a source's three panels, one
deterministic acoustic mix, nothing else. The texts, the groups, the panel
sizes and the sealing are the source's — a derived panel is the source's panel
with the ids re-prefixed, so sealed stays sealed and a `--check` on either side
agrees. Every choice in a mix is a hash of the clip id (which condition, which
noise file, at which offset, which four babble speakers, which RIR), so a
re-preparation is byte-identical and no model has ever seen a specific mix.
One clip, one condition:

- `noise-snr10` / `noise-snr5` / `noise-snr0`: eval-half MUSAN noise (the
  reserved half) at that SNR, measured on the clip's active speech;
- `babble-snr5`: four other speakers summed under the clip at 5 dB, from the
  source's own language (`mls-fr-train` for French, `librispeech-en-train` for
  English, alternating for the code-switched set) — four different
  recordings, never the clip's own group;
- `reverb`: one real-room RIR, direct path first;
- `reverb-noise-snr10`: both, in that order.

The license field is the source clip's licence plus the material's, e.g.
`CC-BY-4.0 (speech) + public domain (noise)`.

## `noisy-fleurs-fr` — FLEURS French in noise, babble and reverb

- **Source**: derived from the `fleurs-fr` panel clips with the material of
  `musan-noise` (eval half), `rirs-noises` (real rooms) and `mls-fr-train`
  (babble). [Card of the source](#fleurs-fr-fleurs-en--read-wikipedia-sentences).
- **License**: per clip, the source's plus the material's — CC-BY-4.0 (speech)
  with public domain (noise), CC-BY-4.0 (babble) and Apache-2.0 (reverb) in
  various combinations, as the field says.
- **Prepared**: 676 clips, 332 groups, 2.0 h, 214 MiB — the full `fleurs-fr`
  panel union (103 `noise-snr10`, 100 `noise-snr5`, 126 `noise-snr0`, 110
  `babble-snr5`, 128 `reverb`, 109 `reverb-noise-snr10`).
- **Panels**: the source's panels re-prefixed — fast 100/100, full 533/266,
  sealed 143/66.

## `noisy-fleurs-en` — FLEURS English in noise, babble and reverb

- **Source**: derived from the `fleurs-en` panel clips with the material of
  `musan-noise` (eval half), `rirs-noises` (real rooms) and `librispeech-en-train`
  (babble). [Card of the source](#fleurs-fr-fleurs-en--read-wikipedia-sentences).
- **License**: per clip, as for `noisy-fleurs-fr`.
- **Prepared**: 647 clips, 350 groups, 1.8 h, 195 MiB (109 `noise-snr10`,
  111 `noise-snr5`, 105 `noise-snr0`, 101 `babble-snr5`, 119 `reverb`,
  102 `reverb-noise-snr10`).
- **Panels**: the source's panels re-prefixed — fast 100/100, full 515/280,
  sealed 132/70.

## `noisy-servicenow-codeswitched-en-fr` — code-switched TTS in noise, babble and reverb

- **Source**: derived from the `servicenow-codeswitched-en-fr` panel clips with
  the material of `musan-noise` (eval half), `rirs-noises` (real rooms) and
  `mls-fr-train` and `librispeech-en-train` (babble, alternating French and
  English speakers). [Card of the source](#servicenow-codeswitched-en-fr-and-servicenow-codeswitched-en-fr-ca--synthetic-it-support-code-switching).
- **License**: the source holds no licence —
  [decision 0011](../decisions/accepted/0011-unlicensed-datasets-for-private-evaluation.md):
  evaluation only, never redistributed. That condition travels with the mix;
  per clip the material's licence is added on top.
- **Prepared**: 298 clips, 298 groups, 0.6 h, 70 MiB (48 `noise-snr10`,
  52 `noise-snr5`, 46 `noise-snr0`, 50 `babble-snr5`, 51 `reverb`,
  51 `reverb-noise-snr10`).
- **Panels**: the source's panels re-prefixed — fast 100/100, full 238/238,
  sealed 60/60.

## `nonspeech-music` — music for the hallucination battery

- **Source**: the `musan/music/` part of
  [MUSAN (OpenSLR 17)](https://www.openslr.org/17/) — the same archive as
  `musan-noise`; only its music part is read here, and the noise part never.
- **License**: per file, from the sub-collection's own `LICENSE`: only blocks
  the corpus marks public domain or CC-BY (any version) are kept — share-alike,
  non-commercial and no-derivatives variants are different licences and are
  dropped, as are files whose `ANNOTATIONS` mark them as having vocals (a sung
  file would hand a hallucinating model words to output). The licence field
  holds the block's wording, e.g. `CC BY 3.0` or
  `Live At Blues Alley by U.S. Army Blues is licensed under a Public Domain
  Mark 1.0 License.`.
- **Prepared**: 261 clips, 39 groups, 0.7 h, 80 MiB — one 10 s excerpt per
  file at a hash-chosen offset, spread over the five sub-collections: rfm 147
  (one sticky CC BY 3.0 header over the list), fma-western-art 48, fma 42,
  hd-classical 19, jamendo 5 (its catalogue is mostly share-alike and
  no-derivatives, so only five of its CC-BY files remain after the vocal
  filter). The 300-file cap was not reached.
- **Group**: the artist where the annotation gives one (`group_kind: artist`),
  else the file (`recording`) — two files by one artist are not independent.
- **Panels**: fast 30, full 114, sealed 147 — the `music` family of the
  non-speech battery, sized on its own so the frozen `nonspeech` panels keep
  checking. The sealed part holds whole artists back, as the speech panels do:
  here that is the entire rfm catalogue, one artist (Kevin MacLeod), which the
  40-clip family budget seals in one go.
- **Why it is here**: the `nonspeech` battery (backlog N-03) had no music, and
  a model that transcribes music is hallucinating: the reference is empty, as
  for the silences and the noises.

## Training sources

Datasets whose name ends in `-train` hold training data. No panel is drawn from
them, no run scores them, and `audiolab data panels` is never run on them: a
source that is both trained on and evaluated stops measuring anything. They exist
for one experiment, the **per-source screening**: fine-tune Parakeet TDT 0.6B v3
once per source, with the same number of clips and the same number of updates, to
see what each source brings by itself. How that screening is run, and what it
found, is in [per-source screening](screening.md); this section says what a
prepared source is.

Every source is prepared the same way, so a difference between two arms of the
screening is a difference between corpora and not between recipes:

- **The source's own train split**, recorded as each clip's `source_split`. No
  evaluation split is read to build a training source.
- **The source's own order**, taken until the cap of `TRAIN_CLIP_CAP` clips
  (`src/audiolab/data/train.py`, 12 000). Nothing is drawn and nothing is
  shuffled, so a pinned revision prepares the same clips twice. The price is that
  the first 12 000 clips of a corpus ordered by speaker or by recording are not a
  uniform sample of it; the group count in the table says what a cap covers.
- **At most `CLIP_CAP_PER_GROUP` clips from one group** (200,
  `src/audiolab/data/train.py`). Several corpora publish their train split one
  speaker after another, and taking the first 12 000 clips in that order spends
  the whole cap on a handful of readers (MLS: 19 speakers, one of them 5 859
  clips). A reader that already holds its share is not read further, and a
  fetcher skips its remaining archives instead of downloading them. The rule
  changes nothing for a source whose groups are already smaller than the share —
  Granary (154 clips for its largest video), SLR57 (156 for its largest speaker)
  and `librispeech-en-train` (165) are below it — and it is what makes the 12 000
  clips of `mls-fr-train` cover 138 readers instead of 19.
- **A clip is one speaker's turn, not one row of an annotation**. Where a
  corpus cuts a speaker at every pause — AMI at each word, ICSI and NOTSOFAR at
  each utterance — consecutive rows of one speaker are joined while at most a
  second of silence separates them, while the row does not end a sentence, and
  while the joined clip stays inside the window; a row that is not a clip (a
  laugh, a word nobody made out, a number read out loud, another speaker's
  microphone) ends the clip being built, so no clip carries audio its own text
  does not describe. The rule is `src/audiolab/data/meeting.py`, shared by the
  three meeting sources, and it is why a "Yeah." of 0.4 s is a clip only when it
  has nothing to join. Ending a clip at a sentence costs speech — ICSI gives 7.7 h
  with that rule and 8.9 h without it, because a short "Yeah." after a sentence is
  dropped as too short instead of being folded into the clip before it — and it is
  kept because AMI is cut that way: the three meeting arms are compared with each
  other, and a rule of its own for one corpus would show in the comparison rather
  than in the corpus.
- **Filters**: 0.5 s to 20 s of audio, text that is not empty, and audio that
  decodes at all — the point where a damaged file would otherwise show up. A clip
  the filters drop is counted in the fetch report and read past, so a source is
  read until the cap or its end, and a source with fewer usable clips than the cap
  is prepared with what it has.
- **Groups**: the speaker wherever the corpus publishes one, the recording
  otherwise, which `group_kind` records. A screening result is a confidence
  interval over groups like every other number here.
- **The text the source publishes**, which each card records. Casing and
  punctuation are not added back: three sources are lowercased or uppercased and
  unpunctuated (MLS, SLR57, LibriSpeech), so their arms teach the model to drop
  what it already writes; VoxPopuli, AMI, ICSI and NOTSOFAR are cased and
  punctuated human text, and Granary is cased, punctuated model output that
  nobody verified. That difference is part of what the screening measures, so
  nothing was rewritten to
  make the arms look alike. Two exceptions. Transport defects of a published
  transcript, which the card of the source names: SLR57 writes a space after
  every apostrophe (`j' ai`), which the training text drops because the model
  learns it and the scorer counts two words. And characters the base model cannot
  write: its tokenizer has no piece for them and no byte fallback, so each one
  trains `<unk>` ("I don ⁇ t know"). Every label is written through `plain_label`
  (`src/audiolab/data/train.py`): `’` and `‘` become `'`; quotation marks (`“ ”
  " « »`), parentheses and brackets are dropped; `;` becomes `,`; an em dash, or a
  dash between spaces, becomes a break between words; an en dash becomes `-`
  (`2014-2020`); `…` becomes `...`. A clip whose label still holds a character
  outside the vocabulary (`+ & @ # ° _`, a letter such as `ş`) is left out and
  counted in the fetch report. The rule dates from 2026-11-09: a source prepared
  earlier keeps its published characters until it is prepared again. The
  prepared sets that predate the rule and still hold unwritable characters
  (measured on 2026-11-10 by
  [`experiments/label-text/check_text.py`](../../experiments/label-text/check_text.py);
  that same run also flagged the two preparations below, since re-made)
  are `debconf-en-train`, `fosdem-en-train`, `granary-fr-train`,
  `peertube-edu-fr-train`, `peertube-fr-train`, `pxcorpus-fr-train` and
  `voxpopuli-fr-train`. Two other preparations were re-made and their pre-rule
  versions archived, and each archive is what its screening numbers came from:

  - `nasa-en-train.curly-2026-11-09` is kept for the
    [NASA re-screening comparison](#nasa-en-train--nasas-podcasts-labelled-from-nasas-own-transcripts);
  - `icsi-en-train.pre-plain-label-2026-11-11` is the frozen input of the
    `icsi-en-train` screening, and `configs/train/screen-icsi-en-train.yaml`
    points at it
    ([card](#icsi-en-train--english-research-meetings-one-close-mic-per-participant)).

  The per-source screening keeps the pre-rule sets unchanged for
  reproducibility, and a re-preparation is what brings a set into the rule.
- **No overlap with the evaluation sets** ([decision
  0002](../decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)),
  below.
- **Larger preparations** go to new directories named after their cap:
  `audiolab data fetch granary-fr-train --clips 40000` prepares
  `granary-fr-train-40k`. The 12 000-clip sets above are what the per-source
  screening drew its runs from (`configs/train/screen-*.yaml`), and they stay
  exactly as they are, so the screening is not scaled and remains
  reproducible. A larger preparation is recorded under its directory's name —
  the clip ids and the groups carry that prefix — reads the source in the same
  order, and reuses the source's cache, so it takes the same clips and the same
  talks first, then reads more. `CLIP_CAP_PER_GROUP` stays 200 at every cap,
  and a source that runs out before the cap is prepared with what it has, as at
  the default.

| dataset | language | license | split | clips | groups | hours | prepared | downloaded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `granary-fr-train` | fr | audio CC-BY-3.0, labels CC-BY-4.0 | asr_only | 12 000 | 2 725 | 8.3 | 0.9 GiB | 1.8 GiB |
| `voxpopuli-fr-train` | fr | CC0-1.0 (data only) | train | 12 000 | 229 | 29.0 | 3.1 GiB | 8.6 GiB |
| `mls-fr-train` | fr | CC-BY-4.0 | train | 12 000 | 138 | 49.8 | 5.3 GiB | 8.6 GiB |
| `african-accented-fr-train` | fr | Apache-2.0 | train | 11 427 | 190 | 13.5 | 1.5 GiB | 1.7 GiB |
| `librispeech-en-train` | en | CC-BY-4.0 | train-clean-100 | 12 000 | 108 | 42.9 | 4.6 GiB | 6.0 GiB |
| `librispeech-pc-en-train` | en | CC-BY-4.0 | train-clean-100 | 12 000 | 119 | 42.8 | 4.6 GiB | 25 MiB, plus the 6.0 GiB archive of `librispeech-en-train` on a machine that does not have it yet |
| `pxcorpus-fr-train` | fr | CC-BY-4.0 | recordings | 1 912 | 54 | 3.7 | 0.4 GiB | 0.4 GiB |
| `ami-fr-speakers-en-train` | en | CC-BY-4.0 | manual | 3 971 | 20 | 4.3 | 0.5 GiB | 2.5 GiB |
| `icsi-en-train` | en | CC-BY-4.0 | mrt | 8 746 | 52 | 7.7 | 0.8 GiB | 4.0 GiB of signals, and 13 GiB of decoded WAV stays in the cache |
| `notsofar-en-train` | en | CC-BY-4.0 | 240825.1_train | 2 330 | 12 | 2.1 | 0.2 GiB | 0.8 GiB |
| `peertube-fr-train` | fr | CC-BY (per video, in the manifest) | listing | 12 000 | 77 | 31.8 | 3.4 GiB | not measured: the videos are deleted once their sound is out, and 7.3 GiB of WAV stays in the cache |
| `peertube-edu-fr-train` | fr | CC-BY / CC0-1.0 (per video, in the manifest) | listing | 12 000 | 113 | 31.2 | 3.4 GiB | not measured: the videos are deleted once their sound is out, and the 119 talks read leave 8.0 GiB of WAV in the cache |
| `debconf-en-train` | en | CC-BY (per video, in the manifest) | listing | 12 000 | 100 | 34.2 | 3.7 GiB | not measured: the videos are deleted once their sound is out, and the 106 talks read leave 7.1 GiB of WAV in the cache |
| `fosdem-en-train` | en | CC-BY-2.0-BE (per talk, in the manifest) | listing | 12 000 | 124 | 35.6 | 3.8 GiB | not measured: the videos are deleted once their sound is out, and the 139 talks heard leave 6.4 GiB of WAV in the cache |
| `nasa-en-train` | en | public domain (US government work) | listing | 11 956 | 265 | 34.6 | 3.7 GiB | not measured: the MP3 files are deleted once their sound is out, and the 265 episodes read leave 15.0 GiB of WAV in the cache |

`prepared` is what the fetch wrote into `$AUDIOLAB_DATA`, `downloaded` what it
read from the network; the cache keeps those bytes, so preparing a source again
costs nothing and a fresh clone downloads all of it again.

### `granary-fr-train` — pseudo-labelled YouTube French

- **Source**: `espnet/yodas-granary`, `data/fr000/asr_only/*.parquet` on Hugging
  Face (the corpus's second pass over French, `fr100` to `fr103`, is not used).
- **License**: audio CC-BY-3.0, labels CC-BY-4.0.
- **Text**: the pseudo-label itself, with casing and punctuation, but written by a
  model: rows are often fragments rather than sentences, and nothing verified
  them.
- **Group**: the recording (`group_kind: recording`) — the YouTube video a row was
  cut from. The corpus publishes no speaker identity, and the rows of one video
  share a speaker, a microphone and a topic. **2 725 videos for 12 000 clips**,
  so this source is a sample of about 2 700 videos rather than of thousands of
  independent utterances; a screening result on it is an interval over 2 725
  groups, and clip-level intervals would be far too narrow.
- **What a cap covers**: the shards in the Hub's order are read until the cap, so
  the preparation covers the beginning of the corpus (98 of 315 shards, 1.8 GiB)
  and the rest was never downloaded.
- **Clips are short**: the median row of this corpus lasts 1.5 s (mean 4.3 s,
  p90 9.2 s, and 13 % of its rows fall outside the 0.5-20 s window), which is what
  8.3 h for 12 000 clips means next to MLS's 14.8 s median. That is the corpus and
  not the shards this preparation happened to read: the rows of the 98 shards read
  and the rows of eight shards sampled further in the split (positions 110 to 314)
  have the same distribution (median 1.5 s and 1.5 s, mean 4.3 s and 3.8 s), and
  the clips written keep exactly the duration distribution of the rows that passed
  the window (mean 2.49 s, p10 0.68 s, median 1.50 s, p90 5.18 s for both), so the
  cap takes rows in order without preferring short ones.
- **Why it is here**: the widest variety of French in the register — real YouTube
  speech, in several registers — and the only pseudo-labelled source of the
  screening, so its arms can compare model labels with human ones.

### `voxpopuli-fr-train` — European Parliament French

- **Source**: `facebook/voxpopuli`, config `fr`, `train-*.parquet`.
- **License**: CC0-1.0 for the data (the corpus's code and models are not used).
- **Text**: the `raw_text` field, which is the human-edited transcript with casing
  and punctuation (the corpus also publishes an unpunctuated `text`, not used).
  The corpus marks its rows as gold or not; all 15 646 rows read were gold, and 562
  of them had no text, so they were passed over.
- **Group**: the speaker (`group_kind: speaker`), 229 of them. VoxPopuli does
  not name the reader of 2 130 rows of this split (13 % of the rows read): its
  `speaker_id` column holds the string `None` for them. A clip whose reader is
  unknown can neither be resampled by reader nor checked against an evaluation
  set, so those rows are left out and counted rather than kept under one made-up
  group.
- **Why it is here**: parliamentary French with human transcripts, a different
  register from the audiobooks and the read Wikipedia sentences, and a source the
  earlier project trained on.
- **What it costs**: 29.0 h for 12 000 clips; the shards are large (about 3.0 GiB
  each), so a cap of 12 000 clips reads three of the thirteen — 8.6 GiB of Parquet.

### `mls-fr-train` — read audiobook French

- **Source**: `facebook/multilingual_librispeech`,
  `data/mls_french/train/{audio_filenames.txt,segments.txt,transcripts.txt}` and
  `data/mls_french/train/audio/*.tar.gz`, read in the published order until the
  cap: 28 479 rows from 184 of the 600 archives, 8.6 GiB in the cache. The file
  list is ordered by book and then by reader — the first entries are all one book
  read by different people — so reading it straight through spends the cap on 19
  readers; the per-group share below is what spreads it, and it left 400 further
  archives of readers that already held their share unread and undownloaded (16
  archives were never reached).
- **License**: CC-BY-4.0.
- **Group**: the speaker (`group_kind: speaker`), carried by the utterance id,
  **138 of them**, at most 200 clips each. A screening result on this source is
  an interval over 138 groups.
- **Text**: lowercase and unpunctuated as the corpus publishes it. **It is also
  damaged in places, and that is the corpus's own defect**: 531 of the 12 000
  clips (4.4 %) hold no accented letter at all, and the loss is concentrated in
  one book — `9061` writes 72 % of its clips that way (`encaiss` for `encaissé`,
  `immdiatement` for `immédiatement`, `rpondit` for `répondit`), while the other
  56 books lose an accent in 1 to 11 % of theirs (names and loan words). Every one
  of the 12 000 texts is byte-identical to its line in the corpus's own
  `transcripts.txt` (checked clip by clip), so nothing here was dropped by the
  preparation; a reader who compares with the evaluation split sees that split
  keeping its accents (98.8 % of its clips hold one), which is a property of the
  published files of the two splits, not of how they are read. Nothing was
  filtered, corrected or replaced: the screening has to train on the source as it
  is, and a model trained on this arm is expected to write `immdiatement`.
- **Why it is here**: the register closest to the target use, read audiobook
  French, with enough volume for a cap — and the source the earlier project
  measured as harmful in a mix ([sources.md](sources.md#used--trained-on-here)),
  which the screening tests again with this source alone. Its clips are the longest
  of the screening (median 14.8 s, at least 10 s).

### `librispeech-pc-en-train` — read audiobook English with restored casing

- **Source**: OpenSLR SLR145 (`manifests.tar.gz`, 25 MiB) for the text, over the
  audio of SLR12's `train-clean-100.tar.gz`, which is the archive
  `librispeech-en-train` prepares. The two sources read the same clips in the
  same order and share one download.
- **License**: CC-BY-4.0, both halves.
- **Text**: the restored text of LibriSpeech-PC — casing and punctuation — in
  place of the corpus's uppercase, unpunctuated transcripts. The restoration
  dropped 2 498 of the split's 28 539 utterances (8.8 %): those clips are left
  out rather than labelled with the original text. It also keeps a chapter title
  in capitals at the start of an utterance's first sentence
  (`THE PEOPLE MARCH He became aware ...`), and a run of three or more words in
  capitals — punctuation stripped from the word, so `MARCH,` counts — matches 144
  of the split's 26 041 rows (0.55 %); the preparation leaves those utterances
  out of the first rows too, because a dictation model should not learn to write
  a heading in capitals. The text is what the restorer wrote, not what the reader
  read: 6 160 of the 12 000 prepared clips begin with a lowercase word, in the
  middle of a sentence. None of them holds a double quote — the raw transcription
  quotes speech in 29 % of the split's rows and the restoration dropped the marks
  — so 4.6 % of the prepared clips carry a quotation mark as a lone apostrophe
  (`'tis`, `'We sound every chord ...`), and 5.1 % of the split's rows do.
- **Group**: the speaker (`group_kind: speaker`), 119 of them in the prepared
  clips.
- **Why it is here**: it is the English arm of the screening with the text the
  other sources lack — and the reason not to expect gains from it. The base
  model very likely trained on this audio with this text: NeMo's ASR Set 3.0
  lists LibriSpeech and keeps punctuation and casing, so this source mostly
  repeats what the model already knows. It is the English guard of a mix, and
  its value in the screening is to test whether mixing in English this way costs
  French, unlike `librispeech-en-train`, whose uppercase text moved every French
  panel by 7 to 52 points.
- **What a cap covers**: the cap was reached after 13 315 rows: 1 243 without
  restored text and 72 with a heading were dropped on the way. Same per-group
  share as `librispeech-en-train`.

### `ami-fr-speakers-en-train` — English meetings spoken by French-L1 participants

- **Source**: the AMI meeting corpus, from its own site: the manual annotations
  (`ami_public_manual_1.6.2.zip`, 23 MB) and the individual headset recordings of
  the meetings it takes, `amicorpus/<meeting>/audio/<meeting>.Headset-N.wav`. The
  audio is served without a login; one preparation downloads 2.5 GiB of it, and
  only for the meetings a French-L1 participant speaks in. The corpus publishes
  no checksum for either part, so the sha256 of the annotations zip and of each
  of the 42 headset recordings this source reads is recorded in
  `src/audiolab/data/ami.py` and checked on download; a recording outside that,
  longer list stops the preparation rather than being taken unverified.
- **License**: CC-BY-4.0, corpus and annotations.
- **Text**: the words the annotators wrote, cased and punctuated, with the
  corpus's own spellings (`'kay`, `uh`), and its spelling of spelled-out letters
  undone: the annotators separate the letters of a spelled word with underscores
  (`T_V_`, `L_C_D_`), which is removed, so the label is `TV` and `LCD`. A manual
  word file also holds `vocalsound` (a laugh, a cough), `disfmarker` (a
  disfluency) and `gap` (a stretch nobody transcribed); none of them carries
  text. All three are dropped as words — they neither join two words nor end a
  sentence — but a `gap` **ends the clip** it falls in: the audio under it is
  speech nobody transcribed, so a clip that spanned it would be labelled with
  text that is not all its audio says. A silence longer than one second ends a
  clip the same way. Most gaps are shorter than that — 3 950 of the corpus's
  5 125 gaps have no length at all — but one ends a clip wherever it falls.
- **Group**: the speaker (`group_kind: speaker`), read from `participants.xml`
  and `meetings.xml`: 20 participants in the prepared clips. Every speaker of a
  meeting is a group of its own, and a clip comes from that speaker's own headset
  recording, so two participants of one meeting never share a clip.
- **What the source is**: of the 41 participants whose native language is
  French, 20 are named by a meeting, in 43 meetings and 69 speaker recordings —
  mostly the IDIAP meetings (IS, IB, IN) and one scenario meeting (ES2007). The
  other 21 appear in `participants.xml` alone: no meeting of `meetings.xml` names
  them, so no word file describes them and they cannot contribute a clip. Their
  meetings are not in the public annotations, which is also why this source is
  smaller than the roughly 10 hours one would expect from the participant list.
  A clip is one sentence of one speaker: cut at sentence-final punctuation, at a
  silence longer than one second, at a gap, or at 20 s. The 69 recordings hold
  12 739 utterances under those rules; the preparation offered 5 489 of them to
  the cap — the rest were never read, because the per-speaker share of 200 clips
  was already full — of which 1 518 were shorter than 0.5 s, and the source is
  3 971 clips of 0.5 s to 20 s: median 2.5 s, 4.3 h in total.
- **What a cap covers**: fewer clips than the cap, and the per-group share is
  what holds it there. The corpus has 20 groups for this source, so the share
  (200 clips of a group) allows 4 000 clips at most: every arm of the screening
  is capped by the same rule, and this source cannot reach 10 000 clips under it.
  Its config sets `clips: 3971` next to the reason.
- **Why it is here**: AMI is spontaneous English spoken by people whose first
  language is French, which is the target use of a dictation model for French
  speakers — the register no other English source of the screening has. It is
  also in the training data of English models, Parakeet among them (NeMo's ASR
  Set 3.0 lists AMI), so the arm is a guard for a mix rather than a source of
  gains, like `librispeech-pc-en-train`.

### `icsi-en-train` — English research meetings, one close mic per participant

- **Source**: the ICSI Meeting Corpus, from the corpus's own site: the manual
  transcripts (`ICSI_original_transcripts.zip`, 3.5 MiB, release 1.0 of 22 July
  2016) and the signals of the meetings it takes,
  `ICSIsignals/SPH/<meeting>/<channel>.sph`. The site serves both over plain HTTP
  without a login, and it publishes no checksum for either, so the sha256 of the
  transcript archive is recorded in `src/audiolab/data/icsi.py` and checked on
  download, and the signals are checked against the count and the sum of samples
  their own SPHERE header declares (below).
- **License**: CC-BY-4.0, corpus and transcripts. Checked on 2026-11-08 at
  <https://groups.inf.ed.ac.uk/ami/icsi/> and
  <https://groups.inf.ed.ac.uk/ami/icsi/license.shtml>, and in the corpus's own
  `LICENCE.txt` inside the NXT archive.
- **Text**: the transcripts of the **MRT** release, not the NXT annotations built
  from it. An MRT segment carries the start and the end time of one utterance by
  one participant, while NXT rebuilds those times from single words and leaves a
  word unplaced when the annotators placed it only at its other end (`We` with a
  start and no end), so an NXT boundary would have to be guessed. The words are
  the annotators' own, cased and punctuated, with their marks of a false start
  (`how much w- how easy`) and their own spellings, except for one mark of how a
  word was written: a token written letter by letter is read as the word it
  spells, the way the AMI source reads its own words — `O_K` is "OK", `S_R_I` is
  "SRI", `T_V_s` is "TVs", and a lone `L_` is the letter "L" — while nothing else
  is touched: `Mm-hmm` stays as the annotators wrote it, and a token that carries
  a letter sequence and something else (`J_RASTA`, `E_commerce`) keeps its
  underscore, because joining it would write a word nobody said (26 tokens in the
  prepared clips).
  The annotation elements between the words are taken out, all the way down:
  `VocalSound` (29 779 of them), `NonVocalSound` (3 708), `Comment` (12 283) and
  `Pause` (7 213) carry no words to transcribe and go with what they hold, while
  `Emphasis` (24 612), `Uncertain`, `Pronounce` and `Foreign` hold words the
  speaker did say and are kept as text — including a word written inside two of
  them (`<Emphasis><Pronounce>places</Pronounce></Emphasis>`). A segment that is
  nothing but a vocal sound cleans to no text, and such a segment is not a clip.
  Where the annotators could not make the words out they wrote `@@` (`So you can
  @@ -`), so a segment holding one is not a clip either: the audio is there and
  the reference is not.
- **Group**: the speaker (`group_kind: speaker`): the 52 participant ids of the
  corpus, each of them a group of its own. A clip is sliced from the close
  microphone the segment's own preamble gives its speaker
  (`<Participant Name="fe066" Channel="chan0"/>`), so two participants of one
  meeting never share a clip. A participant the preamble gives no channel to (25
  of the corpus's 494 participants, in 16 of its 75 meetings) is one the corpus
  says was unmiked — heard on other people's microphones — and their 219 segments
  all say `CloseMic="false"`; a segment that claims the close microphone and has
  no channel stops the preparation instead. A segment whose own `Channel` is not
  the one the preamble gives its speaker (5 of them, in `Bsr001`) is the same
  speech the corpus re-sliced from another microphone, and taking it would
  prepare that speech twice.
- **What the source is**: 75 meetings and 129 813 segments, of which 1 413 are
  not speech on the speaker's own microphone (717 name no participant — a stretch
  of room noise the annotators marked — and 696 say `CloseMic="false"`), 5 241
  are a digit task (in 63 of the meetings the annotators read out digits one at
  a time: `Six three, three three`), 5 were re-sliced from a channel that is not
  the speaker's, 20 072 hold nothing but markup and 745 hold a word nobody made
  out. That leaves 102 337 segments, and consecutive ones of one speaker become
  80 175 stretches, 12 759 of which join two or more segments. The per-speaker
  share of 200 clips is what holds the preparation there: 8.6 % of those segments
  become clips, 8 752 of them (1 462 joining several segments) from 49 of the 75
  meetings, 7.7 h, median 1.9 s; the 2026-11-11 re-preparation of the same
  source holds 8 746 (Prepared, below). The longest clip is 19.98 s and every clip is
  inside the window; 8 segments of the corpus are longer — the longest 95 s, two
  words with the times the annotators gave them — and none of them is a clip. The
  34 speakers that reach their share contribute 200 clips each and the 18 smaller
  ones contribute between 21 and 190.
- **What a cap covers**: fewer clips than the cap, and the per-group share is
  what holds it there. The corpus has 52 participants for this source, so the
  share (200 clips of a group) allows 10 400 clips at most, below the 12 000 of
  the cap: every arm of the screening is capped by the same rule, and this source
  cannot reach 12 000 clips under it. Joining the rows of a speaker changes how
  many clips a corpus gives, not how many a group may have, so the share is still
  what holds this source: at 8 746 clips after the 2026-11-11 re-preparation,
  at 8 752 in the 2026-11-08 set. Its config sets `clips: 8752` next to the
  reason, the size of the set the 2026-11-08 screening ran on.
- **Decoding**: the corpus's signals are shorten-compressed SPHERE and the audio
  readers of this project do not decode them, so `ffmpeg` decodes every channel a
  preparation reads — the only external program any preparation runs — and the
  samples it produced are checked against the count and the checksum the SPHERE
  header of the file itself carries, which is the only checksum the corpus
  offers. `src/audiolab/data/icsi.py` says why the annotation archive is pinned
  by sha256 instead. The 118 channels a full preparation reads are 4.0 GiB of
  downloads, the decoded WAV they leave in the cache is another 13 GiB, and the
  prepared set is 0.8 GiB. The count and the checksum of a channel are checked by
  a function that reads the header alone, so the tests of a preparation run
  without ffmpeg and only one of them runs the real decoder.
- **Prepared** (2026-11-08; re-prepared 2026-11-11): the 2026-11-08 preparation
  of 8 752 clips predates the [label rule](#training-sources) and holds 319
  clips with a character the base model cannot write — 291 with a straight
  quote, 28 with an underscore — so it was moved aside as
  `icsi-en-train.pre-plain-label-2026-11-11` and the source was prepared again on
  2026-11-11 with the current code, the pinned transcript archive and the
  corpus's signals re-downloaded (the download cache held no ICSI entry). The
  291 straight quotes are dropped by `plain_label` — those 291 clips keep
  their ids and their audio — and the 28 clips holding an underscore, which the
  rule cannot write, are left out; 22 stretches of the 15 speakers at their
  200-clip share take their place, the left-out clips no longer counting against
  the share. Of the 8 724 kept labels, 2 776 differ from the archived set:
  2 608 lose a lone dash, 291 lose a straight quote and 21 have a space before a
  comma closed (some clips have more than one). The result is **8 746 clips,
  7.7 h, 52 groups**: the 8 724 clips of the first set that the rule keeps —
  their audio bit-identical to the 2026-11-08 preparation, their labels
  re-normalized — plus the 22 later clips of the freed speakers. The per-source
  screening of 2026-11-08 ran on the first
  set, so its numbers are the archived preparation's: that preparation is the
  frozen input of the screen, and `configs/train/screen-icsi-en-train.yaml`
  names it, with the `clips: 8752` that the re-prepared set no longer holds.
- **Why it is here**: 70 hours of meetings that would have happened anyway, with
  a manual transcript and one microphone per participant: the same kind of data
  as `ami-fr-speakers-en-train` without AMI's scenario meetings and without its
  selection of French-L1 participants, so the two arms say what the selection and
  the corpus contribute. The corpus is not in the list of training sets the base
  model's own card cites (NeMo's ASR Set 3.0 lists AMI, not ICSI), which AMI is.

### `notsofar-en-train` — recorded meetings with one close mic per participant

- **Source**: `microsoft/NOTSOFAR` on Hugging Face, the `train_set` of the
  recorded meetings at the revision `240825.1_train`, pinned by commit
  `ba8fd0f034ce185fe4d24f47e53b4b8194795f07` in `src/audiolab/data/notsofar.py`.
  It is served without a login: 72 meetings under
  `benchmark-datasets/train_set/240825.1_train/MTG/`, each with
  `gt_transcription.json` and the close-talking recordings
  `close_talk/CT_<n>.wav` (16 kHz mono). Every file is checked against the sha256
  (a recording, which the Hub stores through its LFS) or the git blob sha1 (a
  transcript, which it stores as a plain object) that the Hub publishes for that
  revision.
- **License**: CC-BY-4.0, the recorded meetings and their transcripts. Checked on
  2026-11-08 on the dataset's own page and in its `README.md`, which covers the
  recorded data and not only the challenge's code.
- **Text**: what the annotators wrote, cased and punctuated, with the corpus's
  markup taken out: `<ST/>` is its own mark inside a row and nothing here reads
  it, `<FILL/>` and `<FILLlaugh/>` mark a filler and a laugh (the filler goes
  with the markup, where AMI and ICSI keep theirs in the reference), `<BA/>` a
  backchannel, and `<PName> Peter </PName>` wraps a name, which stays as text.
  Two kinds of row are left out rather than half transcribed: the 980 rows of the
  split that hold markup and nothing else, and the 531 rows holding a word nobody
  caught (`<UNKNOWN/>`) or a stretch the annotators had a problem with
  (`<ISSUE/>`), 239 of them counted in both, so 1 511 of the split's 18 178 rows
  never become clips. The text is the corpus's own: 131 of the 2 330 clips start
  with a lowercase letter and 300 do not end with a sentence-final mark, because
  the annotators transcribed the utterance as it came (and joining the rows of
  one speaker joins the text of both), and the preparation rewrites none of it.
  The corpus pads its tags with spaces, so removing one can leave the padding in
  front of a comma or a period (`So Sophie , what`); punctuation is joined to the
  word before it, as the AMI source does, and the words themselves are untouched.
- **Group**: the speaker (`group_kind: speaker`): the alias the corpus gives the
  participant (`Peter`, `Sophie`, `Bert`). 12 aliases over the 72 meetings, and
  each of them is one person, recognizable across the meetings of the corpus,
  which is what makes them groups; a clip is sliced from that participant's own
  close-talking recording, so two participants never share one.
- **What the source is**: the recorded meetings of the challenge — meetings of
  about six minutes, four to eight attendees in one of 30 conference rooms,
  transcribed by hand — not the simulated thousand hours published in the same
  repository, and not the `dev_set`, which is not read for anything; the
  `eval_set` is read as [`notsofar-en-eval`](#notsofar-en-eval--recorded-meetings-one-close-mic-per-participant). The challenge's README counts 237 meetings of it and the dataset's
  page on Hugging Face 315; the revision this source pins holds 72 of them in
  its train split and 12 speakers, and the challenge says 10 of those 12 are also
  in the dev set of the campaign, which is one more reason not to read it here.
  The split is the corpus's `240501.1_train` (108 meetings) without the 36
  meetings that became the dev set of the challenge, and with a faulty device
  removed from three meetings, as the `logs/README.txt` of the split itself says.
  It holds 18 178 rows, 2 014 of which last less than 0.5 s, and 1 511 of them are
  left out before the plan (above); the rest become 12 993 stretches by joining
  the rows of one speaker, 475 of the prepared clips spanning two rows or more.
  No row of the split is longer than 8.7 s, and the longest clip is 19.83 s, so
  the 20 s window holds but is not far away. The share of a group holds the
  preparation: 2 330 clips from 38 of the 72 meetings, 2.1 h, median 1.7 s, and
  10 327 stretches left out because their speaker had its 200 clips already.
  Eleven aliases reach their 200 clips and Walter has 130.
- **What a cap covers**: fewer clips than the cap, and the per-group share is
  what holds it there. The split of 72 meetings names 12 participants, so the
  share (200 clips of a group) allows 2 400 clips at most, far below the 12 000
  of the cap: this source cannot reach it under the rule, and joining the rows of
  a speaker leaves it at 2 330 clips. Its config sets `clips: 2330` next to the
  reason.
- **Why it is here**: meetings that were really held, a few people around a
  table, each with a close-talking microphone, transcribed by hand — the same
  register as AMI and ICSI and a third corpus of it, recorded twenty years later
  and in other rooms. Like ICSI, it is not in the training list the base model's
  card cites. What the split is not is large: 2 330 clips of 12 speakers, so an
  arm presents every clip 13.7 times and a score on it says as much about those
  twelve people as about the corpus.

### `pxcorpus-fr-train` — spoken French drug prescriptions

- **Source**: Zenodo, record 10080490, `PxCorpus.zip` (376 MB): the `PxSLU`
  recordings of the corpus, 1 981 WAV files of 16 kHz mono speech under
  `recordings/<session>/`, with the three parallel text files that describe them
  (`paths.txt`, `metadata.txt`, `seq.in`).
- **License**: CC-BY-4.0.
- **Text**: lowercase and unpunctuated, taken as published except for one mark:
  110 of its lines put a slash inside a word the recording button cut off or that
  the participant repaired (`comp/primés`, `/pone`). The slash is a mark of the
  transcription and not something the speaker said, so the label drops it and
  rejoins the word. Numbers are written as digits (`2,5 milligrammes`). The
  corpus's semantic labels (a spoken-language-understanding task) are not read.
- **Group**: the speaker (`group_kind: speaker`), read from `metadata.txt`. A
  session holds one participant, so the session directory is the speaker id;
  `metadata.txt` names 54 participant ids and the corpus's readme says it has 55
  participants. 54 groups over 1 912 clips is 35 clips per group, above the
  smallest groups of the other sources and far below their thousands.
- **What a cap covers**: the corpus is much smaller than the cap. All 1 981
  recordings are offered; 1 is shorter than 0.5 s and 68 are longer than 20 s, so
  the source is **1 912 clips** — 16.7 passes over the cap's worth of updates
  instead of the 3.2 of a 10 000-clip arm. The screening compares sources at an
  equal number of clips, so this arm cannot match the others on that axis; its
  config says so next to `clips: 1912`, and its numbers are to be read as what
  this corpus brings when it is all a run has, not as a per-clip comparison.
- **Why it is here**: the only source of the screening that is spoken dialogue
  with an application rather than read or prompted speech — short turns, real
  disfluencies (`euh`), drug names and dosages, and 54 participant ids the readme
  counts as 55. Its authors also say they checked the transcripts by hand against
  the recordings.

### `african-accented-fr-train` — African accented French

- **Source**: OpenSLR SLR57, `African_Accented_French/speech/train` with the three
  transcript files under `transcripts/train`.
- **License**: Apache-2.0.
- **Text**: lowercase and unpunctuated, as the corpus publishes it, with two
  transport defects of the published transcript removed: the spaces the corpus
  writes around an apostrophe that stands between two letters (`d' accord`,
  `grand ' mère` — it joins all 5 697 apostrophes of the three train transcript
  files, and an apostrophe that quotes something is left where the annotator put
  it) and the runs of whitespace the Yaoundé part leaves where it dropped
  punctuation (`il y en a plus  il y en a encore`, 4 580 runs in 3 607 of its
  lines). Eight lines of the Yaoundé transcript file are the question put to the
  reader (`etes-vous sûr que c' est lui / elle`), not an answer, and are left
  out: the slash offers a choice (`lui / elle`), so the transcript cannot say
  which option was chosen and its text would not match the audio. All eight lines
  are that one sentence; they are the only lines of the whole corpus that hold
  `" / "`.
  **Casing and punctuation are not restored**: putting them back would take
  another model, and a source that has neither still teaches the base model to
  drop what it writes (the confound below). The cleanup is deterministic and only
  removes spaces, so the same archive prepares the same text. Evaluation
  references are never cleaned like this: the scorer normalizes what the corpus
  published.
- **What the screening measured**: the arm of the 2026-11-08 screening in
  [screening.md](screening.md) trained on the text as published, with the
  apostrophe spacing, and most of its damage was that spacing. The source was
  re-prepared afterwards, so its screen config trains on the cleaned text and its
  old numbers no longer describe it.
- **Group**: the speaker (`group_kind: speaker`), 190 of them, read from the
  directory above the recording. One speaker appears in both the read and the
  answer recordings of the Yaoundé part, so the group is the directory and not the
  subdirectory.
- **What a cap covers**: the whole train part is smaller than the cap. Its 14 394
  recordings hold 11 460 with a transcript, of which 8 are prompting lines and 25
  are longer than 20 s, so the source is 11 427 clips: **the screening's arms do
  not all have the same number of clips**, and this one is 4.8 % short of the
  cap. Nothing was padded or repeated to close the gap.
- **Why it is here**: African accents are otherwise absent from open French ASR
  data. Its transcripts are also the least punctuated of the screening, which is
  the confound below.

### `librispeech-en-train` — read audiobook English

- **Source**: OpenSLR SLR12, `train-clean-100.tar.gz` (6.0 GiB), the corpus's
  clean training audio, read in the archive's own order (speaker, then chapter,
  then utterance) until the cap.
- **License**: CC-BY-4.0.
- **Text**: uppercase and unpunctuated, as the corpus publishes it.
- **Group**: the speaker (`group_kind: speaker`), 108 of them, at most 165 clips
  each, which is below the per-group share, so this source is untouched by it.
  Its clips are long and close to the cap (median 14.1 s, 1.4 s to 17.3 s).
- **Why it is here**: the English arm of the screening — the same register as
  `mls-fr-train` in the other language, and the corpus the English evaluation
  panel comes from, with its test speakers left out.

The next four sources are **talks** rather than corpora: a pinned listing of
videos, read in a seeded order, whose audio is labelled by two speech models where
they agree. Three modules hold that machinery for all of them
(`src/audiolab/data/talks.py`, `talk_listing.py` and `talk_clips.py`), and each
source is a listing module plus one entry in `src/audiolab/data/talk_sources.py`,
which says what the listing must hold, in which language the talks are labelled,
where a talk's audio comes from and what a seeded selection spreads over. What is
specific to a source — its catalogue, its licence, its selection, its size — is in
its card; everything else is the same for the four and is described in the card
of `peertube-fr-train`, the first of them. All of them depart from the rules above
for the same three reasons: no splits (a video platform has none), a seeded order
instead of the catalogue's, and model labels instead of a transcript somebody
published.

### `peertube-fr-train` — Capitole du Libre talks in French

- **Source**: [`videos.capitoledulibre.org`](https://videos.capitoledulibre.org),
  the PeerTube instance the Capitole du Libre conference runs for its own talks.
  `audiolab data list-talks peertube-fr-train` asks its REST API for every video
  the instance hosts itself, tagged French, under licence CC BY (PeerTube licence
  id 1) or CC0 (id 7), and writes the answer to
  [`peertube_listing.json`](../../src/audiolab/data/peertube_listing.json). The
  listing of 2026-11-08 holds **519 talks, 378.9 h**; the shortest is 525 s and
  the median 2 863 s. It applies no length, channel or title filter: every listed
  video is filed under the one account and channel `Capitole du Libre`.
- **License and attribution**: CC BY, as the instance declares it on every listed
  video; PeerTube names the licence ("Attribution") without a version, so the
  manifest records `CC-BY`. No listed video is CC0. Every clip carries its talk's
  `source_url`, `source_title`, `source_author` (the PeerTube account, `Capitole
  du Libre` for all of them) and `license`; the speakers are named in some titles
  only.
- **Language**: French, from the video's own language tag, and the tag is
  required: an untagged video is not listed (the instance has about 50 more CC BY
  videos without one, [sources.md](sources.md#french-variety), row 3 bis).
- **Split and order**: `source_split` is `listing`; the listing is written in uuid
  order and read in a seeded shuffle (`SEED`, `src/audiolab/data/talks.py`), with
  nothing to spread over: one instance, one channel. Audio comes from the smallest
  file the instance still serves for a video, converted to 16 kHz mono WAV; the
  video is deleted once its sound is out.
- **Why it is here**: technical French from free-software conferences — French
  full of English terms and project names, the vocabulary of a developer
  dictating to a coding agent ([D-009](../owner/directives.md)) — spoken live, in
  front of a room, which no published French corpus of the repository has.
- **Group**: the recording (`group_kind: recording`, the video uuid). A talk
  usually has one speaker; the listing names none, and the questions from the
  room add others.
- **Labels**: nobody transcribed these talks, so served models label them, in
  three passes (`src/audiolab/data/talks.py`):
  1. **Whisper large-v3** (`stt-v-whisper-large-v3`, served by vLLM) hears each
     whole talk in five-minute chunks and answers with timestamped segments. A
     chunk's last segment is not trusted, since the chunk may end inside a word:
     the next chunk starts where the segment before it ended. The segments are
     joined into candidate clips (`src/audiolab/data/talk_clips.py`) until a full
     stop once the clip lasts 5 s, a pause longer than 1 s, or 20 s. A segment
     without words or without duration is left out and ends its clip; the words a
     segment repeats from the start of the one before are dropped; a clip whose
     text compresses better than 2.4 to 1 is a decoding loop and is dropped. Each
     clip is widened by up to 0.2 s on each side, never past half the gap to the
     next segment, and a clip outside 0.5–20 s is not labelled.
  2. **Qwen3-ASR 1.7B** (`stt-v-qwen3-asr-1.7B`, served by vLLM) hears each
     candidate clip alone. The clip is kept when Qwen's text and Whisper's differ
     by at most 15 % of the words in the scorer's normalized view
     (`MAX_DISAGREEMENT`; casing, punctuation and how a number is written do not
     count): one word in a clip of seven passes, two do not. The label is Qwen's
     text, cased and punctuated. Qwen hears only the clip, so the test checks the
     cut as well as the words: a clip whose edges miss a word is heard differently.
  3. **Parakeet TDT 0.6B v3** (`stt-c-parakeet-tdt-0.6B-v3`), the base model of
     the fine-tune, hears every kept clip, and the manifest records its word error
     rate against the label, normalized, as `base_wer`. It does not vote: the field
     lets a mix choose the clips the base model gets wrong.

  The server holds one model at a time, so the passes run over a batch of talks:
  every Whisper request, then every Qwen one, then every Parakeet one
  ([D-020](../owner/directives.md)). Each model is sent as many requests at once
  as its entry in `AUDIOLAB_CONCURRENCY` says, and one when it has none
  ([D-022](../owner/directives.md); the variable is described in
  [evaluation](evaluation.md#4-run-and-read-the-result)): several talks at once
  for Whisper, several clips of one talk for Qwen and Parakeet. The answers are
  read in the order a one-at-a-time pass would read them, so the clips kept do
  not depend on which answer arrives first. How many requests a model takes at
  once is a fact of the server that serves it: a server that batches requests,
  such as vLLM, may take ten
  (`AUDIOLAB_CONCURRENCY=stt-v-whisper-large-v3=10,stt-v-qwen3-asr-1.7B=10`),
  while one that answers one request after the other gains nothing from more.
  Before its first request, each pass waits for its model to answer a warmup, as
  an evaluation does ([evaluation](evaluation.md#4-run-and-read-the-result)): the
  server may have unloaded the model since its last pass, and loading it again
  can take minutes. A pass whose answers are all cached sends nothing, warmup
  included, so a resumed preparation makes the server load no model for nothing.
  A talk is no longer labelled once it holds its 200 clips or the cap is full.
  Every raw answer is cached per talk and per model under
  `$AUDIOLAB_CACHE/peertube-fr-train`, so an interrupted preparation resumes where
  it stopped.
- **Prepared** (2026-11-09): the first 65 h of the reading order (84 talks) were
  downloaded beforehand (`audiolab data fetch-audio peertube-fr-train`), and the
  labelling run read **86 talks, 66.9 h**, in one batch. Qwen labelled 17 862
  candidate clips and **5 862 of them (32.8 %) were rejected** because the two
  models disagree; on the way, 457 candidates fell outside 0.5–20 s, 102 segments
  had no duration and 865 no words, and 199 segments lost the words they repeated
  from the one before. The cap was full after 77 talks, so the last 9 were heard by
  Whisper and never labelled. The result is **12 000 clips (the cap), 31.8 h, 77
  groups**, median 8.4 s, all CC BY; 30 talks give their full 200 clips and the
  smallest gives 45. The prepared set is 3.4 GiB, and 7.3 GiB of WAV stays in the
  cache.
- **What the base model gets wrong**: Parakeet's text differs from the label on
  8 871 of the 12 000 clips (73.9 %); the mean `base_wer` is 18.6 %, and 26 % of
  the clips have a `base_wer` of 0. This measures the base model against model
  labels, not against the speech: a wrong label counts as a base-model error.
- **Known limits**: **the label error of the kept clips is not measured** — no
  human has checked a sample, so how many wrong labels the agreement test lets
  through is unknown. It lets through a word both models get wrong, and one word
  only one of them gets wrong in a long enough clip: Qwen writes "on utilise
  l'ADN" where the speaker says "FDN" (French Data Network, which Whisper and
  Parakeet both heard), and the clip is kept with Qwen's text. Technical terms and
  names are where Qwen errs, and they are the vocabulary this source is for. The
  labels also depend on the run: vLLM does not answer bit for bit the same when
  the requests sent beside one change (a comma, a capital letter), and the cache
  keeps the first answer, so a preparation from an empty cache can keep a slightly
  different set of clips and labels.

### `peertube-edu-fr-train` — the education ministry's French videos

- **Source**: the eleven PeerTube instances the French ministry of education runs on
  `apps.education.fr`, one per teaching area or school level, each with its own
  catalogue and its own REST API. `audiolab data list-talks peertube-edu-fr-train`
  asks every instance for its own videos in French under licence CC BY (PeerTube
  licence id 1) or CC0 (id 7) and writes the answer to
  [`peertube_edu_listing.json`](../../src/audiolab/data/peertube_edu_listing.json).
  The eleven instances are committed in `src/audiolab/data/peertube_edu.py`: they
  are the hosts the PeerTube instance index (`instances.joinpeertube.org`) lists
  under that domain (checked 2026-11-08); the ministry's own `tube-test` instance is
  left out. The listing of 2026-11-08 holds **2 595 videos, 1 481.2 h** — every
  instance is in it, the shortest video is 600 s and the median 1 192 s. The videos
  and the hours per instance, which the table at the end of this card holds, are
  lopsided: the largest instance (`tube-arts-lettres-sciences-humaines`) holds 649
  of the 2 595 videos, which is why a seeded selection spreads over the instance and
  not over the listing.
- **License and attribution**: per video, as each uploader declared it: PeerTube
  licence id 1 ("Attribution") is recorded as `CC-BY` and id 7 ("Public Domain
  Dedication") as `CC0-1.0` — PeerTube names a licence without a version, so the
  manifest records none. 2 433 videos are CC BY and 162 CC0. Every clip carries its
  own video's `source_url`, `source_title`, `source_author` (the PeerTube account)
  and `license`; the channel — a teacher's channel, a mission, a discipline — is the
  collection the video sits in.
- **Language**: French, from the video's own language tag, and the tag is required:
  an untagged video is not listed. These instances tag their videos (unlike the
  Debian one, whose card says how that is handled), and a video another instance
  tagged in another language is not this source's.
- **What the filter leaves out**: videos shorter than ten minutes, videos the
  instance does not host itself, videos under another licence, channels whose
  name holds a word of music, song or the stage (`NON_SPEECH_WORDS` in
  `src/audiolab/data/peertube_edu.py`), and videos whose title names a performance
  (`PERFORMANCE_WORDS`: a concert, a choir, a show, a musical). Those words are
  matched whole, so `Éducation musicale et chant choral` and `Ville en choeur 2026`
  are left out where `Main chantallayoussaint channel` is not. The title words are
  the fewer of the two because a lesson is often about music or the stage (`Le
  professeur de musique`, `Interview radio théâtre`), and a teacher's own channel
  holds the lesson and the end-of-year concert alike. What escapes both lists
  yields few clips, the two labelling models not agreeing on music. The
  classes, the staff meetings and the teachers explaining a lesson are what the
  source keeps — the ministry's videos are lessons and webinars, not conference
  talks only.
- **Split and order**: `source_split` is `listing`; the listing is written in uuid
  order and read in a seeded shuffle (`SEED`) spread over the instance, so the first
  talks of a preparation cover all eleven catalogues instead of the biggest one.
  Audio comes from the smallest file the instance serves for a video, converted to
  16 kHz mono WAV; the video is deleted once its sound is out.
- **Why it is here**: it is the French arm that is neither a conference nor a
  published corpus — teachers and the ministry speaking French, on every school
  subject, in a register no evaluation set of this repository has. It is the French
  counterpart of the conference sources with a wider vocabulary: one speaker per
  video, on arithmetic as well as on physics.
- **Group**: the recording (`group_kind: recording`, the video uuid). A video
  usually holds one speaker; the listing names none, and a video that records a
  class holds the pupils' voices as well.
- **Labels**: the three passes of the [card of
  `peertube-fr-train`](#peertube-fr-train--capitole-du-libre-talks-in-french),
  with its models and thresholds, cached under
  `$AUDIOLAB_CACHE/peertube-edu-fr-train`.
- **Prepared** (2026-11-09): the first 65 h of the reading order (101 talks) were
  downloaded beforehand (`audiolab data fetch-audio peertube-edu-fr-train`), and the
  labelling run read **119 talks, 74.9 h**, in two batches: 86 talks (57.3 h), then
  33 (17.7 h), sized on the yield of the first (`batch_size`,
  `src/audiolab/data/talks.py`); the 18 talks past the first 101 were downloaded as
  Whisper went. Qwen labelled 16 768 candidate clips and **4 768 of them (28.4 %)
  were rejected** because the two models disagree; on the way, 601 candidates fell outside 0.5–20 s and 21 were
  dropped as a decoding loop, 277 segments had no duration and 2 590 no words, and
  1 029 segments lost the words they repeated from the one before. The cap was full
  after 114 talks, so the last 5 were heard by Whisper and never labelled; one
  labelled talk, a travel vlog on which Whisper heard mostly music, kept none of its
  6 candidates. The result is **12 000 clips (the cap), 31.2 h, 113 groups**, median
  8.1 s; 11 774 clips (109 talks) are CC BY and 226 (4 talks) CC0. Each of the
  eleven instances gives 9 to 11 talks; 25 talks give their full 200 clips and the
  smallest gives 3. The prepared set is 3.4 GiB, and the 119 talks read leave
  8.0 GiB of WAV in the cache.
- **No overlap with the evaluation sets** (decision 0002): the prepared set
  predates every evaluation set made from this listing, but one of its talks —
  73 clips — is among the human-captioned talks of
  [`peertube-edu-captions-fr`](#peertube-edu-captions-fr--the-education-ministrys-french-talks-with-human-captions),
  and that set's preparation leaves the talk out. Every preparation made after
  an evaluation set exists skips the set's talks (`evaluation_datasets` in
  `talk_listing.py`), checked in `tests/test_data_peertube_edu_captions.py`.
- **What the base model gets wrong**: Parakeet's text differs from the label on
  7 316 of the 12 000 clips (61.0 %); the mean `base_wer` is 12.7 %, and 39 % of
  the clips have a `base_wer` of 0. This measures the base model against model
  labels, not against the speech: a wrong label counts as a base-model error.
- **Known limits**: **the label error of the kept clips is not measured** — no
  human has checked a sample, so how many wrong labels the agreement test lets
  through is unknown. What the test lets through, and why a preparation from an
  empty cache can keep a slightly different set of clips and labels, is in the
  [card of `peertube-fr-train`](#peertube-fr-train--capitole-du-libre-talks-in-french).

Per instance, as the listing of 2026-11-08 holds them:

| instance | videos | hours |
| --- | --- | --- |
| `tube-action-educative` | 236 | 163.4 |
| `tube-arts-lettres-sciences-humaines` | 649 | 423.8 |
| `tube-cycle-2` | 96 | 42.8 |
| `tube-cycle-3` | 113 | 53.5 |
| `tube-education-physique-et-sportive` | 28 | 17.4 |
| `tube-enseignement-professionnel` | 84 | 41.4 |
| `tube-institutionnel` | 120 | 82.9 |
| `tube-langues-vivantes` | 29 | 18.5 |
| `tube-maternelle` | 85 | 56.7 |
| `tube-numerique-educatif` | 518 | 313.4 |
| `tube-sciences-technologies` | 637 | 267.5 |

### `peertube-tech-fr-train` — the tech talks of the two French PeerTube listings

- **Source**: the union of the pinned listings of
  [`peertube-fr-train`](#peertube-fr-train--capitole-du-libre-talks-in-french)
  (519 talks) and [`peertube-edu-fr-train`](#peertube-edu-fr-train--the-education-ministrys-french-videos)
  (2 595 videos), kept when the talk's title holds a word of the technology the
  model trains on — the list of words is `TECH_TITLE` in
  [`src/audiolab/data/peertube_tech.py`](../../src/audiolab/data/peertube_tech.py).
  `audiolab data list-talks peertube-tech-fr-train` computes the union from the
  two parents' pinned listings — no network — and writes it to
  [`peertube_tech_listing.json`](../../src/audiolab/data/peertube_tech_listing.json).
  The listing of 2026-11-13 holds **421 talks, 289.5 h**: 240 of the Capitole's
  talks and 181 of the ministry's videos, no video in both listings. The shortest
  is 525 s and the median 2 179 s.
- **License and attribution**: CC-BY or CC0-1.0 per video, as each parent
  declares it (406 CC-BY, 15 CC0-1.0; the Capitole is all CC-BY). It is the
  parents' audio under the parents' licences — no acquisition beyond the two
  sources of [sources.md](sources.md#french-variety), rows 3 bis and 22. Every
  clip carries its video's `source_url`, `source_title`, `source_author` and
  `license`, as in the parents' cards.
- **Why it is here**: scale-up of the French technical vocabulary the screening
  measured on the two parents (2026-11-10, plan 03 of the training-data
  campaign): the talks whose titles already say they are about what a developer
  dictates — programming, networks, data, artificial intelligence — in the
  source the base model's target use is built for
  ([D-009](../owner/directives.md)). The title words are the source's own
  definition; the listing pins what they kept.
- **Split and order**: `source_split` is `listing`; the listing is written with
  the Capitole's talks first and the ministry's after, each in its parent's
  order, and read in a seeded shuffle (`SEED`, `src/audiolab/data/talks.py`)
  spread over the instance (`spread_by` = the talk's instance): the two parents
  are very uneven, and a plain shuffle could spend a selection on one of them.
  The audio and the three labelling passes are the parents', as described in the
  [card of `peertube-fr-train`](#peertube-fr-train--capitole-du-libre-talks-in-french);
  the answers are cached per talk and per model under
  `$AUDIOLAB_CACHE/peertube-tech-fr-train`, a directory of its own, since the
  clips it keeps are recorded under its own name.
- **Group**: the recording (`group_kind: recording`, the video uuid), at most
  200 clips per talk.
- **Overlap with the parents**: the talks are the parents' talks, so a
  preparation of this source and a preparation of a parent can keep the same
  audio. That is allowed: the guard against a training mix double-counting
  audio is the `sha256` check of `src/audiolab/train/data.py`, which refuses a
  mix whose two datasets share one.
- **Evaluation exclusion**: the talks of
  [`peertube-edu-captions-fr`](#peertube-edu-captions-fr--the-education-ministrys-french-talks-with-human-captions)
  are left out of every preparation made after that set exists
  (`evaluation_datasets`, [decision 0002](../decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md));
  the listing of 2026-11-13 holds one such talk.
- **Prepared**: pending, the first preparation of the scale-up campaign; the
  numbers go here when it is written.

### `debconf-en-train` — DebConf talks on the Debian instance

- **Source**: [`peertube.debian.social`](https://peertube.debian.social), the
  Debian project's own PeerTube instance. `audiolab data list-talks
  debconf-en-train` asks its REST API for every local video under licence CC BY
  (PeerTube licence id 1) or CC0 (id 7) whose channel or account name holds
  `debconf`, and writes the answer to
  [`debconf_listing.json`](../../src/audiolab/data/debconf_listing.json). The
  listing of 2026-11-08 holds **1 919 videos, 1 244.4 h**, in 30 channels: the
  yearly conference from DebConf 4 to DebConf 26, six mini-DebConfs, and the `QA
  Meetings` channel (12 videos) that a DebConf account publishes talks under;
  616 s is the shortest and 2 467 s the median video. A preparation reads that
  file and not the live instance, so a talk published or removed later changes
  nothing.
- **License and attribution**: CC BY, as each uploader declared it; PeerTube names
  the licence ("Attribution") without a Creative Commons version, so the manifest
  says `CC-BY` and no version, and no listed video is CC0. CC BY asks for
  attribution, so every clip carries its video's `source_url`, `source_title` and
  `source_author` (the PeerTube account), and its `license`. Attribution also
  rests on the instance and the conference the video belongs to, both of which the
  card and the channel name record.
- **Language, and why the filter is not about the language tag**: the instance
  leaves the language of most of its videos unset — the DebConf uploads were made
  in bulk — so requiring `en` would keep 121 of the 2 779 local videos the
  instance hosts under those two licences (4 %), where 2 343 of them carry no
  language tag at all. That is also why this listing is a PeerTube-specific
  filter and not a language one: the tag would answer a question the corpus never
  asked. The filter
  therefore accepts a video tagged English **or untagged**, and takes the
  conference's language as its own: DebConf is held in English, and the editions
  that are not held in English are left out by name — `argentina`, `brazil`,
  `brasil`, `es`, `france`, `gatherings`, `india`, `portugal` in a channel or
  account name (`DebConf ES`, MiniDebConf Brasil and Portugal, the Debian Brasil
  channel, the mini-DebConf France, the German Debian-edu gatherings and the
  miniDebConf India 2021, held in Hindi, Malayalam, Marathi and Telugu). Those
  words are matched as whole words, so an account called `Andreas @ DebConf` is
  not dropped by the `es` of its name.
- **Talks in another language at an English DebConf**: a DebConf has talks in
  the language of its host country too, and the instance tags them no more than
  the others — Spanish at DebConf 9, 12, 20 and 26, Portuguese at DebConf 19,
  Chinese at DebConf 18, Malayalam and Telugu at DebConf 20 and 21. Labelled as
  English, such a talk is translated by Whisper, and the two models can agree on
  the translation. So a talk is left out when its title is not written in English
  (`english_title` in `src/audiolab/data/debconf.py`): a title in another script,
  or holding a Spanish, Portuguese, French or German word that English titles do
  not use (`en`, `para`, `avec`, `mit`, …) or a letter English does not write
  (`ñ`, `ç`, `ó`, …). In the catalogue read on 2026-11-08, the miniDebConf India,
  the mini-DebConf France and the Debian-edu gatherings hold 91 videos of at least
  ten minutes (55.4 h), and the title rule leaves out 75 more (51.8 h). A talk
  given in another language under an English title still escapes the filter, and
  an English title that names a place such as `Brasília` is lost to it. Videos
  shorter than ten minutes, videos under another licence and videos the instance
  does not host itself are left out as in the other talk sources.
- **Labels are English**: this is the first source of the repository labelled in
  another language than French, so the language is not a constant any more: the
  three models are asked to transcribe in English (`send_language`), the clips are
  cut and normalized as English (`candidates`, `doubt`, `disagreement`), and each
  manifest line says `language: en`. A French speaker reading English words
  therefore yields an English label, and the source is the English arm of the
  screening that is not read speech.
- **Split and order**: `source_split` is `listing`; the listing is written in uuid
  order, which only keeps the file stable, and the talks are read in a seeded
  shuffle of it (`SEED`) spread over the **conference** — the 30 channels are
  covered one after the other, so the talks of a first cap come from every edition
  instead of the one that happens to sort first. Audio comes from the smallest
  file the instance serves for a video, converted to 16 kHz mono WAV; the video is
  deleted once its sound is out, and a talk already in the cache is not fetched
  again.
- **Group**: the recording (`group_kind: recording`, the video uuid). A talk
  usually has one speaker, the listing names none, and the questions from the room
  add others.
- **Labels**: the three passes of the [card of
  `peertube-fr-train`](#peertube-fr-train--capitole-du-libre-talks-in-french),
  with its models and thresholds, asked in English (above), cached under
  `$AUDIOLAB_CACHE/debconf-en-train`.
- **Prepared** (2026-11-09): the first 65 h of the reading order (106 talks) were
  downloaded beforehand (`audiolab data fetch-audio debconf-en-train`), and the
  labelling run read **106 talks, 65.7 h**, in two batches: 86 talks (53.6 h), then
  20 (12.1 h), sized on the yield of the first (`batch_size`,
  `src/audiolab/data/talks.py`). Qwen labelled 17 680 candidate clips and **5 680
  of them (32.1 %) were rejected** because the two models disagree; on the way,
  1 273 candidates fell outside 0.5–20 s, 15 segments had no duration and 396 no
  words, and 290 segments lost the words they repeated from the one before. The
  cap was full after 101 talks, so the last 5 were heard by Whisper and never
  labelled; one labelled talk, a DebConf 6 venue proposal for Sarajevo spoken in
  Bosnian, which Whisper translated into English, kept none of its 27 candidates.
  The result is **12 000 clips (the cap), 34.2 h, 100 groups**, median 9.3 s, all
  CC BY. Each of the 30 channels gives 1 to 4 talks (`DebConf 4` holds one video);
  10 talks give their full 200 clips and the smallest gives 1. The prepared set is
  3.7 GiB, and the 106 talks read leave 7.1 GiB of WAV in the cache.
- **What the base model gets wrong**: Parakeet's text differs from the label on
  7 978 of the 12 000 clips (66.5 %); the mean `base_wer` is 8.3 %, and 34 % of
  the clips have a `base_wer` of 0. This measures the base model against model
  labels, not against the speech: a wrong label counts as a base-model error.
- **Known limits**: **the label error of the kept clips is not measured** — no
  human has checked a sample, so how many wrong labels the agreement test lets
  through is unknown. What the test lets through, and why a preparation from an
  empty cache can keep a slightly different set of clips and labels, is in the
  [card of `peertube-fr-train`](#peertube-fr-train--capitole-du-libre-talks-in-french).

### `fosdem-en-train` — FOSDEM talks, seven years of the conference

- **Source**: FOSDEM publishes a talk twice, and neither side is enough on its
  own. The schedule (`https://fosdem.org/<year>/schedule/xml`, which redirects to
  FOSDEM's archive) holds one `<event>` per talk with its title, speakers, room,
  language and slot, and lists talks whose recording was never published; the
  video tree (`https://video.fosdem.org/<year>/<room>/`) is a plain HTTP directory
  of `.mp4` and `.webm` files, and a file name says nothing about who spoke. A
  talk is therefore listed when the schedule points at a file the tree **actually
  serves** — the join is a match on the URL, decoded on both sides, because the
  tree percent-encodes a room whose name holds a space (`H.1301%20%28Cornil%29/`)
  and the schedule does not — and the schedule is where the metadata comes from.
  The listing of 2026-11-08 holds **5 477 talks, 2 574.3 h**, in 108 rooms over
  the seven years 2020 to 2026 ([`fosdem_listing.json`](
  ../../src/audiolab/data/fosdem_listing.json)). Those hours are the **lengths
  of the schedule's slots**, not of the recordings, which the listing never
  reads: a talk can end before its slot does, and a recording can be cut short.
  The tree goes back to 2003, which this source does not read: seven years
  already cover every devroom and topic the conference has.
- **The join, year by year**: the schedule of 2020 lists 872 events, 818 of them
  with a link into the tree; 811 of those last ten minutes and are in English, and
  811 are listed, the one difference being an event whose recording the tree does
  not have. The same holds for 2021 (773 events, 644 listed), 2022 (730, 644),
  2023 (780, 713), 2024 (874, 799), 2025 (1 105, 923) and 2026 (1 079, 943). The
  listing holds what a viewer of the conference could watch, not what it
  announced.
- **License and attribution**: CC BY 2.0 BE, stated on the tree's own pages, whose
  footer links to <https://creativecommons.org/licenses/by/2.0/be/deed.en>
  (checked 2026-11-08, on the pages of several years). That is the one source here
  whose licence is a version of CC BY with a jurisdiction, so the manifest records
  `CC-BY-2.0-BE` rather than `CC-BY`. Every clip carries the talk's `source_url`
  (the event's page in the schedule, which names the talk, its speakers and the
  recording: the `<url>` the schedule gives, or `/<year>/schedule/event/<slug>/`
  for the years whose schedule gives none), `source_title` and `source_author`
  (the speakers the schedule names, joined with commas); 16 of the 5 477 talks
  name no speaker at all — a lightning-talk slot, a stand — and their lines carry
  no author.
- **Language**: the schedules of 2020 to 2023 tag no language at all — every
  event of those years has an empty `<language>` — and those of 2024 to 2026 tag
  every event `en`. So the language rule drops nothing over the years listed: a
  talk is taken as English when its year says so or says nothing, FOSDEM being
  held in English, and it would leave out an event a year tags in another
  language. A talk that was not given in English in an untagged year is therefore
  labelled as English like the others, and its clips go through the same
  agreement test; the programme is in English, so this is a defect of the source
  and not a rule of the filter. Talks shorter than ten minutes are left out as in
  the other sources.
- **Split and order**: `source_split` is `listing`; talks are read in a seeded
  shuffle (`SEED`) spread over **year and room**, so a first cap covers every year
  and every devroom instead of the biggest one. The audio is the MP4 the listing
  pinned — preferred over the WebM because it plays everywhere, and over the AV1
  WebM, which a decoder may refuse — converted to 16 kHz mono WAV.
- **Group**: the recording (`group_kind: recording`), the talk. Its id is the
  year and the schedule's event id (`2025-6712`): an event id is unique within one
  year only, and a slug — over 200 characters for some talks — would make cache
  file names close to the length a file system allows.
- **Labels**: the three passes of the [card of
  `peertube-fr-train`](#peertube-fr-train--capitole-du-libre-talks-in-french),
  with its models and thresholds, asked in English as for
  [`debconf-en-train`](#debconf-en-train--debconf-talks-on-the-debian-instance),
  cached under `$AUDIOLAB_CACHE/fosdem-en-train`.
- **Prepared** (2026-11-09): the first 65 h of the reading order (135 talks) were
  downloaded beforehand (`audiolab data fetch-audio fosdem-en-train`); 134 talks
  (64.7 h) arrived, and the 135th, `2023-14136` ("What Does Rugby Have To Do With
  Sigstore?"), could not be read: ffmpeg rejects the MP4 the tree serves for it
  (`moov atom not found`), and rejected it again in the labelling run. That run
  read **140 talks, 67.0 h**, in two batches: 86 talks (39.8 h), then 54 (27.2 h),
  sized on the yield of the first (`batch_size`, `src/audiolab/data/talks.py`);
  it downloaded the last 5 itself. Those hours are the slots' (above): the WAV of
  the 139 talks heard lasts 59.9 h. Qwen labelled 15 490 candidate clips and
  **3 490 of them (22.5 %) were rejected** because the two models disagree; on
  the way, 867 candidates fell outside 0.5–20 s, 2 segments had no duration and
  132 no words, and 104 segments lost the words they repeated from the one
  before. The cap was full after 124 talks, so the last 15 were heard by Whisper
  and never labelled; every labelled talk kept clips. The result is **12 000
  clips (the cap), 35.6 h, 124 groups**, median 9.8 s, all CC-BY-2.0-BE. The 124
  talks cover the seven years, 13 to 26 a year, and no two come from the same
  room in the same year; 3 talks give their full 200 clips and the smallest
  gives 12. The prepared set is 3.8 GiB, and the 139 talks heard leave 6.4 GiB of
  WAV in the cache.
- **What the base model gets wrong**: Parakeet's text differs from the label on
  7 768 of the 12 000 clips (64.7 %); the mean `base_wer` is 7.0 %, and 35 % of
  the clips have a `base_wer` of 0. This measures the base model against model
  labels, not against the speech: a wrong label counts as a base-model error.
- **Known limits**: **the label error of the kept clips is not measured** — no
  human has checked a sample, so how many wrong labels the agreement test lets
  through is unknown. What the test lets through, and why a preparation from an
  empty cache can keep a slightly different set of clips and labels, is in the
  [card of `peertube-fr-train`](#peertube-fr-train--capitole-du-libre-talks-in-french).

### `nasa-en-train` — NASA's podcasts, labelled from NASA's own transcripts

This source is listed and downloaded like the talks above, but its labels are not
model output: NASA publishes a transcript with each episode, and that human text,
cased and punctuated, is the label. A model only says *when* each word was said.
Its 60 reserved episodes are the evaluation set
[`nasa-en-test`](#nasa-en-test--nasas-podcasts-the-human-transcribed-evaluation-set);
this preparation leaves them out (decision 0002).

- **Source**: four NASA shows, read from their RSS feeds
  (`https://www.nasa.gov/feeds/podcasts/<show>`): each item gives the episode's
  MP3 and the page on `www.nasa.gov/podcasts/` whose body holds the transcript.
  NASA's Curious Universe is not read: the music beds under the speech and the
  archival recordings the rules below leave out concentrate in that show, so
  the source excludes it entirely rather than episode by episode
  (`EXCLUDED_SHOWS` in `src/audiolab/data/nasa.py`).
  `audiolab data list-talks nasa-en-train` reads the feeds, fetches every episode
  page once (kept raw in `$AUDIOLAB_CACHE/nasa-en-train/pages`, never in the
  repository) and pins the episodes in
  [`nasa_listing.json`](../../src/audiolab/data/nasa_listing.json): show, title,
  page, MP3 and duration, no text. The listing of 2026-11-13 holds **768
  episodes, 521.8 h**, the median episode 39 minutes long:

  | show | episodes | hours |
  | --- | --- | --- |
  | Houston We Have a Podcast | 419 | 373.5 |
  | Small Steps, Giant Leaps | 151 | 65.6 |
  | NASA in Silicon Valley | 102 | 46.9 |
  | Gravity Assist | 96 | 35.9 |

- **What the listing leaves out**, from the 884 items of the four feeds on
  2026-11-13, each rule in `src/audiolab/data/nasa.py`:
  - 39 items the feed reader drops before the listing's checks, because they
    link no usable page under `www.nasa.gov/podcasts/`: 33 carry an empty
    `<link>` (6 in Houston We Have a Podcast, 27 in Small Steps, Giant Leaps),
    5 in Small Steps, Giant Leaps link the show's section page (2) or a page
    on `appel.nasa.gov` (3), and 1 in Houston We Have a Podcast ends its page
    URL with a space and a U+200E that `episode_slug` refuses
    (`call-sign-white-flight`). All 39 carry an MP3, so it is the page check
    that drops them; the reader keeps its page and MP3 guards at every
    listing. The 845 they leave are what the rules below work on;
  - 17 repeats, an episode aired again being listed once, under its first
    airing. Two episodes are the same when they share a page, an MP3 file or a
    title once an "Encore" or a series' "Ep. N:" prefix and punctuation are set
    aside: the 11 episodes of Houston We Have a Podcast's "Mars Ep. N" series,
    which air earlier episodes again, and 6 episodes posted twice (3 in Houston
    We Have a Podcast, 3 in NASA in Silicon Valley). The feeds are read oldest
    first, so the copy kept is the first aired. 3 more pages say their episode
    aired before ("This episode originally aired as episode 355"
    (`callsign-white-flight`), "Originally aired on February 2…"
    (`podcast-archive-2009-nasas-kepler-mission-to-look-for-earth-like-planets`),
    "the rebroadcast of a conversation"
    (`throwback-thursday-featuring-veteran-astronaut-steve-smith`));
  - 48 pages without a transcript, all of NASA in Silicon Valley (its "Live"
    video streams and its short news pieces, whose page holds a caption or a
    link), a page counting as a transcript when it holds 60 words per minute of
    audio (people say about 150); and 2 pages that hold more than 250 words per
    minute, each with its transcript written twice (the median page holds 169,
    the fastest of the others 224);
  - 5 episodes that play recordings: a transcript note or a speaker label names
    archival audio, a broadcast or a clip, an archive, an excerpt or an
    announcer (`[Apollo 8 launch clip]`, `[NASA TV Report]`, `Announcer:`,
    `[Sound clip.]`). The episode is left out whole: some of those recordings
    are another broadcaster's, the page does not say whose each one is, and a
    transcript sometimes writes their words as lines of the show, which a clip
    would then carry;
  - 1 transcript that writes its notes as plain words (below, under *The
    transcript*);
  - 1 page with a copyright notice (below). No page of the current feeds
    credits music of a third party; both rules are still applied at every
    listing.
- **License and attribution**: public domain. A work of the US government is not
  protected by copyright in the United States (17 U.S.C. § 105), and NASA's media
  guidelines (<https://www.nasa.gov/nasa-brand-center/images-and-media/>, checked
  2026-11-08) say that NASA content, audio included, "generally [is] not subject
  to copyright in the United States", and that "NASA occasionally uses
  copyright-protected material of third parties", which is then "identified as
  copyright protected with the name of the copyright holder".
  - What § 105 covers: a "work of the United States Government" is one prepared
    by an officer or employee of the government as part of their official duties
    (17 U.S.C. § 101). Work a contractor makes for NASA is not one: it can be
    under copyright, the contractor's or assigned to the government, which
    § 105 lets the government hold. The pages do not say which minutes, if any,
    a contractor produced. For those, the source relies on NASA's statement and
    its rule of naming a holder, through the two rules below; that is NASA's
    word, not a proof for each minute.
  - A show is read only when its feed names NASA, or NASA's knowledge programme
    ("NASA/APPEL Knowledge Services"), as the copyright holder; a feed naming
    anyone else is refused at the next listing. That leaves out the sixth NASA
    show, **On a Mission**, whose feed names "NASA/JPL/Caltech": JPL is run for
    NASA by Caltech, so its work is not a US government work.
  - An episode whose page carries a notice — a ©, "all rights reserved",
    "copyright" followed by a holder or a year, material "used courtesy of"
    or "used with permission", or audio, a recording, a clip or footage
    "provided by" or "courtesy of" someone — is left out whole, since a notice
    does not say which minutes it covers. A notice and not a word: guests say
    "copyright" in passing. The one left out plays audio of a conversation
    "provided by Explore Mars and the Planetary Society"
    (`gravity-assist-what-if-we-found-life-on-mars`); the other page of the
    listing of 2026-11-09 (`sun-series-soaring-toward-the-sun`) is no longer in
    the feeds.
  - Each preparation fetches every page it reads again and applies the
    listing's rules to it, so an episode whose page took a notice, or lost its
    transcript, since the listing is skipped and counted.
  - An episode whose page credits music of a third party is left out whole:
    the series' library ("Music for the series comes from Universal Production
    Music"), its theme ("composed by … of SYSTEM Sounds") or a note naming a
    piece and its composer (`[Song: Ants Instrumental by Taylor Welsh]`). Such
    music plays under the speech where the transcript does not note it, and its
    licence is unknown, which D-019 does not train on. The listing of
    2026-11-13 leaves none: all 45 of the listing of 2026-11-09 (17.2 h) were
    of NASA's Curious Universe, which the source no longer reads. Elsewhere,
    clips are kept away
    from music where the transcript notes it: a line with a note of a sound
    (`[ Music ]`, `[Song: …]`, `<Intro Music>`) and the line of text on each side
    of it are not labelled. Music the transcript does not note is not detected.
  - Every clip carries `license: public domain (US government work)`,
    `source_author` "NASA, <show>", `source_title` (the episode) and `source_url`
    (the episode's page), which is the attribution NASA asks for ("NASA should be
    acknowledged as the source of the material").
- **The transcript**: the episode's content block, from a "Transcript" heading
  when the page has one (the description above it is text nobody says), to the
  site's footer. A speaker label is taken off its line — a bold name alone, a
  capitalised name before the line's first colon, bold or not (`Host:`, `Andres
  Almeida (Host):`, `NICKY FOX:`, `Willie Lyles:`), a name alone on its line
  with a time (`Steve Altemus 11:32`), or a short plain name alone on its line
  that recurs — and remembered: the label changes where the speaker does. A
  phrase of speech before a colon (`Today: the Moon`) is read as a label too:
  that only costs the clip across it, while a label read as speech would put a
  name, and two voices, in a clip. Notes in brackets, parentheses, braces or
  asterisks (`[Music]`, `(laughs)`, `{Laughter}`, `*cough*`) are not words
  anybody says. The one transcript that writes its notes as plain words
  (`a-seismic-journey`: "those phone calls laughter") is left out, by name.
- **Labels, by alignment** (`src/audiolab/data/aligned_clips.py`): Whisper
  large-v3 hears each episode as the talks are heard (five-minute chunks, raw
  answers cached, segments joined into candidate clips of 20 s at most). Its
  words and the transcript's are put in the normalized view of the scorer —
  `Crew-4` and `crew four` are one word stream, `2,000` and `two thousand` too —
  and aligned with `difflib.SequenceMatcher`. A candidate clip is labelled when
  it lasts 1 s to 20 s; its first and last words are matched; no transcript
  word is left unmatched just outside it (it would be audio the label misses);
  its edges fall on whole written words; it holds no speaker change and no note,
  and is not on or next to a line that notes a sound; at least 90 % of its words
  are matched, on both sides; and no run of more than two words is unmatched on
  either side. The label is then the stretch of the
  transcript between the first and the last matched word, as NASA wrote it:
  cased, punctuated, digits kept, its typography in the characters the base model
  writes (`I don’t know` is labelled `I don't know`, `“ ”` are dropped, `—` is a
  break; the rule is under [Training sources](#training-sources)). A transcript is near-verbatim, not verbatim —
  "um" and false starts are often edited out — and the two thresholds bound how
  far a label may stray from the audio. No second model labels the clips, so
  there is no Qwen pass, and Parakeet TDT 0.6B v3 is the one model that hears a
  clip alone: Whisper heard its words in the whole episode, so its timestamps
  can cut the first or last word, or let in one of the next sentence. A clip is
  kept only when the first and last words Parakeet hears in it, normalized, are
  the label's; the same answer records its `base_wer`. That also drops a clip
  whose edge word Parakeet gets wrong for another reason; the report counts
  them. The room left under the cap is shared over the episodes of a batch one
  clip at a time, and every Whisper request of a batch is sent before the first
  Parakeet one.
- **Split and order**: `source_split` is `listing`; episodes are read in a seeded
  shuffle (`SEED`) spread over the show, so a first cap covers the four shows
  rather than Houston We Have a Podcast, which holds 71 % of the hours. The
  audio is the MP3 the listing pinned, converted to 16 kHz mono WAV.
- **Prepared audio**: before the labelling run, the first 65 h of the reading
  order of the listing of 2026-11-09 — 128 episodes, 66.0 h: 30 or 31 per
  show, and the 7 left of NASA's Curious Universe — were
  downloaded (`audiolab data fetch-audio nasa-en-train --hours 65`) and kept as
  16 kHz mono WAV, 7.0 GiB in the pipeline's cache (`$AUDIOLAB_CACHE`,
  `.cache/audiolab` by default, git-ignored; the MP3 is deleted once its sound is
  out). Every episode of that selection arrived.
- **Prepared** (2026-11-09): the labelling run read **265 episodes, 140.3 h**, in
  seven batches sized on the yield (`batch_size`, `src/audiolab/data/talks.py`):
  86 episodes (43.0 h), 147 (79.9 h), 25 (13.8 h), then 4, 1, 1 and 1 as the cap
  neared. Of the 137 episodes past the selection above, it downloaded 91; the
  cache already held the other 46. Every page read again passed the listing's
  rules, and every episode was downloaded and heard. Whisper's segments, of which
  479 lost the words they repeated from the one before and 6 without duration and
  757 without words were left out, gave 46 044 candidate clips, and the
  transcripts labelled **19 019 of them (41.3 %)**. The 27 025 others are counted
  under the first test of the alignment (above) they fail, in the order the tests
  run:

  | left out because | clips |
  | --- | --- |
  | shorter than 1 s or longer than 20 s | 769 |
  | the first or last word Whisper heard has no pair in the transcript | 7 417 |
  | the two texts do not meet at an edge: the words just outside the clip are not each other's pair, so a transcript word there may have been said inside it | 9 277 |
  | a word the scorer drops ("Um,") is just outside the clip | 299 |
  | an edge falls inside a written word (`1,000`) | 74 |
  | a change of speaker or a note inside the clip | 5 247 |
  | on or next to a line that notes a sound | 925 |
  | fewer than 90 % of the words paired on one side | 2 620 |
  | more than two unpaired words in a row | 397 |

  Parakeet heard 13 834 of the labelled clips: in 1 834 of them (13.3 %) the
  first or last word it hears is not the label's, and the 12 000 others were
  kept. The other 5 185 labelled clips were never sent to it, being past their
  episode's share of 200 or past the room left under the cap. That first set
  had 12 000 clips (the cap), 34.8 h, and wrote NASA's typography (Known limits).
  It was prepared again the same day, from the cache only, with the labels in
  the characters the base model can write ([Training sources](#training-sources)):
  6 916 labels rewritten, 38 clips left out for a character still outside the
  vocabulary (`+ & # @ ° ō ʻ`), and 6 that would have needed a new request. The
  result is **11 956 clips, 34.6 h, 265 groups**, a subset of the first set
  (same audio, same `base_wer`), median 9.3 s, all public domain
  (US government work). Every episode read keeps clips, 1 to 176 (median 44), so
  none gives its full 200; six of the eight that give fewer than 10 are trailers
  and season introductions of under 3 minutes:

  | show | episodes | clips | hours |
  | --- | --- | --- | --- |
  | Houston We Have a Podcast | 65 | 3 990 | 11.5 |
  | Gravity Assist | 65 | 2 918 | 8.6 |
  | Small Steps, Giant Leaps | 64 | 2 897 | 8.8 |
  | NASA in Silicon Valley | 64 | 1 903 | 5.1 |
  | NASA's Curious Universe | 7 | 248 | 0.6 |

  The prepared set is 3.7 GiB, and the 265 episodes read leave 15.0 GiB of WAV
  in the cache.
- **What the base model gets wrong**: Parakeet's text differs from the label on
  7 229 of the 11 956 clips (60.5 %); the mean `base_wer` is 5.3 %, and 39 % of
  the clips have a `base_wer` of 0. By show, it differs on 29 % (NASA's Curious
  Universe) to 73 % (NASA in Silicon Valley) of the clips. The labels are human
  text, so this is closer to the base model's own error than on the talk
  sources, but not equal to it: a false start or a repeat the transcript edited
  out counts as a base-model error, and the edge check keeps only clips whose
  first and last words Parakeet already hears.
- **Known limits**: **the label error of the kept clips is not measured** — the
  thresholds bound how far a label may stray from Whisper's words, and nobody
  has checked a sample against the audio. The first set kept NASA's typography:
  `’` for every apostrophe (6 452 clips held one, none a straight `'`), and
  `“ ”`, `—` and `–`. The tokenizer of Parakeet TDT 0.6B v3 has no piece for
  them, so a model trained on those labels learned to write its unknown token,
  which is most of this source's harm on English in the
  [screening](screening.md#what-each-source-does-alone). The set prepared again
  writes none of them: its re-screen (2026-11-13, `screen-nasa-en-plain`)
  holds the unknown token in 10 of 7 712 answers, none of them in a purely
  English panel (against 592 of 4 116 in the first screen), and the first
  screen's English harm is gone.
- **Group**: the recording (`group_kind: recording`), the episode, by the slug of
  its page; at most 200 clips from one episode, as for every source.
- **Why it is here**: conversational technical English — interviews of
  engineers, scientists and astronauts, several voices per episode — with labels
  a human wrote. It is the one English source of that register whose labels are
  not a model's, so its arm says what the talks' model labels cost.

### No overlap with the evaluation sets

[Decision 0002](../decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)
freezes the evaluation sets of a campaign: a clip that was trained on can no
longer measure a model. A training source therefore leaves out every group of an
evaluation dataset that comes from the same corpus — the whole group, not the clip
that happens to repeat, because the clips of one speaker or one sentence are not
independent.

`evaluation_groups()` (`src/audiolab/data/train.py`) reads the prepared manifests
of the evaluation datasets and returns their groups; a fetcher whose corpus also
feeds an evaluation set passes them to its `Take`, which counts what it left out
in the fetch report. The evaluation dataset must be prepared (`audiolab data
fetch`) before the training source is, because the check reads its manifest.

| training source | evaluation set checked | left out |
| --- | --- | --- |
| `voxpopuli-fr-train` | `voxpopuli-en-nonnative` (accent `en_fr`) | no shared speaker: the 896 clips of the set's 27 speakers are left out and none of them is in the prepared source; **206 of the set's 307 plenary sessions are in the source too** — 1 956 train clips (4.67 h) share the rooms and microphones of 480 evaluation clips (1.6 h); session = clip id's prefix before the language tag, population = the prepared manifests ([audit](../../experiments/leakage-audit/README.md)), [proposed decision 0014](../decisions/proposed/0014-voxpopuli-shared-sessions.md) |
| `mls-fr-train` | `mls-fr` (test split of the same corpus) | 0 clips: MLS keeps the speakers of its splits apart, and the 18 test speakers are absent from the 138 prepared readers |
| `librispeech-en-train` | `librispeech-en` (test-clean of the same corpus) | 0 clips: LibriSpeech keeps its splits apart, and the 40 test speakers are absent from the 108 prepared readers |
| `librispeech-pc-en-train` | `librispeech-en` (test-clean of the same corpus) | 0 clips: LibriSpeech keeps its splits apart, and none of the 40 test speakers is among the 119 prepared readers |
| `pxcorpus-fr-train` | none: no evaluation set comes from this corpus | — |
| `ami-fr-speakers-en-train` | none: no evaluation set comes from this corpus | — |
| `nasa-en-train` | `nasa-en-test` (60 episodes reserved from the same listing) | 0 clips: the 2026-11-09 preparations predate the set; the fetcher skips the reserved episodes of every preparation made after the set exists, checked in `tests/test_data_nasa.py` |
| `granary-fr-train` | none: no evaluation set comes from this corpus | — |
| `icsi-en-train` | none: no evaluation set comes from this corpus | — |
| `notsofar-en-train` | `notsofar-en-eval` (the eval split of the same corpus) | 0 clips: the twelve eval aliases are absent from the twelve prepared train aliases, checked against the prepared eval manifest by the fetcher |
| `african-accented-fr-train` | `african-accented-fr-test` (test part of the same corpus) | 0 clips: the 12 test speakers are absent from the 190 prepared speakers, checked against the prepared test manifest by the fetcher |
| `peertube-fr-train` | none: no evaluation set comes from this instance | — |
| `peertube-edu-fr-train` | `peertube-edu-captions-fr` (the human-captioned talks of the same listing) | 1 talk: the training set (prepared 2026-11-09, before the evaluation set existed) holds one of the 140 human-captioned talks, with 73 of its clips; the evaluation fetcher leaves the talk out, and the train fetcher now skips the set's talks via `evaluation_groups` in every preparation it makes after the set exists, checked in `tests/test_data_peertube_edu_captions.py` |
| `debconf-en-train` | none: no evaluation set comes from PeerTube | — |
| `fosdem-en-train` | none: no evaluation set comes from a video tree | — |

No other source of the screening shares a corpus with an evaluation set: FLEURS
is not used for training, and the sources that are not evaluation sets at all
(Granary, PxCorpus, AMI, ICSI) publish no other split here. The dev split of
NOTSOFAR is still not read, so no prepared source overlaps it.

The voxpopuli check is the one that bites. The camera-ready way to break decision
0002 would be to train on French parliamentary speech and then score the same
people reading English: the exclusion is by speaker, and the
prepared French source holds 229 speakers, none of them one of the 27. The
session check goes further: 206 of the evaluation set's 307 plenary sessions
are in the prepared source too, so 480 of its 737 clips (1.6 h) and 1 956 of the
source's 12 000 (4.67 h) were recorded in the same sittings, rooms and
microphones. A session is the clip id's prefix before the language tag, the
population the prepared manifests
([`experiments/leakage-audit`](../../experiments/leakage-audit/README.md)).
That shares acoustics, not text — the languages and the utterances
are different — and whether it disqualifies those train clips is
[proposed decision 0014](../decisions/proposed/0014-voxpopuli-shared-sessions.md).

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
- **`voxpopuli-en-nonnative` is a test set by construction.** The accented
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
- **Accents are covered once.** `voxpopuli-en-nonnative` is English spoken by
  the Parliament's non-native speakers, drawn from the one accent the corpus
  labels `en_fr`; English with another accent, and French with a foreign accent,
  are not covered, and the label is the corpus's own claim rather than something
  checked here.
- The panels are samples of these test splits and were chosen by us, so absolute
  numbers describe the panels, not the whole datasets.
