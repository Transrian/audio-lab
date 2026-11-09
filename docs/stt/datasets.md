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
| `librispeech-en-train` | OpenSLR SLR12 | `train-clean-100.tar.gz` | sha256 of the archive (`src/audiolab/data/librispeech.py`) |
| `librispeech-pc-en-train` | OpenSLR SLR145 | `manifests.tar.gz` | sha256 of the archive, recorded in the code because OpenSLR publishes no checksum for it; the audio is `librispeech-en-train`'s own archive, checked against its sha256 (`src/audiolab/data/librispeech_pc.py`) |

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
- **Filters**: 0.5 s to 20 s of audio, text that is not empty, and audio that
  decodes at all — the point where a damaged file would otherwise show up. A clip
  the filters drop is counted in the fetch report and read past, so a source is
  read until the cap or its end, and a source with fewer usable clips than the cap
  is prepared with what it has.
- **Groups**: the speaker wherever the corpus publishes one, the recording
  otherwise, which `group_kind` records. A screening result is a confidence
  interval over groups like every other number here.
- **The text the source publishes**, which each card records. Casing and
  punctuation are not added back: three of the five sources are lowercased or
  uppercased and unpunctuated (MLS, SLR57, LibriSpeech), so their arms teach the
  model to drop what it already writes; VoxPopuli is cased and punctuated human
  text, and Granary is cased, punctuated model output that nobody verified. That
  difference is part of what the screening measures, so nothing was rewritten to
  make the arms look alike. Transport defects of a published transcript are the
  one exception, and the card of the source says which: SLR57 writes a space
  after every apostrophe (`j' ai`), which the training text drops because the
  model learns it and the scorer counts two words.
- **No overlap with the evaluation sets** ([decision
  0002](../decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)),
  below.

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
| `peertube-edu-fr-train` | fr | CC-BY / CC0-1.0 (per video, in the manifest) | listing | TBD | TBD | TBD | TBD | TBD |
| `debconf-en-train` | en | CC-BY (per video, in the manifest) | listing | TBD | TBD | TBD | TBD | TBD |
| `fosdem-en-train` | en | CC-BY-2.0-BE (per talk, in the manifest) | listing | TBD | TBD | TBD | TBD | TBD |
| `nasa-en-train` | en | public domain (US government work) | listing | TBD | TBD | TBD | TBD | TBD |

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

The sources below, like `peertube-fr-train`, are **talks** rather than corpora: a
pinned listing of videos, read in a seeded order, whose audio is labelled by two
speech models where they agree. Three modules hold that machinery for all of
them (`src/audiolab/data/talks.py`, `talk_listing.py` and `talk_clips.py`), and
each source is a listing module plus one entry in
`src/audiolab/data/talk_sources.py`, which says what the
listing must hold, in which language the talks are labelled, where a talk's audio
comes from and what a seeded selection spreads over. What is specific to a
source — its catalogue, its licence, its selection, its size — is in its card;
everything else is the same for the four and is described in the card of
`peertube-fr-train`. All of them depart from the rules above for the same three
reasons: no splits (a video platform has none), a seeded order instead of the
catalogue's, and model labels instead of a transcript somebody published.

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

### `nasa-en-train` — NASA's podcasts, labelled from NASA's own transcripts

This source is listed and downloaded like the talks above, but its labels are not
model output: NASA publishes a transcript with each episode, and that human text,
cased and punctuated, is the label. A model only says *when* each word was said.

- **Source**: five NASA shows, read from their RSS feeds
  (`https://www.nasa.gov/feeds/podcasts/<show>`): each item gives the episode's
  MP3 and the page on `www.nasa.gov/podcasts/` whose body holds the transcript.
  `audiolab data list-talks nasa-en-train` reads the feeds, fetches every episode
  page once (kept raw in `$AUDIOLAB_CACHE/nasa-en-train/pages`, never in the
  repository) and pins the episodes in
  [`nasa_listing.json`](../../src/audiolab/data/nasa_listing.json): show, title,
  page, MP3 and duration, no text. The listing of 2026-11-09 holds **774
  episodes, 523.0 h**, the median episode 38 minutes long:

  | show | episodes | hours |
  | --- | --- | --- |
  | Houston We Have a Podcast | 419 | 373.5 |
  | Small Steps, Giant Leaps | 150 | 65.3 |
  | NASA in Silicon Valley | 102 | 46.9 |
  | Gravity Assist | 96 | 35.9 |
  | NASA's Curious Universe | 7 | 1.4 |

- **What the listing leaves out**, from the 993 items of the five feeds, each
  rule in `src/audiolab/data/nasa.py`:
  - 50 items that link no page under `www.nasa.gov/podcasts/` (no link, or a page
    on `appel.nasa.gov`);
  - 20 repeats, an episode aired again being listed once, under its first airing.
    Two episodes are the same when they share a page, an MP3 file or a title once
    an "Encore" or a series' "Ep. N:" prefix and punctuation are set aside: the
    11 episodes of Houston We Have a Podcast's "Mars Ep. N" series, which air
    earlier episodes again, 3 "Encore" republications and 6 episodes posted
    twice. The feeds are read oldest first, so the copy kept is the first aired.
    3 more pages say their episode aired before ("This episode originally aired
    as episode 355", "Podcast Archive 2009 … originally aired on", "the
    rebroadcast of a conversation");
  - 48 pages without a transcript, all of NASA in Silicon Valley (its "Live"
    video streams and its short news pieces, whose page holds a caption or a
    link), a page counting as a transcript when it holds 60 words per minute of
    audio (people say about 150); and 3 pages that hold more than 250 words per
    minute, each with its transcript written twice (the median page holds 168,
    the fastest of the others 239);
  - 47 episodes that play recordings, 42 of them from NASA's Curious Universe: a
    transcript note or a speaker label names archival audio, a broadcast or a
    clip, an archive, an excerpt or an announcer (`[ARCHIVAL Apollo 11
    Broadcast: …]`, `APOLLO 8 ARCHIVAL:`, `[Clip of Sidedoor podcast begins]`,
    `[NBC NIGHTLINE's CHRIS WALLACE: …]`, `[Meteorologist audio: …]`,
    `[John F. Kennedy Archive: …]`, `Voyager excerpt:`, `Launch announcer:`,
    `Unidentified launch voice 1:`). The episode is left out whole: some of those
    recordings are another broadcaster's, the page does not say whose each one
    is, and a transcript sometimes writes their words as lines of the show
    (`Archival: 5…4…3…2…1 and liftoff…`), which a clip would then carry;
  - 1 transcript that writes its notes as plain words (below, under *The
    transcript*);
  - 2 pages with a copyright notice and 45 pages crediting music of a third
    party (below).
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
    "copyright" in passing. The two episodes left out play archive clips "used
    courtesy of the New Mexico Public Media Collection"
    (`sun-series-soaring-toward-the-sun`) and audio of a conversation "provided
    by Explore Mars and the Planetary Society"
    (`gravity-assist-what-if-we-found-life-on-mars`).
  - Each preparation fetches every page it reads again and applies the
    listing's rules to it, so an episode whose page took a notice, or lost its
    transcript, since the listing is skipped and counted.
  - An episode whose page credits music of a third party is left out whole:
    the series' library ("Music for the series comes from Universal Production
    Music"), its theme ("composed by … of SYSTEM Sounds") or a note naming a
    piece and its composer (`[Song: Ants Instrumental by Taylor Welsh]`). Such
    music plays under the speech where the transcript does not note it, and its
    licence is unknown, which D-019 does not train on. That costs 45 episodes,
    17.2 h, all of NASA's Curious Universe (the episodes no other rule leaves
    out; the show keeps 7, 1.4 h). Elsewhere, clips are kept away
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
  cased, punctuated, digits kept. A transcript is near-verbatim, not verbatim —
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
  shuffle (`SEED`) spread over the show, so a first cap covers the five shows
  rather than Houston We Have a Podcast, which holds two thirds of the hours. The
  audio is the MP3 the listing pinned, converted to 16 kHz mono WAV.
- **Prepared audio**: before the labelling run, the first 65 h of the reading
  order of the listing of 2026-11-09 — 128 episodes, 66.0 h: 30 or 31 per
  show, and the 7 left of NASA's Curious Universe — were
  downloaded (`audiolab data fetch-audio nasa-en-train --hours 65`) and kept as
  16 kHz mono WAV, 7.0 GiB in the pipeline's cache (`$AUDIOLAB_CACHE`,
  `.cache/audiolab` by default, git-ignored; the MP3 is deleted once its sound is
  out). Every episode of that selection arrived.
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
| `voxpopuli-fr-train` | `voxpopuli-en-nonnative` (accent `en_fr`) | 896 clips, the clips of the 27 speakers of that set; none of them is in the prepared French source |
| `mls-fr-train` | `mls-fr` (test split of the same corpus) | 0 clips: MLS keeps the speakers of its splits apart, and the 18 test speakers are absent from the 138 prepared readers |
| `librispeech-en-train` | `librispeech-en` (test-clean of the same corpus) | 0 clips: LibriSpeech keeps its splits apart, and the 40 test speakers are absent from the 108 prepared readers |
| `librispeech-pc-en-train` | `librispeech-en` (test-clean of the same corpus) | 0 clips: LibriSpeech keeps its splits apart, and none of the 40 test speakers is among the 119 prepared readers |
| `pxcorpus-fr-train` | none: no evaluation set comes from this corpus | — |
| `ami-fr-speakers-en-train` | none: no evaluation set comes from this corpus | — |
| `nasa-en-train` | none: no evaluation set comes from this corpus | — |
| `granary-fr-train` | none: no evaluation set comes from this corpus | — |
| `african-accented-fr-train` | none: no evaluation set comes from this corpus | — |
| `peertube-edu-fr-train` | none: no evaluation set comes from PeerTube | — |
| `debconf-en-train` | none: no evaluation set comes from PeerTube | — |
| `fosdem-en-train` | none: no evaluation set comes from a video tree | — |

No other source of the screening shares a corpus with an evaluation set: FLEURS
is not used for training, and the two French sources that are not evaluation sets
at all (SLR57, Granary) publish no other split here.

The voxpopuli check is the one that bites. The camera-ready way to break decision
0002 would be to train on French parliamentary speech and then score the same
people reading English: the exclusion is by speaker, and the
prepared French source holds 229 speakers, none of them one of the 27.
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
