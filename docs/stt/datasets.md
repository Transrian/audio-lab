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
  punctuation are not added back: three sources are lowercased or uppercased and
  unpunctuated (MLS, SLR57, LibriSpeech), so their arms teach the model to drop
  what it already writes; VoxPopuli, AMI, ICSI and NOTSOFAR are cased and
  punctuated human text, and Granary is cased, punctuated model output that
  nobody verified. That difference is part of what the screening measures, so
  nothing was rewritten to
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
| `icsi-en-train` | en | CC-BY-4.0 | mrt | 9 393 | 52 | 6.3 | 0.7 GiB | 3.6 GiB of signals, and 12 GiB of decoded WAV stays in the cache |
| `notsofar-en-train` | en | CC-BY-4.0 | 240825.1_train | 2 372 | 12 | 1.5 | 0.2 GiB | 0.8 GiB |

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
  the annotators' own, cased and punctuated, with their spellings (`O_K`,
  `Mm-hmm`, acronyms split by underscores as written) and their marks of a false
  start (`how much w- how easy`). The annotation elements between the words are
  taken out: `VocalSound` (29 779 of them), `NonVocalSound` (3 708), `Comment`
  (12 283) and `Pause` (7 213) carry no words to transcribe, while `Emphasis`
  (24 612), `Uncertain`, `Pronounce` and `Foreign` hold words the speaker did say
  and are kept as text. A segment that is nothing but a vocal sound cleans to no
  text, and such a segment is not a clip.
- **Group**: the speaker (`group_kind: speaker`): the 52 participant ids of the
  corpus, each of them a group of its own. A clip is sliced from the close
  microphone the segment's own preamble gives its speaker
  (`<Participant Name="fe066" Channel="chan0"/>`), so two participants of one
  meeting never share a clip. A participant the preamble gives no channel to
  (51 slots) is one the corpus says was unmiked — heard on other people's
  microphones — and each of their 219 segments says `CloseMic="false"`; a segment
  that claims the close microphone and has no channel stops the preparation
  instead.
- **What the source is**: 75 meetings and 129 813 segments. 1 413 of those
  segments say the close microphone did not carry the speech, and 717 of them
  name no participant either — a stretch of room noise the annotators marked —
  which leaves 108 323 segments that are speech one of the microphones this
  source reads did carry. 20 077 of the 129 813 clean to no text (a laugh, a
  cough, an annotator's remark), so the corpus for this source is the 108 323
  that are left. The per-speaker share of 200 clips is what holds the
  preparation there: 8.7 % of those segments become clips, 9 393 of them from 45
  of the 75 meetings, 6.3 h, median 1.7 s, and one segment longer than 20 s. The
  42 speakers that reach their share contribute 200 clips each and the 10
  smaller ones contribute what they have.
- **What a cap covers**: fewer clips than the cap, and the per-group share is
  what holds it there. The corpus has 52 participants for this source, so the
  share (200 clips of a group) allows 10 400 clips at most, below the 12 000 of
  the cap: every arm of the screening is capped by the same rule, and this source
  cannot reach 12 000 clips under it. Its config sets `clips: 9393` next to the
  reason.
- **Decoding**: the corpus's signals are shorten-compressed SPHERE and the audio
  readers of this project do not decode them, so `ffmpeg` decodes every channel a
  preparation reads — the only external program any preparation runs — and the
  samples it produced are checked against the count and the checksum the SPHERE
  header of the file itself carries, which is the only checksum the corpus
  offers. `src/audiolab/data/icsi.py` says why the annotation archive is pinned
  by sha256 instead. The 105 channels a full preparation reads are 3.6 GiB of
  downloads, the decoded WAV they leave in the cache is another 12 GiB, and the
  prepared set is 0.7 GiB.
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
  markup taken out: `<ST/>` (a pause), `<FILL/>`, `<FILLlaugh/>`, `<BA/>`,
  `<UNKNOWN/>`, `<ISSUE/>` mark something that is not a word, and
  `<PName> Peter </PName>` wraps a name, which stays as text. The 980 rows of the
  split that were nothing but markup clean to no text, and the clip filter drops
  them for that. The text is
  the corpus's own: 582 of the 2 372 clips start with a lowercase letter and 840
  do not end with a sentence-final mark, because the annotators transcribed the
  utterance as it came, and the preparation rewrites none of it. The corpus pads
  its tags with spaces, so removing one can leave the padding in front of a comma
  or a period (`So Sophie , what`); punctuation is joined to the word before it,
  as the AMI source does, and the words themselves are untouched.
- **Group**: the speaker (`group_kind: speaker`): the alias the corpus gives the
  participant (`Peter`, `Sophie`, `Bert`). 12 aliases over the 72 meetings, and
  each of them is one person, recognizable across the meetings of the corpus,
  which is what makes them groups; a clip is sliced from that participant's own
  close-talking recording, so two participants never share one.
- **What the source is**: the recorded meetings of the challenge — meetings of
  about six minutes, four to eight attendees in one of 30 conference rooms,
  transcribed by hand — not the simulated thousand hours published in the same
  repository, and not the `dev_set` or the `eval_set`, which are not read for
  anything. The challenge's README counts 237 meetings of it and the dataset's
  page on Hugging Face 315; the revision this source pins holds 72 of them in
  its train split and 12 speakers, and the challenge says 10 of those 12 are also
  in the dev set of the campaign, which is one more reason not to read it here.
  The split holds 18 178 utterances: 980 of them are markup and nothing else,
  317 last less than 0.5 s, and the rest fall in the window the clips are cut
  from — the longest utterance of the split is 8.7 s, so the 20 s rule never
  applies here. The share of a group holds the preparation: 2 372 clips from 32
  of the 72 meetings, 1.5 h, median 1.7 s, and 14 509 rows left out because
  their speaker had its 200 clips already. Every alias reaches its 200 clips
  except Walter, who has 172.
- **What a cap covers**: fewer clips than the cap, and the per-group share is
  what holds it there. The split of 72 meetings names 12 participants, so the
  share (200 clips of a group) allows 2 400 clips at most, far below the 12 000
  of the cap: this source cannot reach it under the rule. Its config sets
  `clips: 2372` next to the reason.
- **Why it is here**: meetings that were really held, a few people around a
  table, each with a close-talking microphone, transcribed by hand — the same
  register as AMI and ICSI and a third corpus of it, recorded twenty years later
  and in other rooms. Like ICSI, it is not in the training list the base model's
  card cites.

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
| `granary-fr-train` | none: no evaluation set comes from this corpus | — |
| `icsi-en-train` | none: no evaluation set comes from this corpus | — |
| `notsofar-en-train` | none: no evaluation set comes from this corpus, and the dev and eval splits the challenge publishes are not read either | — |
| `african-accented-fr-train` | none: no evaluation set comes from this corpus | — |

No other source of the screening shares a corpus with an evaluation set: FLEURS
is not used for training, and the sources that are not evaluation sets at all
(SLR57, Granary, PxCorpus, AMI, ICSI) publish no other split here. The recordings
of NOTSOFAR are scored by no evaluation set of this campaign either: the corpus
is one of the challenge's, and its dev and eval splits were left alone.

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
