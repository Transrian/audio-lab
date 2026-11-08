# Data sources

The register of every data source considered for the speech-to-text work: what it
is, where it stands, its license and why. It is required by
[D-018](../owner/directives.md), which asks for the status, the license and the
reason of every source considered, whether it is used here or not.

This file is the decision log; it is not the description of the datasets that
`audiolab data fetch` prepares. Those have one card each in
[datasets.md](datasets.md), and the register links to them instead of repeating
them.

**Editing it.** Add a source, or move one to another status, with the date and
the reason. An entry may be rewritten, but the reason a source was abandoned or
rejected stays readable here: a source that comes back keeps the text of why it
left. Before any acquisition, check the license on the primary source — for the
audio *and* for the transcripts — and check the source against the frozen
evaluation sets.

License dates below are the dates the earlier private project recorded them, and
the licenses were not verified again here: recheck before acquisition. Nothing
here is legal advice, and dataset cards change.

## Statuses

Statuses describe *this* repository: "used" means trained on here, and a source
the earlier project trained on is a candidate until this repository trains on it.

| Status | Meaning |
| --- | --- |
| `used` | Trained on here. |
| `evaluation only` | Used here to evaluate models; never to train. |
| `candidate` | Researched, not acquired here. |
| `abandoned` | Tried, then withdrawn, here or in the earlier project. |
| `rejected` | Never used, by decision. |

## Selection criteria

Only what the repository settles is used here as a criterion.

- **French first, English kept, dictation as the main use**
  ([D-009](../owner/directives.md)): French is the priority, English is kept,
  other languages are out of scope. The main target is a French developer
  dictating to coding agents: French with English words and technical terms,
  FR↔EN switches inside a sentence, passages entirely in English, and English
  with a French accent. Sources that cover these cases rank first.
- **No audio in the repository** ([D-013](../owner/directives.md)): a dataset is
  prepared by a script into `$AUDIOLAB_DATA` and never committed. That avoids
  redistributing it, not the need for the right to use it.
- **Licenses that can be trained on** ([D-019](../owner/directives.md)):
  public domain, CC0, CC-BY (any version), Apache-2.0, MIT or Licence Ouverte,
  for the audio and for the text alike. Share-alike, non-commercial,
  no-derivatives and unknown licenses are not trained on.
- **Non-commercial data may be used for testing only**
  ([decision 0009](../decisions/accepted/0009-non-commercial-datasets-for-testing-only.md)):
  a dataset restricted to non-commercial use may be used to evaluate models and
  to publish the scores, but never to train a model meant for publication, and
  the repository never redistributes it. Sources concerned are marked *testing
  only* below.
- **The evaluation sets are frozen** ([decision 0002](../decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)):
  they are fixed before a training campaign starts, so a new source is checked
  against them before it is acquired.

## Prepared 2026-11-08 for the per-source screening

The per-source screening fine-tunes Parakeet TDT 0.6B v3 once per training
source, with the same number of clips and the same number of updates, to see what
each source brings by itself. On 2026-11-08 the training sources of five corpora
were prepared for it with `audiolab data fetch`, as the datasets whose name ends
in `-train`; [Training sources](datasets.md#training-sources) has one card per
dataset.

The screening then trained on five of them, one at a time (2026-11-08), so those
five are `used` below; its results are in [per-source screening](screening.md).
GLOBE V2 was not prepared (no accent labels to select French speakers by), and
its row stays a candidate.

Later the same day, five more were prepared and are not screened yet:
`librispeech-pc-en-train`, `pxcorpus-fr-train`, `ami-fr-speakers-en-train`,
`icsi-en-train` and `notsofar-en-train` (rows below); and the text of `african-accented-fr-train` was cleaned, so its
screening result describes the uncleaned text.

## Used — trained on here

Trained on in the per-source screening of 2026-11-08: 10 000 clips and 2 000
updates per source, one source per run. The numbers are in
[per-source screening](screening.md); the card of each prepared dataset is under
[training sources](datasets.md#training-sources). "Screening" below is what that
source did alone, against the base model.

| Source | Language | License | Reason / note |
| --- | --- | --- | --- |
| [Granary / YODAS FR](https://huggingface.co/datasets/espnet/yodas-granary) (`granary-fr-train`) | fr | audio CC-BY-3.0, labels CC-BY-4.0 (2026-11-01) | Pseudo-labelled YouTube speech, the widest variety found. Earlier private project (2026-10-28): built a filtered 100 h subset and trained on it; its audit removed duplicate audio, repetition loops and hallucination patterns; not reproduced here. Screening: the only source with no measurable harm; a small gain on Québec code-switching, and one on read French whose interval is approximate. The base model was trained on Granary, so it is a candidate anchor for a mix rather than a source of new gains ([screening](screening.md#what-the-base-model-already-saw)). |
| [VoxPopuli FR](https://github.com/facebookresearch/voxpopuli) (`voxpopuli-fr-train`) | fr | data CC0, code and models CC-BY-NC (not used) (2026-11-01) | European Parliament speech with human-edited transcripts. Earlier private project (2026-11-01): trained on the train split; not reproduced here. Screening: neutral to slightly better on French, but 4.7 points worse on English with a French accent. Its English `en_accented` set is used for evaluation, with its speakers kept out of this train split. |
| [MLS FR](https://www.openslr.org/94/) (`mls-fr-train`) | fr | CC-BY-4.0 (2026-11-05) | Read audiobooks. Earlier private project (2026-11-01): measured to hurt in training — a 10 % share of a mix degraded one panel by 8–14 % relative — and abandoned there for good; not reproduced here. Trained on again here alone, to test that finding. Screening: hurts the main French panel and accented English, and its lowercase, unpunctuated text doubles the formatted WER on fleurs-fr — the earlier finding holds. Its **test** split stays an evaluation set ([evaluation only](#evaluation-only--used-here-to-evaluate)). |
| [African Accented French](https://www.openslr.org/57) (SLR57, `african-accented-fr-train`) | fr | Apache-2.0 (2026-11-05) | French spoken with Cameroonian, Chadian, Congolese, Gabonese and Nigerian accents, otherwise almost absent from open French data. Screening: its transcripts put a space after apostrophes (`j' ai`), which the model learns; with that removed it still costs on code-switched French and accented English. Its apostrophes and spacing were cleaned on 2026-11-08 and the dataset prepared again; casing and punctuation are still missing, since restoring them would need a model. |
| [LibriSpeech](https://www.openslr.org/12) train (`librispeech-en-train`) | en | CC-BY-4.0 (2026-11-08) | English read audiobooks, as the English guard of a French mix. Screening: its all-uppercase transcripts teach the model to write in capitals, so it is unusable as prepared; the cased and punctuated version of the same audio is `librispeech-pc-en-train`. |

## Evaluation only — used here to evaluate

These are the sets this repository scores models on today. Each has a card in
[datasets.md](datasets.md) with its source, its license, its pinned revision and
its panels; nothing about them is repeated here.

| Source | Language | License | Reason / note |
| --- | --- | --- | --- |
| [`fleurs-fr`](datasets.md#fleurs-fr-fleurs-en--read-wikipedia-sentences) | fr | CC-BY-4.0 | Test split only: short read speech, the main French panel. |
| [`fleurs-en`](datasets.md#fleurs-fr-fleurs-en--read-wikipedia-sentences) | en | CC-BY-4.0 | Same corpus, English side; the everyday English panel. |
| [`mls-fr`](datasets.md#mls-fr--read-audiobook-french) | fr | CC-BY-4.0 | Test split only. The earlier project trained on this dataset and dropped it; its train split is used here in the screening (see [Used](#used--trained-on-here)); here it measures read audiobook French, and no number is corrected for the overlap with training data. |
| [`librispeech-en`](datasets.md#librispeech-en--read-audiobook-english) | en | CC-BY-4.0 | The standard English read-speech panel. |
| [ESC-50](datasets.md#nonspeech--the-hallucination-battery) | none | CC-BY-NC-3.0 | Everyday sounds inside `nonspeech`. Non-commercial: **testing only** per [decision 0009](../decisions/accepted/0009-non-commercial-datasets-for-testing-only.md). |
| [Generated silence and noise](datasets.md#nonspeech--the-hallucination-battery) | none | none: generated here from a fixed seed | There is no license to honour and no redistribution question. |
| [`voxpopuli-en-nonnative`](datasets.md#voxpopuli-en-nonnative--english-spoken-by-non-native-speakers-in-the-european-parliament) | en | CC0-1.0, data only (2026-11-05) | The `en_accented` config of VoxPopuli filtered to the accent the corpus labels `en_fr`: English read in the European Parliament by non-native speakers, with human transcripts and never trained on. Added 2026-11-08 for the accented English of [D-009](../owner/directives.md), renamed 2026-11-08 from `voxpopuli-en-french-accent` because matching its speakers to the Parliament's own member list showed the label did not hold; a test split by construction. |
| [`servicenow-codeswitched-en-fr`](datasets.md#servicenow-codeswitched-en-fr-and-servicenow-codeswitched-en-fr-ca--synthetic-code-switching-in-it-support) | fr, en | **no licence on the card** (2026-11-05) | French framing English words inside one sentence, over the IT, support and HR terms of the target use. The audio is synthetic (two text-to-speech voices per configuration). With no licence it is used for evaluation only: never trained on, never redistributed ([proposed decision 0011](../decisions/proposed/0011-unlicensed-datasets-for-private-evaluation.md)). Owner-approved for evaluation on 2026-11-08 (proposed decision 0011). |
| [`servicenow-codeswitched-en-fr-ca`](datasets.md#servicenow-codeswitched-en-fr-and-servicenow-codeswitched-en-fr-ca--synthetic-code-switching-in-it-support) | fr, en | **no licence on the card** (2026-11-05) | The same corpus spoken in Québec French, prepared as its own dataset so its numbers read apart. Same conditions as the row above. |
| [`cs-fleurs-fr-en`](datasets.md#cs-fleurs-fr-en--code-switched-fleurs-sentences) | fr, en | **CC-BY-NC-4.0** (2026-11-05) | FLEURS sentences with part of their words replaced by the other language, read by a person and rendered by synthetic voices. Non-commercial: **testing only** per [decision 0009](../decisions/accepted/0009-non-commercial-datasets-for-testing-only.md). Added 2026-11-08 for the switches inside a sentence of [D-009](../owner/directives.md). |

## Candidates — researched, not acquired here

Nothing here is downloaded. The priority order and the four groups are the
earlier project's (a search run in four angles: French variety, accents and
public sector, code-switching and technical vocabulary, permissive text to
speech), kept because it follows the same target use (D-009). Licenses are as
recorded there on the dates given, and classed as recorded: **publishable**
(CC0, CC-BY, Apache, MIT, Licence Ouverte), **SA**, **NC** or **unknown**.

### Candidates the earlier project already trained on

Trained on there, candidates here: they are the closest to being acquired, and
the note says what that project did and found. Granary and VoxPopuli FR, which
stood here, are `used` since 2026-11-08 ([Used](#used--trained-on-here)).

| Source | Language | License | Reason / note |
| --- | --- | --- | --- |
| [Common Voice Spontaneous Speech 5.0 FR](https://mozilladatacollective.com/datasets/cmu5z135s00ginq072mliz1as) | fr | CC0 + Mozilla Data Collective terms | Spontaneous speech. Earlier private project (2026-11-01): it could use very few clips, because almost every speaker overlapped the protected evaluation pool. Not reproduced here. The MDC terms forbid re-identification and redistribution; treat as publishable-with-conditions. |
| Common Voice Spontaneous Speech 5.0 EN | en | CC0 + Mozilla Data Collective terms | English guard in a French mix. Earlier private project (2026-11-01): trained on a small capped share of it for that purpose. Not reproduced here. |
| [MUSAN](https://www.openslr.org/17/) `noise/` | none | CC-BY-4.0 (2026-11-01) | Noise augmentation. Earlier private project: kept half of the noise files for training and reserved the other half, so a later acquisition must not reuse the reserved half. Not reproduced here. |

The owner's private recordings: never published, never in this repository
([D-013](../owner/directives.md)).

### French variety

| # | Source | Language | License | Reason / note |
| --- | --- | --- | --- | --- |
| 1 | [Emilia-YODAS](https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/README.md) FR | fr | CC-BY-4.0 (2026-11-05) | Talk shows, interviews, debates and sport: the closest legal audio to "film" speech. Pseudo-labelled with Whisper; same YODAS2 origin as Granary, so deduplicate by video. Hugging Face access has conditions the owner must accept. |
| 1 bis | French **CC-BY** YouTube videos published after the 2023 YODAS collection | fr | CC-BY chosen by the uploader (2026-11-05) | Same recipe as Granary on recent content, but downloading is legally contested: YouTube's terms forbid automated access unless the law allows it, and getting the audio may mean circumventing a technical measure. Check the licence video by video; label with open ASR models only. |
| 2 | [Conférences Pierre Mendès France](https://www.data.gouv.fr/fr/datasets/transcriptionsxml-audiomp3-mefr-ccpmf-2012-2020-zip/) (2012–2020) | fr | Licence Ouverte 2.0, publishable (2026-11-05) | Conferences and round tables on economy, digital and administration: a register close to dictation. The data.gouv.fr exports point at a download site that no longer serves the set, so the archive must be found again, and the XML must be checked to be faithful, alignable transcripts. Checked 2026-11-08: the download answers 404 and no archived copy was found. |
| 3 | [« Libre à vous ! »](https://libreavous.org) (April, radio Cause Commune) | fr | audio **SA** (CC-BY-SA 2.0+ / Art Libre 1.3+ / GFDL 1.3+); transcripts on [librealire.org](https://www.librealire.org) under the "Verbatim" licence (2026-11-05) | Free-software radio: English loanwords and IT vocabulary in conversational French, with human transcripts. Music has its own licences and must be cut; the transcripts may be copied but not republished modified. |
| 3 bis | French **PeerTube** via [SepiaSearch](https://sepiasearch.org) | fr | per video, declared by the uploader (BY, BY-SA or public domain) (2026-11-05) | Conference recordings from the free-software world, so technical French, plus public-sector instances. A declared licence is not a real one: the sample checked contained a badly labelled anime dump, so instances and channels need a whitelist, and the language too. Pseudo-labels as in 1 bis. Checked 2026-11-08: the [Capitole du Libre](https://videos.capitoledulibre.org) instance, run by the conference itself, publishes 519 French talks (about 379 h; a recount on 2026-11-08 found 569 CC-BY videos, about 405 h, counting those with no language set) with the licence "Attribution" (CC-BY) on every video, and no captions; it is being prepared as `peertube-fr-train`, labelled with two ASR models other than Parakeet. On SepiaSearch, French Science & Tech has 3 210 CC-BY and 605 CC0 videos (2026-11-08). |
| 4 | [African Accented French](https://www.openslr.org/57) (SLR57) | fr | Apache-2.0 (2026-11-05) | **Moved to `used`** on 2026-11-08: see [Used](#used--trained-on-here). |
| 5 | [CEREALES](https://www.isca-archive.org/interspeech_2025/maison25_interspeech.html) (Interspeech 2025) | fr | **unknown**: the paper announces an OpenSLR release and gives no licence (2026-11-05) | Québec accent and spontaneous speech with human stenography. Absent from OpenSLR and Hugging Face when checked, so the authors must be asked for distribution and licence before anything else. |
| 6 | [CommissionsQC](https://www.isca-archive.org/interspeech_2025/serrand25_interspeech.html) | fr | access on request, research only, no voice synthesis or imitation ([arXiv:2508.21193](https://arxiv.org/abs/2508.21193)) | Long Québec public-commission recordings, reverberant rooms, human stenography. In practice non-commercial with no publication, so a written request would have to state the ASR use. |
| 7 | Common Voice **scripted** FR, official [MDC release](https://github.com/common-voice/cv-dataset) | fr | CC0 + Mozilla Data Collective terms | Thousands of voices and microphones, but read prompts. Use the train split only, and exclude the speakers and sentences of the protected sets. |
| 9 | [SUMM-RE](https://huggingface.co/datasets/linagora/SUMM-RE) (Linagora / AMU) | fr | **CC-BY-SA-4.0**, SA (2026-11-05) | Meetings with three or four participants, one microphone each: a meeting register no other source covers. Only the dev and test parts are human transcripts; the train part is Whisper output, and the whole set is large, so take the transcribed part or pseudo-label. |
| 10 | ESTER 1/2, EPAC, MEDIA (French evaluation corpora, via ELDA) | fr | **unknown**: several licence levels, with and without commercial use (unverified) | Radio, including Moroccan and African stations in ESTER 2: Maghreb and African accents. The exact conditions of the free level must be read, and the transcripts for the African stations are rough ones. |
| 12 | More Granary FR (D300) and [Granary-v2](https://huggingface.co/datasets/nvidia/Granary-v2) | fr | v1 as above; v2 labels **CC-BY-SA-4.0 + OpenMDW-1.1** (2026-11-05) | v1 adds volume with no new pipeline, but the same source, so little new variety; v2's French comes mostly from audiobooks, pseudo-labelled VoxPopuli and Common Voice, so little gain is expected. |
| 13 | [MOSEL](https://huggingface.co/datasets/FBK-MT/mosel) (FBK) | fr | pseudo-labels CC-BY-4.0; the underlying audio is untranscribed VoxPopuli and LibriLight (2026-11-05) | Volume of Whisper pseudo-labels over parliament and audiobooks: low priority, because it repeats sources already listed. |
| 14 | ALIPE (Université Clermont-Auvergne) | fr | CC-BY-SA-4.0 in a secondary list, **not verified** on the primary source | Parent–child conversations, the only French children's corpus found. Licence, access and parental consent all have to be checked before any use. |
| 15 | Channel augmentation: telephone and video-call codecs (G.711, AMR, low-bitrate Opus) + [RIRS_NOISES](https://www.openslr.org/28/) | none | ffmpeg; RIRS is Apache-2.0 (2026-11-05) | Variety of microphone, channel and room. No open call-centre corpus existed when checked. RIRS contains MUSAN material, so deduplicate against it and keep the half reserved in the earlier project free. |
| 16 | Recordings of consenting relatives | fr | written consent of each speaker | Real ages, accents, rooms and microphones. No third party is recorded without agreement, and the recordings stay out of the repository (D-013). |

Found 2026-11-08, after the screening showed that the base model had already
trained on the French sources tried (its French is Granary, Common Voice, MLS
and FLEURS):

| # | Source | Language | License | Reason / note |
| --- | --- | --- | --- | --- |
| 22 | [apps.education.fr](https://apps.education.fr) PeerTube instances (the French education ministry's) | fr | CC-BY or public domain, declared per video (2026-11-08) | Teachers on digital and technical topics: many voices and microphones, not on YouTube. Counted through the API: numerique-educatif 442 h CC-BY + 52 h public domain; sciences-technologies about 600 h (estimated, the API failed after 340 h); institutionnel 108 h. Whitelist the channels and cut the music. Pseudo-labels. |
| 23 | Other French PeerTube instances: peertube.openstreetmap.fr (221 h, CC-BY), videos-libr.es (270 h, public domain), and smaller ones (interhop, framatube, lescommuns, lqdn, aquilenet: about 130 h) | fr | CC-BY or public domain per video (2026-11-08) | Free-software, mapping and health-IT talks and tutorials; part of OpenStreetMap's is English. Same pipeline as Capitole du Libre. Pas Sage en Seine is all CC-BY-SA (rejected). |
| 17 | [European Commission Audiovisual Service](https://audiovisual.ec.europa.eu): midday briefings and press conferences | fr, en | the conditions of use (updated 2026-04-09, read 2026-11-08): "content owned by the EU on this website is licensed under […] CC BY 4.0" unless an item says otherwise; third-party rights and identifiable private individuals may need clearing. The per-item `ccBy` field is set even on agency items, so the filter is the copyright holder ("European Union" or "European Communities") | 4 642 complete midday briefings 2004–2026 (about 2 100 h) and about 510 complete press conferences in 2025 (about 240 h), listed by a public API without login; audio-only floor, English and French tracks since 2024, MP3 before. The floor is mostly English (two samples: 5–15 % French), and its language can be told without a model, since an interpretation track carries the floor sound when the speaker uses its language. The French interpretation track is about 400 h of professional French. No transcripts of the briefings, so our labels; press conferences have their opening remarks as text. |
| 18 | [PxCorpus](https://zenodo.org/records/10080490) | fr | **CC-BY-4.0** (Zenodo, 2026-11-08) | About 4 h of spoken drug prescriptions with human-checked transcripts: a dictation register with its own terminology. Prepared as `pxcorpus-fr-train` (2026-11-08): 1 912 clips, 3.7 h, 54 speakers named by its metadata (its readme says 55); far fewer than the screening's 10 000 clips. |
| 19 | [SIWIS French Speech Synthesis Database](https://datashare.ed.ac.uk/handle/10283/2353) | fr | **CC-BY-4.0** (README, 2026-11-08) | More than 10 h of read text from one professional speaker: clean, but one voice and no target-use register. At most a clean share of a mix. |
| 20 | [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14) fr-FR | fr | CC-BY, no version on the card (2026-11-08) | Spontaneous telephone requests to a bank, 8 kHz, about an hour: too small and the wrong channel for more than a test. |
| 21 | [VOA Afrique](https://www.voaafrique.com/p/5739.html) | fr | **public domain** for "material produced exclusively by the Voice of America"; agency material (AP is named) and licensed material are not (2026-11-08) | African French news broadcasts, 2015 to March 2025, several thousand hours with heavy repetition between editions; its feeds have been frozen since March 2025. No transcripts, the agency share cannot be measured without a model, and the music shows carry commercial music. The weakest of the candidates; an owner decision first. |

Leads not verified yet, to look at before classifying them: the Québec National
Assembly (Crown copyright presumed), the Belgian Chamber, and corpora of elderly
speakers (none found).

### English technical and French-accented English

| # | Source | Language | License | Reason / note |
| --- | --- | --- | --- | --- |
| 1 | [FOSDEM](https://video.fosdem.org) | en | **CC-BY-2.0-BE**, publishable; the slides are not covered (2026-11-05) | Developer conferences: real technical vocabulary and varied English accents, French ones included. Its schedules list 5 689 h of talks from 2012 to 2026 (fewer were recorded), served as plain files by year and room. No human transcripts — the subtitles are machine output with no guarantee — so labels would be ours, and the room audio has an audience. |
| 2 | [GLOBE V2](https://huggingface.co/datasets/MushanW/GLOBE_V2) | en | **CC0** (2026-11-05) | Acoustic adaptation to English spoken by French speakers, which is the case this project cares about. It derives from Common Voice 14, so its speakers and sentences must be excluded from the protected sets, and it is read speech, not technical. **Inspected 2026-11-08 and not prepared**: its accent labels are free text, not a vocabulary — five of its 164 train shards (3 % of the split) hold four rows that mention French at all, none of them as an accent, so a French-accented subset cannot be built from it without reading the whole train split (164 shards, 69.7 GiB). Taking English generally was possible and was not done: it would have repeated the LibriSpeech source in the same screening. A full scan of its accent column (542 261 rows, 2026-11-08) found 50 rows of "minor french accent": not worth preparing. |
| 3 | [media.ccc.de](https://media.ccc.de) | en (+ de) | per video, typically CC-BY-4.0, announced in the recording | Hacker, security and systems conferences. Counted 2026-11-08 through its API: 1 074 English talks (664 h) state CC-BY-4.0 in their description; about 5 550 h of English state nothing there (the licence is only in the video intro, so unverified per file), and 287 h are CC-BY-SA, excluded. Subtitles exist for about 5 % of talks, made with Whisper and human correction; their licence is not verified. Exclude the German ones. |
| 4 | [Granary / YODAS EN](https://huggingface.co/datasets/espnet/yodas-granary) | en | as the French side; the English side to recheck (2026-11-05) | English volume that can be filtered by technical vocabulary, through the pipeline already worked out for French. Pseudo-labels. |
| 5 | [DebConf](https://meetings-archive.debian.net) | en | a permissive licence in the archive's `pub/debian-meetings/LICENSE` for the team's recordings since DebConf6; the Debian PeerTube marks them CC-BY (2026-11-08) | Debian and free-software conferences: at least 1 485 h labelled CC-BY on peertube.debian.social, about 293 h of them Brazilian Portuguese to drop. No transcripts. |
| 6 | [AMI](https://groups.inf.ed.ac.uk/ami/corpus/), the French-L1 participants | en | **CC-BY-4.0** for the corpus and its annotations ([licence page](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml), 2026-11-08); this register recorded it as NC-SA until then, which was wrong | Meetings in English: 41 participants give French as their first language, about 10.8 h of their manually transcribed speech, cased and punctuated, one headset microphone each. The base model trained on AMI. Prepared as `ami-fr-speakers-en-train` (2026-11-08): only 20 of the 41 appear in a meeting with recordings, so 3 971 clips, 4.3 h. |
| 7 | [LibriSpeech-PC](https://www.openslr.org/145/) | en | **CC-BY-4.0**, text and audio (2026-11-08) | LibriSpeech with its casing and punctuation restored: the English guard of a mix, in place of the uppercase LibriSpeech text that failed the screening. The base model very likely trained on it. Prepared as `librispeech-pc-en-train` (2026-11-08): 12 000 clips, 42.8 h, 119 speakers, the evaluation speakers left out. |
| 9 | [NASA podcasts](https://www.nasa.gov/podcasts/) (Houston We Have a Podcast and five other shows) | en | **US government work**, "generally not subject to copyright"; third-party material is "identified as copyright protected" ([media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/), 2026-11-08), and none was found on the sampled pages | Conversational technical English: about 660 h in the feeds (Houston We Have a Podcast 447 episodes, 400 h), about 620 h with a transcript on the episode page — near-verbatim (repetitions kept, few fillers), with speaker lines and bracketed notes to drop. Labels would be the human text, aligned to the audio with the Whisper word timestamps; an estimated 350–450 h survive the alignment. Repeat episodes ("Encore") to remove. Public domain is allowed ([D-019](../owner/directives.md)). |
| 10 | [ICSI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/icsi/) | en | **CC-BY-4.0** (2026-11-08) | About 70 h of real research meetings with human transcripts. The base model trained on AMI; ICSI is not cited. Prepared as `icsi-en-train` (2026-11-08): 8 752 clips, 7.7 h, 52 speakers; the per-speaker share and its short utterances keep it far below its 70 h. |
| 11 | [NOTSOFAR-1](https://github.com/microsoft/NOTSOFAR1-Challenge) | en | **CC-BY-4.0** (README, 2026-11-08) | About 31 h of real meetings with human transcripts (and 1 000 h simulated), released in 2024, after the base model's data. Prepared as `notsofar-en-train` (2026-11-08) from the recorded train meetings only: 2 330 clips, 2.1 h, 12 speakers, so prone to overfitting in a run alone. |
| 12 | English **PeerTube**, CC-BY or public domain | en | per video, declared by the uploader | At least 2 187 h in Science & Tech (2026-11-08), noisy (crypto and distribution channels), so a whitelist as for French. |
| 8 | [LibriTTS-R](https://www.openslr.org/141/), [Hi-Fi TTS](https://www.openslr.org/109/) | en | **CC-BY-4.0** (2026-11-08) | Cased and punctuated read English: alternatives to LibriSpeech-PC (LibriTTS-R is the same LibriVox audio; Hi-Fi TTS has only ten speakers). |

### Evaluation sets considered

These sets are for evaluation only, never for training. Acquiring one adds a card
to [datasets.md](datasets.md) for it, and it joins the frozen evaluation sets of
a campaign ([decision 0002](../decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)).

| Source | Language | License | Reason / note |
| --- | --- | --- | --- |
| [MediaSpeech](https://www.openslr.org/108) FR (SLR108) | fr | CC-BY-4.0; the video copyright stays with its owners (2026-11-05) | Radio and television extracts transcribed by hand: a media register, better suited to evaluation than to training. |

The ServiceNow, VoxPopuli `en_accented` and CS-FLEURS rows that stood here were
acquired on 2026-11-08 and moved to
[Evaluation only](#evaluation-only--used-here-to-evaluate).

### Synthetic audio and text to speech

A replacement path for the abandoned Voxtral plan, for anything that could be
published: the model, the voices and the texts must all allow it, so a
publishable model may only use voices whose licence permits it (CC0 or CC-BY).
A voice cloned from a person is personal audio and stays out of the repository
([D-013](../owner/directives.md)).

| Source | Language | License | Reason / note |
| --- | --- | --- | --- |
| [Qwen3-TTS-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) base | fr, en | **Apache-2.0** (2026-11-05) | Native French and English with no preset French voice, so a three-second clone or voice design. The best candidate for English words inside French sentences. |
| [Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox) | fr, en | **MIT** (2026-11-05) | Clones a French reference; its output carries an inaudible watermark. |
| [CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | fr, en | **Apache-2.0** (2026-11-05) | French by cross-lingual cloning; the earlier CosyVoice2 did not really cover French. |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | fr, en | **Apache-2.0** (2026-11-05) | Thirty languages, French included. |
| [Kyutai TTS 1.6B en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr) | fr, en | **CC-BY-4.0**; voices under their own licences | Native English and French. Voice licences must be checked one by one: some voice sets are CC-BY-NC. |
| [Kyutai Pocket TTS](https://huggingface.co/kyutai) (upstream, with a French model) | fr, en | weights CC-BY-4.0, code MIT; gated; cloning only with the speaker's consent (2026-11-08) | Small. Usable voices for both Kyutai models: the CC0 voice donations, CML-TTS French and VCTK (CC-BY); not the Expresso or EARS voices (NC). |
| Models served on this fleet: OmniVoice 0.6B, Voxtral TTS 4B, the owner's `pocket-tts-fr` | fr, en | OmniVoice CC-BY-NC (from Emilia), Voxtral CC-BY-NC-4.0; `pocket-tts-fr` was trained on Voxtral output and is private (2026-11-08) | **Not for training data**: none of them allows it. A permissive TTS would have to be deployed first. |
| Piper `fr_FR` | fr | per voice: some CC-BY-4.0, one CC0, one CC-BY-SA, one AGPL (2026-11-05) | French-only volume, runs on CPU, and handles English badly. |
| Kokoro-82M | fr | Apache-2.0 (2026-11-05) | A single mediocre French voice: filler only. |
| Texts for synthetic sentences | fr, en | mixed: package-name glossaries, [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) (MIT), Wiktionary and French Wikipedia (CC-BY-SA-4.0), MDN FR (CC-BY-SA-2.5), Python documentation FR (PSF-2.0), linuxfr.org (CC-BY-SA-4.0) | Package and project names are not protected, and generated sentences record model, revision and prompt. Whether training on BY-SA text imposes SA on the weights is not settled; republishing the sentence set would. |

No FR/EN code-switched set with a permissive licence existed when checked
(2026-11-05), so such data has to be generated.

What the literature says (checked 2026-11-08): synthetic speech alone gives
about twice the error rate of real speech, but mixed with real speech it helps
(for example −23 % relative from 100 h synthetic with 100 h real, best at two to
three times as much synthetic, arXiv 2601.00935); on a strong model,
LLM-written code-switched sentences read by a TTS made WER 74 % worse with plain
LoRA and 5 % better with a Bayesian LoRA (arXiv 2606.21990). Two cautions for
this project: a TTS mispronounces English terms, so audio and label can
disagree; and the servicenow and CS-FLEURS panels are themselves synthetic, so a
gain on them would overstate the gain on real dictation.

## Abandoned — tried, then withdrawn

Kept with the reason, as [D-018](../owner/directives.md) requires. "Earlier
private project (date): <finding>; not reproduced here" marks a result that
cannot be checked in this repository, because the data and the runs are private.

| Source | Language | License | Reason it was withdrawn |
| --- | --- | --- | --- |
| An unofficial Hugging Face mirror of Common Voice FR | fr | CC0 announced (2026-11-05) | Version inconsistency (16.1 files in a set announced as 17) with no attestable checksum: withdrawn by decision in the earlier project and never used here. Replaced by the official MDC release, a candidate above. |
| Synthetic speech from Voxtral-4B-TTS FR-IT | fr, it | **model and preset voices CC-BY-NC-4.0**; the rights on the output were not established (2026-11-05) | Personal and non-commercial scope only, so unusable for a model that could be published: withdrawn by decision. The candidate list of permissive text-to-speech models replaces it. |
| [CEFC-ORFEO](https://huggingface.co/datasets/datasets-CNRS/CEFC-ORFEO) | fr | **CC-BY-NC-SA** — 4.0 on the card, 3.0 in the individual notices; contradiction kept, not resolved (2026-10-15) | Tried as a mechanical pipeline test in the earlier project, then withdrawn: the version contradiction and the NC/SA clause are unresolved, and NC plus SA are a poor fit for a model meant to be published. License as recorded, recheck the version question before any use. |

## Rejected — never used, by decision

The reasons below are the ones the earlier project's register recorded on
2026-11-05; the rows say when a later check was added. Where a source was
rejected for a non-commercial licence, the rejection concerns training: [decision
0009](../decisions/accepted/0009-non-commercial-datasets-for-testing-only.md)
allows a non-commercial source to be used for evaluation and its scores
published, so such a source could come back as an evaluation candidate.

| Source | Language | License | Reason for the decision |
| --- | --- | --- | --- |
| Audio and subtitles of films and series (including subtitle archives and rips) | fr | owners' copyright | No lawful access, so the text-and-data-mining exception does not apply, and circumventing a technical measure is forbidden; the subtitles are protected in their own right. Decided in the earlier project, 2026-11-05. |
| Old films on purchased discs | fr | authors' copyright: 70 years after the death of the last of its authors | Almost no French sound film is in the public domain yet, so each film would need its authors checked before it could even be a candidate. Recent restorations, subtitles and disc extras stay protected, and 1930s–1950s theatrical diction, narrow-band mono sound and the absence of transcripts make it poor ASR material. |
| YouTube videos that are not CC-BY | fr | creators' copyright | The mining exception does not apply: the terms reserve the rights, and a machine-readable per-video training setting defaults to refusing. Pseudo-labelling with another ASR model changes nothing about the rights on the audio. |
| [Archime french_tv_media_dataset_2026](https://huggingface.co/datasets/Archime/french_tv_media_dataset_2026) | fr | card says CC-BY-4.0, but the audio is France Télévisions property, academic use only | Conditions incompatible with personal use or publication, and the labels are pseudo-labels. |
| [mTEDx](https://openslr.org/100) FR | fr | CC-BY-NC-ND-4.0 | Non-commercial **and** no derivatives. |
| Emilia (the part that is not YODAS) FR | fr | CC-BY-NC-4.0 | Non-commercial; the YODAS part of Emilia stays a candidate above. |
| [Audiocite.net](https://www.openslr.org/139/) (SLR139) | fr | per file: BY, BY-SA, BY-ND or BY-NC | Heterogeneous licences to sort file by file, and audiobooks like MLS, whose training result was negative. |
| ESLO, CFPP2000, TCOF and other ORTOLANG corpora | fr | mostly CC-BY-NC-SA | Non-commercial and share-alike, although they are excellent spontaneous speech; to reconsider only for a strictly unpublished use. |
| FLEURS FR train | fr | CC-BY-4.0 | A few hours of read speech adds little next to the test split used here. |
| MagicHub | fr | portal conditions | Excluded by decision in the earlier project; confirmed later: the free French set is small and NC-ND, and the two large French corpora are paid. |
| TalkBank / FrenchTalkBank | fr | CC-BY-NC-SA-3.0 | Non-commercial and share-alike. |
| LibriVox FR | fr | public domain | This is the source of MLS: read audiobook speech with a negative training result. |
| [UniDataPro french-speech-recognition](https://huggingface.co/datasets/UniDataPro/french-speech-recognition-dataset) | fr | CC-BY-NC-ND-4.0, excerpt only, full corpus paid | Non-commercial and no derivatives, although it was the only telephone corpus found. |
| Speech-data / French-Speech-Dataset | fr | CC-BY-NC-ND-4.0 | Non-commercial, no derivatives, and doubtful provenance. |
| [SwitchLingua](https://huggingface.co/datasets/Shelton1013/SwitchLingua_audio) | multi, FR-EN among them | CC-BY-NC-4.0 + signed agreement | Non-commercial code-switching; to reconsider only for a strictly unpublished use. |
| CS-FLEURS training set | fr, en | CC-BY-NC-4.0 | Non-commercial, so no training; its test part is used for evaluation only ([Evaluation only](#evaluation-only--used-here-to-evaluate)). |
| TED-LIUM 3, linux.conf.au | en | NC-ND, NC-SA | Non-commercial. AMI stood in this row as NC-SA until 2026-11-08; its licence is CC-BY-4.0, and it is now a candidate. |
| PyCon US | en | speaker agreement: "non-commercial, private" use | Non-commercial; contrary to a widespread belief, it is not CC-BY. |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) | en | metadata Apache-2.0, but the access conditions say the organisers do not own the audio, and allow non-commercial research only, behind a named form (2026-11-05) | Non-commercial, English only, and personal use is not research; Granary EN and FOSDEM cover the same content lawfully. |
| Earnings-21/22, ConEC | en | no licence declared ("free-to-use benchmark") | No licence means all rights reserved. |
| Silencio english-accents, westbrook English_Accent_DataSet, Speech Accent Archive | en | NC ("the weights inherit the restriction"), no licence with a research-only mix, and NC-SA | Non-commercial or unknown. |
| Parliament of Canada (Hansard and audio), Swiss Federal Assembly | en, fr | Crown copyright, permission required, "no rights transferred" | No licence to reuse the audio. |
| [EuroSpeech](https://huggingface.co/datasets/disco-eth/EuroSpeech) France, and the Assemblée nationale's videos | fr | the card says Licence Ouverte; the Assembly's [own page](https://www.assemblee-nationale.fr/dyn/info-site) says its multimedia resources need prior agreement and no commercial use, and Licence Ouverte covers its open data only; EuroSpeech's README says "TBD" | No licence for the audio. Checked 2026-11-08; it was candidate 8 until then. |
| The French Senate's videos | fr | [legal notice](https://www.senat.fr/mentions-legales.html): no modification, logo conditions, the audio needs the Senate's agreement | No licence for the audio. Checked 2026-11-08. |
| New European Parliament plenaries, Europarl-ASR | en, fr | the Parliament's own notice (the entire item must be reproduced), not an open licence | Not in the licence list, and the base model has VoxPopuli and Europarl-ASR from the same Parliament. Checked 2026-11-08. |
| Élysée videos | fr | its Licence Ouverte excludes multimedia | No licence for the audio. Checked 2026-11-08. |
| [speech-wikimedia](https://huggingface.co/datasets/MLCommons/speech-wikimedia) FR | fr | the compilation is CC-BY-SA-4.0 | Share-alike, and its 562 French subtitle files are mostly translations of videos in other languages. Checked 2026-11-08; it was candidate 11 until then. |
| Les Cast Codeurs (podcast) | fr | CC-BY-NC-ND-4.0 | Non-commercial and no derivatives, although it is French developer talk; its licence page invites requests for another licence, which only the owner can make. Checked 2026-11-08. |
| NASA *On a Mission* (podcast) | en | © NASA/JPL/Caltech, as its feed states | The Jet Propulsion Laboratory is run by Caltech, a contractor, so its works are not US government works and are not in the public domain ([D-019](../owner/directives.md)). The other NASA shows are a candidate (English row 9). Checked 2026-11-08. |
| VoiceStick (Zenodo 20354233) | fr | CC-BY-NC-4.0 | French voice commands: non-commercial, so a testing-only candidate under decision 0009. Checked 2026-11-08. |
| MediaParl, Cameroon French, CAFE, French_Transcribed_Podcast, french-education-speech, voxozi french-b2b-tts, Emilia-French mirrors | fr | restricted access, no licence, unknown audio origin, or a redistribution of gated Emilia | Nothing that can be checked or published. Checked 2026-11-08. |
| L2-ARCTIC, EdAcc, Kernel Recipes, VoxCeleb, People's Speech | en | CC-BY-NC, CC-BY-SA, none found, owners' copyright, BY and BY-SA mixed per file | Non-commercial, share-alike or no licence. Checked 2026-11-08. |
| VoxPopuli EN train, as English with a French accent | en | CC0 | Seven French MEPs, 33 clips, and it holds 26 of the 27 speakers of the `voxpopuli-en-nonnative` evaluation set: no usable accented English source, and a leak risk for any English use. Checked 2026-11-08. |
| Vibravox | fr | CC-BY-4.0 | Bone-conduction sensors rather than an ordinary microphone: not the channel this project records through. |
| IndexTTS-2, Fish-Speech/OpenAudio, F5-TTS, XTTS-v2, Higgs Audio, MaskGCT, MMS-TTS, Spark-TTS, VibeVoice | multi | a licence forbidding improving another model except non-commercially, NC-SA, NC, a custom non-commercial licence, a research-only licence, and one with an audible warning in the output | Their output cannot be used for a model that may be published; the permissive text-to-speech candidates above replace them. |
