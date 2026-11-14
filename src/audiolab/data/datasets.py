"""What a dataset prepared by ``audiolab data fetch`` contains.

Every dataset is described here once, including its license, and ``audiolab
data list`` prints this registry: a license is recorded in exactly one place.
The prose cards in `docs/stt/datasets.md` are written by hand and link here.

A license cites its decision as it stood when the dataset was prepared, so a
prepared manifest keeps the wording of its own preparation and a
re-preparation after the decision moved (proposed to accepted, for instance)
writes the new wording; the grant itself is unchanged either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A larger preparation of a training source keeps the source's name and appends
# its cap: `granary-fr-train-40k` is the 40 000-clip preparation of
# `granary-fr-train`. The registry and the fetchers know the source, not the
# size, so every lookup of a prepared name goes through :func:`base_dataset`.
_SIZE_SUFFIX = re.compile(r"-\d+k$")


def base_dataset(name: str) -> str:
    """The registry name behind a prepared name: strip a trailing ``-<k>k`` cap.

    `granary-fr-train-40k` is the 40 000-clip preparation of `granary-fr-train`,
    and both are prepared from the same source under the same cache. A name
    without the suffix is its own base.
    """
    return _SIZE_SUFFIX.sub("", name)


@dataclass(frozen=True)
class DatasetInfo:
    """Everything a reader needs to know before using a dataset."""

    name: str
    source: str
    license: str
    split: str
    reference_style: str
    grouping: str
    notes: str
    #: The cache directory's name, when the fetcher stores under another
    #: dataset's cache (the test part shares its archive with the train part).
    cache_name: str | None = None


DATASETS: dict[str, DatasetInfo] = {
    "fleurs-fr": DatasetInfo(
        name="fleurs-fr",
        source="https://huggingface.co/datasets/google/fleurs (config fr_fr)",
        license="CC-BY-4.0",
        split="test",
        reference_style="raw transcription: casing and punctuation kept",
        grouping="sentence (group = FLEURS sentence id)",
        notes="Read Wikipedia sentences by different speakers. A sentence id can be "
        "shared by several speakers, hence sentence-level grouping.",
    ),
    "fleurs-en": DatasetInfo(
        name="fleurs-en",
        source="https://huggingface.co/datasets/google/fleurs (config en_us)",
        license="CC-BY-4.0",
        split="test",
        reference_style="raw transcription: casing and punctuation kept",
        grouping="sentence (group = FLEURS sentence id)",
        notes="Same corpus as fleurs-fr, English side.",
    ),
    "mls-fr": DatasetInfo(
        name="mls-fr",
        source="https://huggingface.co/datasets/facebook/multilingual_librispeech "
        "(config french, data/mls_french)",
        license="CC-BY-4.0",
        split="test",
        reference_style="lowercase, no punctuation",
        grouping="speaker (group = MLS speaker id)",
        notes="Audiobooks. Recorded for possible future training overlap: the train "
        "split is in the same corpus and shares speakers with no other split used here.",
    ),
    "librispeech-en": DatasetInfo(
        name="librispeech-en",
        source="https://www.openslr.org/12 (test-clean.tar.gz)",
        license="CC-BY-4.0",
        split="test-clean",
        reference_style="uppercase, no punctuation",
        grouping="speaker (group = LibriSpeech speaker id)",
        notes="Audiobooks. Only test-clean is used; test-other is left for later.",
    ),
    "notsofar-en-eval": DatasetInfo(
        name="notsofar-en-eval",
        source="https://huggingface.co/datasets/microsoft/NOTSOFAR "
        "(benchmark-datasets/eval_set/240825.1_eval_full_with_GT)",
        license="CC-BY-4.0",
        split="240825.1_eval_full_with_GT",
        reference_style="the annotators' words, cased and punctuated, without their markup",
        grouping="speaker (group = the participant alias the corpus gives the speaker)",
        notes="The challenge's evaluation meetings: real recorded meetings with one "
        "close microphone per participant and human word-by-word transcripts. The "
        "240629.1_eval_small subset of the same split holds no transcripts and is "
        "not read. An alias the prepared notsofar-en-train holds is dropped from "
        "this set, and the train fetcher leaves the set's aliases out (decision "
        "0002).",
    ),
    "icsi-en-train": DatasetInfo(
        name="icsi-en-train",
        source="https://groups.inf.ed.ac.uk/ami/icsi/ (ICSI_original_transcripts.zip and "
        "ICSIsignals/SPH/<meeting>/<channel>.sph)",
        license="CC-BY-4.0",
        split="mrt",
        reference_style="the annotators' words, cased and punctuated, as they wrote "
        "them: a false start and a partial word are kept, a laugh or a noise is not",
        grouping="speaker (group = the ICSI participant id)",
        notes="Research meetings with one close microphone per participant and word-by-word "
        "manual transcripts, in the MRT release, which carries the start and end time of "
        "every utterance. Audio is shorten-compressed SPHERE and is decoded with ffmpeg. "
        "Training data for the per-source screening, as documented in docs/stt/datasets.md. "
        "No panels.",
    ),
    "notsofar-en-train": DatasetInfo(
        name="notsofar-en-train",
        source="https://huggingface.co/datasets/microsoft/NOTSOFAR "
        "(benchmark-datasets/train_set/240825.1_train)",
        license="CC-BY-4.0",
        split="240825.1_train",
        reference_style="the annotators' words, cased and punctuated, without their markup",
        grouping="speaker (group = the participant alias)",
        notes="Real recorded meetings with close-talking per-participant recordings and human "
        "transcripts. Only the recorded train subset is read; the simulated set and the dev "
        "and evaluation subsets are not. Training data for the per-source screening, as "
        "documented in docs/stt/datasets.md. No panels.",
    ),
    "librispeech-en-train": DatasetInfo(
        name="librispeech-en-train",
        source="https://www.openslr.org/12 (train-clean-100.tar.gz)",
        license="CC-BY-4.0",
        split="train-clean-100",
        reference_style="uppercase, unpunctuated",
        grouping="speaker",
        notes="Training data for the per-source screening: the first clips of "
        "train-clean-100 in the archive's order, with the speakers of "
        "librispeech-en left out (decision 0002), as documented in "
        "docs/stt/datasets.md. No panels.",
    ),
    "librispeech-pc-en-train": DatasetInfo(
        name="librispeech-pc-en-train",
        source="https://www.openslr.org/145 (manifests.tar.gz, train-clean-100) over the "
        "audio of https://www.openslr.org/12 (train-clean-100.tar.gz)",
        license="CC-BY-4.0",
        split="train-clean-100",
        reference_style="restored text: casing and punctuation",
        grouping="speaker (group = LibriSpeech speaker id)",
        notes="Training data for the per-source screening: the same clips as "
        "librispeech-en-train with the cased and punctuated text of LibriSpeech-PC, "
        "as documented in docs/stt/datasets.md. The base model very likely trained on "
        "this audio with this text, so it guards English rather than bringing gains. "
        "No panels.",
    ),
    "ami-fr-speakers-en-train": DatasetInfo(
        name="ami-fr-speakers-en-train",
        source="https://groups.inf.ed.ac.uk/ami/download/ (ami_public_manual_1.6.2.zip "
        "annotations and the corpus's own AMICorpusMirror audio)",
        license="CC-BY-4.0",
        split="manual",
        reference_style="manual words: casing and punctuation as the annotators wrote them",
        grouping="speaker (group = the AMI participant id)",
        notes="Training data for the per-source screening: English meetings spoken by "
        "the participants whose native language is French, one clip per sentence of "
        "their own headset recording, as documented in docs/stt/datasets.md. AMI is in "
        "the training data of English models, Parakeet among them, so this is a guard "
        "for a mix and not a source of new material. No panels.",
    ),
    "african-accented-fr-train": DatasetInfo(
        name="african-accented-fr-train",
        source="https://www.openslr.org/57 (African_Accented_French.tar.gz, speech/train)",
        license="Apache-2.0",
        split="train",
        reference_style="lowercase, no punctuation; the corpus's apostrophe spacing removed",
        grouping="speaker (group = directory of the recordings)",
        notes="Training data for the per-source screening: every train recording that "
        "has a transcript and passes the filters documented in docs/stt/datasets.md, "
        "which is fewer clips than the cap. The corpus has no transcript for part "
        "of its train recordings, and eight prompting lines are left out: they offer "
        "the speaker two answers separated by a slash, so no one text matches the "
        "recording. No panels.",
    ),
    "african-accented-fr-test": DatasetInfo(
        name="african-accented-fr-test",
        source="https://www.openslr.org/57 (African_Accented_French.tar.gz, speech/test)",
        license="Apache-2.0",
        split="test",
        reference_style="the corpus's own prompted sentences, casing and punctuation kept; "
        "the corpus's apostrophe spacing removed",
        grouping="speaker (group = directory of the recordings)",
        notes="The test part of the corpus african-accented-fr-train trains on: 515 "
        "prompted sentences of 12 speakers, a single transcript per part. Its speakers "
        "do not overlap the train part's (checked against the prepared train manifest "
        "at preparation, a speaker in both is dropped, more than 10 % overlapping "
        "stops). dev and devtest stay untouched as a reserve for a later campaign. "
        "The train fetcher leaves its groups out (decision 0002).",
        cache_name="african-accented-fr-train",
    ),
    "pxcorpus-fr-train": DatasetInfo(
        name="pxcorpus-fr-train",
        source="https://zenodo.org/records/10080490 (PxCorpus.zip, PxSLU recordings)",
        license="CC-BY-4.0",
        split="recordings",
        reference_style="lowercase, no punctuation",
        grouping="speaker (group = the participant id of the recording)",
        notes="Training data for the per-source screening: every recording of the "
        "corpus, which is far fewer clips than the cap, so its arm of the screening "
        "has fewer clips than the others, as documented in docs/stt/datasets.md. "
        "Spoken drug prescriptions, one utterance per recording. No panels.",
    ),
    "granary-fr-train": DatasetInfo(
        name="granary-fr-train",
        source="https://huggingface.co/datasets/espnet/yodas-granary (data/fr000, asr_only)",
        license="audio CC-BY-3.0, labels CC-BY-4.0",
        split="asr_only",
        reference_style="pseudo-labelled text: casing and punctuation kept",
        grouping="recording (group = YouTube video id)",
        notes="Training data for the per-source screening: a capped number of rows "
        "of the French directory, selected and filtered as documented in "
        "docs/stt/datasets.md. Labels are model output, not human transcripts. "
        "No panels.",
    ),
    "mls-fr-train": DatasetInfo(
        name="mls-fr-train",
        source="https://huggingface.co/datasets/facebook/multilingual_librispeech "
        "(config french, data/mls_french/train)",
        license="CC-BY-4.0",
        split="train",
        reference_style="lowercase, no punctuation",
        grouping="speaker (group = MLS speaker id)",
        notes="Training data for the per-source screening: a capped number of clips "
        "of the train split, selected and filtered as documented in "
        "docs/stt/datasets.md. MLS keeps the speakers of its splits apart, and "
        "the speakers of the test split are excluded anyway. No panels.",
    ),
    "peertube-edu-fr-train": DatasetInfo(
        name="peertube-edu-fr-train",
        source="the French education ministry's PeerTube instances on apps.education.fr "
        "(PeerTube API; the pinned list of videos is "
        "src/audiolab/data/peertube_edu_listing.json)",
        license="CC-BY / CC0-1.0",
        split="listing",
        reference_style="model labels (Qwen3-ASR checked against Whisper), casing and punctuation",
        grouping="recording (group = PeerTube video uuid)",
        notes="Training data for the per-source screening: the ministry's own French "
        "videos, ten minutes or longer, from channels that are not music or stage "
        "performances, labelled as described in docs/stt/datasets.md. Labels are "
        "model output, not human transcripts. Every clip carries the attribution of "
        "its video (channel, title, URL, license). No panels.",
    ),
    "peertube-edu-captions-fr": DatasetInfo(
        name="peertube-edu-captions-fr",
        source="the French education ministry's PeerTube instances on apps.education.fr "
        "(the human French caption tracks of the same listing as peertube-edu-fr-train)",
        license="CC-BY / CC0-1.0",
        split="listing (human captions)",
        reference_style="the captions as written, cased and punctuated, cut at the "
        "cues' time codes",
        grouping="recording (group = PeerTube video uuid), at most 40 clips per talk",
        notes="Conversational classroom French, labelled from human captions: no "
        "labelling model is in the loop. The clips are kept only when Whisper "
        "transcribes them within WER 0.35 of the caption, which is what removes "
        "out-of-sync cues and biases the set toward clips Whisper gets right; the "
        "talks of peertube-edu-fr-train are left out (decision 0002), and that "
        "fetcher leaves this set's talks out of every preparation it makes after "
        "the set exists. The category is tech or general by the talk's title. "
        "Panels: as described in docs/stt/datasets.md.",
    ),
    "debconf-en-train": DatasetInfo(
        name="debconf-en-train",
        source="https://peertube.debian.social (PeerTube API; the pinned list of videos "
        "is src/audiolab/data/debconf_listing.json)",
        license="CC-BY",
        split="listing",
        reference_style="model labels (Qwen3-ASR checked against Whisper), casing and punctuation",
        grouping="recording (group = PeerTube video uuid)",
        notes="Training data for the per-source screening: the DebConf and mini-DebConf "
        "talks of the Debian project, in English, labelled in English, as described in "
        "docs/stt/datasets.md. Labels are model output, not human transcripts. Every "
        "clip carries the attribution of its video (channel, title, URL, license). "
        "No panels.",
    ),
    "fosdem-en-train": DatasetInfo(
        name="fosdem-en-train",
        source="https://video.fosdem.org (the video tree, joined to the per-year schedule "
        "XML; the pinned list of talks is src/audiolab/data/fosdem_listing.json)",
        license="CC-BY-2.0-BE",
        split="listing",
        reference_style="model labels (Qwen3-ASR checked against Whisper), casing and punctuation",
        grouping="recording (group = talk slug)",
        notes="Training data for the per-source screening: FOSDEM talks of 2020 to 2026, "
        "at least ten minutes long, labelled in English as described in "
        "docs/stt/datasets.md. Labels are model output, not human transcripts. Every "
        "clip carries the attribution CC BY asks for (speakers, title, URL, license). "
        "No panels.",
    ),
    "nasa-en-train": DatasetInfo(
        name="nasa-en-train",
        source="https://www.nasa.gov/podcasts/ (five shows' RSS feeds and the episode pages; "
        "the pinned list of episodes is src/audiolab/data/nasa_listing.json)",
        license="public domain (US government work)",
        split="listing",
        reference_style="NASA's own transcript of each episode, cased and punctuated, "
        "aligned to the audio by Whisper large-v3; speaker labels and notes removed",
        grouping="recording (group = episode page slug)",
        notes="Training data for the per-source screening: conversational technical "
        "English from NASA's podcasts, labelled from human transcripts as described in "
        "docs/stt/datasets.md. Shows and episodes that name another copyright holder "
        "are left out. Every clip carries its attribution (NASA and the show, episode "
        "title, page URL). No panels.",
    ),
    "nasa-en-train-qwen": DatasetInfo(
        name="nasa-en-train-qwen",
        source="the same clips and audio as nasa-en-train, relabelled (no new source)",
        license="public domain (US government work)",
        split="listing",
        reference_style="model labels (Qwen3-ASR alone, no agreement filter), "
        "casing and punctuation",
        grouping="recording (group = episode page slug), as nasa-en-train",
        notes="The label-style test set: the same clips, audio and clip order as "
        "nasa-en-train, with the text of every clip replaced by the Qwen3-ASR answer, "
        "so a run on it differs from a run on nasa-en-train in the labels alone. "
        "Prepared from the source's manifest by src/audiolab/data/relabel.py; "
        "described in docs/stt/datasets.md. No panels.",
    ),
    "nasa-en-test": DatasetInfo(
        name="nasa-en-test",
        source="https://www.nasa.gov/podcasts/ (the same listing as nasa-en-train, "
        "60 episodes reserved by a constant rule, at most 15 per show)",
        license="public domain (US government work)",
        split="listing (reserved episodes)",
        reference_style="NASA's own transcript of each episode, cased and punctuated, "
        "aligned to the audio by Whisper large-v3; speaker labels and notes removed",
        grouping="recording (group = episode page slug), at most 30 clips per episode",
        notes="Conversational technical English, labelled from human transcripts. "
        "The edge check is Whisper's, not the base model's, which is the model the "
        "set is scored on; the set is thereby biased toward clips Whisper hears well "
        "at the edges. Its episodes are left out of every nasa-en-train preparation "
        "(decision 0002). Curious Universe is never reserved.",
    ),
    "peertube-fr-train": DatasetInfo(
        name="peertube-fr-train",
        source="https://videos.capitoledulibre.org (PeerTube API; the pinned list of "
        "videos is src/audiolab/data/peertube_listing.json)",
        license="CC-BY",
        split="listing",
        reference_style="model labels (Qwen3-ASR checked against Whisper), casing and punctuation",
        grouping="recording (group = PeerTube video uuid)",
        notes="Training data for the per-source screening: free-software talks in French "
        "with English technical terms, labelled as described in docs/stt/datasets.md. "
        "Labels are model output, not human transcripts. Every clip carries the "
        "attribution of its video (author, title, URL, license). No panels.",
    ),
    "peertube-tech-fr-train": DatasetInfo(
        name="peertube-tech-fr-train",
        source="https://videos.capitoledulibre.org and the French education ministry's "
        "PeerTube instances (the tech titles of the pinned listings of peertube-fr-train "
        "and peertube-edu-fr-train; the pinned list is "
        "src/audiolab/data/peertube_tech_listing.json)",
        license="CC-BY / CC0-1.0",
        split="listing",
        reference_style="model labels (Qwen3-ASR checked against Whisper), casing and punctuation",
        grouping="recording (group = PeerTube video uuid)",
        notes="Training data at scale for the base model: the talks of the two French "
        "PeerTube listings whose titles carry a word of the technology the model trains "
        "(peertube_tech.TECH_TITLE), labelled as described in docs/stt/datasets.md. "
        "Labels are model output, not human transcripts. Every clip carries the "
        "attribution of its video (channel, title, URL, license). The talks of "
        "peertube-edu-captions-fr are left out (decision 0002). No panels.",
    ),
    "voxpopuli-fr-train": DatasetInfo(
        name="voxpopuli-fr-train",
        source="https://huggingface.co/datasets/facebook/voxpopuli (config fr, split train)",
        license="CC0-1.0 (data only)",
        split="train",
        reference_style="raw_text: casing and punctuation kept",
        grouping="speaker (group = VoxPopuli speaker id)",
        notes="Training data for the per-source screening: a capped number of clips "
        "of the train split, selected and filtered as documented in "
        "docs/stt/datasets.md. Speakers of the non-native English "
        "evaluation subset are left out. No panels.",
    ),
    "nonspeech": DatasetInfo(
        name="nonspeech",
        source="generated here (silence, noise) + https://github.com/karolpiczak/ESC-50",
        license="generated clips: none; ESC-50: CC-BY-NC-3.0 (test use only, decision 0009)",
        split="generated + esc50",
        reference_style="no speech: the reference text is empty",
        grouping="clip (group = clip id)",
        notes="Hallucination battery: 50 silences, 50 low-level noises, 200 ESC-50 "
        "clips. The music of the battery lives in `nonspeech-music`, with its "
        "own panels and its own family numbers.",
    ),
    "noisy-fleurs-fr": DatasetInfo(
        name="noisy-fleurs-fr",
        source="derived: the fleurs-fr panel clips mixed with musan-noise (eval half), "
        "rirs-noises (real rooms) and mls-fr-train babble",
        license="per clip: the speech licence plus the material's, e.g. "
        "CC-BY-4.0 (speech) + public domain (noise)",
        split="the source's panels, in one acoustic condition per clip",
        reference_style="the source clip's text, unchanged",
        grouping="the source's group, re-prefixed (group = the FLEURS sentence)",
        notes="A derived evaluation set: same texts, same groups, same sealing as "
        "fleurs-fr, in noise, babble, reverberation and their mix. Every choice is "
        "a hash of the clip id, so a re-preparation is byte-identical.",
    ),
    "noisy-fleurs-en": DatasetInfo(
        name="noisy-fleurs-en",
        source="derived: the fleurs-en panel clips mixed with musan-noise (eval half), "
        "rirs-noises (real rooms) and librispeech-en-train babble",
        license="per clip: the speech licence plus the material's, e.g. "
        "CC-BY-4.0 (speech) + public domain (noise)",
        split="the source's panels, in one acoustic condition per clip",
        reference_style="the source clip's text, unchanged",
        grouping="the source's group, re-prefixed (group = the FLEURS sentence)",
        notes="The English side of noisy-fleurs-fr.",
    ),
    "noisy-servicenow-codeswitched-en-fr": DatasetInfo(
        name="noisy-servicenow-codeswitched-en-fr",
        source="derived: the servicenow-codeswitched-en-fr panel clips mixed with "
        "musan-noise (eval half), rirs-noises (real rooms) and mls-fr-train and "
        "librispeech-en-train babble",
        license="no licence on the card (decision 0011): evaluation only, "
        "never redistributed; per clip the material's licence is added",
        split="the source's panels, in one acoustic condition per clip",
        reference_style="the source clip's transcript, unchanged",
        grouping="the source's group, re-prefixed (group = the TTS voice)",
        notes="Owner-approved (Q4, 2026-11-10). The babble alternates French and "
        "English speakers. As for the source: synthetic speech with no licence, "
        "used for evaluation only and never redistributed (decision 0011).",
    ),
    "nonspeech-music": DatasetInfo(
        name="nonspeech-music",
        source="https://www.openslr.org/17 (musan.tar.gz, musan/music/)",
        license="per file, from the sub-collection's LICENSE: public domain or CC-BY "
        "(any version); files with vocals or other licences are not read",
        split="10 s excerpts, one per file, up to 300 files spread over the sub-collections",
        reference_style="no speech: the reference text is empty",
        grouping="artist where the annotation gives one, else the file",
        notes="Music for the hallucination battery (backlog N-03): a model should "
        "transcribe it as empty. Only instrumental, redistributable files are "
        "kept. The speech part of the archive is not read.",
    ),
    "musan-noise": DatasetInfo(
        name="musan-noise",
        source="https://www.openslr.org/17 (musan.tar.gz, musan/noise/)",
        license="CC-BY-4.0 (corpus); free-sound files public domain, sound-bible files "
        "per block CC BY 3.0 or public domain",
        split="eval + train by the musan-split hash rule",
        reference_style="no speech: the reference text is empty",
        grouping="recording (group = the noise file)",
        notes="Noise for the noisy evaluation sets (T4) and for training "
        "augmentation (plan 02). One hash rule decides forever which half of the "
        "files is for evaluation and which for training; the point-source noises "
        "of RIRS_NOISES are copies of these files and are never read. No panels.",
    ),
    "rirs-noises": DatasetInfo(
        name="rirs-noises",
        source="https://www.openslr.org/28 (rirs_noises.zip)",
        license="Apache-2.0",
        split="eval = real rooms, train = simulated rooms",
        reference_style="no speech: the reference text is empty",
        grouping="recording (group = the RIR file)",
        notes="Room impulse responses: the real-room RIRs are the evaluation "
        "half, the simulated ones the training half, so a model augmented in "
        "simulated rooms is scored in real ones. The point-source noises are "
        "MUSAN copies and are never read. No panels.",
    ),
    "voxpopuli-en-nonnative": DatasetInfo(
        name="voxpopuli-en-nonnative",
        source="https://huggingface.co/datasets/facebook/voxpopuli (config en_accented)",
        license="CC0-1.0 (data; the repository's code and models are CC-BY-NC and unused here)",
        split="test (accent en_fr only)",
        reference_style="editor's transcript, no punctuation; every clip is a gold transcript",
        grouping="speaker (group = VoxPopuli speaker id)",
        notes="English read in the European Parliament by non-native speakers, in the accented "
        "English subset of VoxPopuli. The whole config is downloaded and the 737 clips the "
        "corpus labels French (accent en_fr) are kept; matching those speakers to the "
        "Parliament's member list found only about 26 of them French-speaking, so the "
        "dataset is named after what it holds, not after the label.",
    ),
    "servicenow-codeswitched-en-fr": DatasetInfo(
        name="servicenow-codeswitched-en-fr",
        source="https://huggingface.co/datasets/ServiceNow-AI/asr_codeswitched (config en_fr)",
        license="no license on the card: evaluation only, never trained on (decision 0011)",
        split="test",
        reference_style="corpus transcript of the code-switched sentence: casing "
        "and punctuation kept",
        grouping="voice (group = the TTS voice that reads the clip)",
        notes="Synthetic IT-support speech switching between French and English inside a "
        "sentence. French frames every utterance; the source labels each word with its "
        "language. Never redistributed.",
    ),
    "servicenow-codeswitched-en-fr-ca": DatasetInfo(
        name="servicenow-codeswitched-en-fr-ca",
        source="https://huggingface.co/datasets/ServiceNow-AI/asr_codeswitched (config en_fr_ca)",
        license="no license on the card: evaluation only, never trained on (decision 0011)",
        split="test",
        reference_style="corpus transcript of the code-switched sentence: casing "
        "and punctuation kept",
        grouping="voice (group = the TTS voice that reads the clip)",
        notes="Québec French side of the same corpus: same sentences as en_fr, told in "
        "Québec French. Never redistributed.",
    ),
    "cs-fleurs-fr-en": DatasetInfo(
        name="cs-fleurs-fr-en",
        source="https://huggingface.co/datasets/byan/cs-fleurs "
        "(read/test and xtts/test1, French-English pairs)",
        license="CC-BY-NC-4.0 (testing only, decision 0009)",
        split="read/test + xtts/test1",
        reference_style="code-switched sentence: casing and punctuation kept",
        grouping="sentence (group = the FLEURS sentence it was built from)",
        notes="FLEURS sentences with part of their words replaced by the other language. "
        "The read subset is real audio, the xtts one is synthetic; category tells them "
        "apart.",
    ),
}
