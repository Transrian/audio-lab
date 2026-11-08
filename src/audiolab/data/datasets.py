"""What a dataset prepared by ``audiolab data fetch`` contains.

Every dataset is described here once, including its license, and ``audiolab
data list`` prints this registry: a license is recorded in exactly one place.
The prose cards in `docs/stt/datasets.md` are written by hand and link here.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        "clips. No music (known gap).",
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
        license="no license on the card: evaluation only, never trained on "
        "(proposed decision 0011)",
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
        license="no license on the card: evaluation only, never trained on "
        "(proposed decision 0011)",
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
