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
    "african-accented-fr-train": DatasetInfo(
        name="african-accented-fr-train",
        source="https://www.openslr.org/57 (African_Accented_French.tar.gz, speech/train)",
        license="Apache-2.0",
        split="train",
        reference_style="lowercase, no punctuation",
        grouping="speaker (group = directory of the recordings)",
        notes="Training data for the per-source screening: every train recording that "
        "has a transcript and passes the filters documented in docs/stt/datasets.md, "
        "which is fewer clips than the cap. The corpus has no transcript for part "
        "of its train recordings. No panels.",
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
    "voxpopuli-fr-train": DatasetInfo(
        name="voxpopuli-fr-train",
        source="https://huggingface.co/datasets/facebook/voxpopuli (config fr, split train)",
        license="CC0-1.0 (data only)",
        split="train",
        reference_style="raw_text: casing and punctuation kept",
        grouping="speaker (group = VoxPopuli speaker id)",
        notes="Training data for the per-source screening: a capped number of clips "
        "of the train split, selected and filtered as documented in "
        "docs/stt/datasets.md. Speakers of the French-accented English "
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
    "voxpopuli-en-french-accent": DatasetInfo(
        name="voxpopuli-en-french-accent",
        source="https://huggingface.co/datasets/facebook/voxpopuli (config en_accented)",
        license="CC0-1.0 (data; the repository's code and models are CC-BY-NC and unused here)",
        split="test (accent en_fr only)",
        reference_style="editor's transcript, no punctuation; every clip is a gold transcript",
        grouping="speaker (group = VoxPopuli speaker id)",
        notes="English read in the European Parliament by French speakers, in the accented "
        "English subset of VoxPopuli. The whole config is downloaded and the 737 clips of "
        "the French accent are kept.",
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
