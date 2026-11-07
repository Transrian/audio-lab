"""What a dataset prepared by ``audiolab data fetch`` contains.

Every dataset is described here once, including its license: the cards in
`docs/stt/datasets.md` and the ``audiolab data list`` output are both built
from this registry, so a license is recorded in exactly one place.
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
}


def dataset_info(name: str) -> DatasetInfo:
    """The registry entry of a dataset, or a clear error listing the known names."""
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown dataset {name!r}; known datasets: {known}") from None
