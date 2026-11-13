"""ServiceNow code-switched French and English (``servicenow-codeswitched-*``).

Short IT-support utterances — a password reset, a ticket that does not move, a
voice agent reading back an address — in which the speaker switches language
inside the sentence (``Je viens de rebooter mon laptop après un update``). That
is the case the owner's target use is about (D-009): a French developer
dictating French with English technical words in it, and the vocabulary is the
right one (ticket, update, VPN, Okta, Outlook, laptop).

Two configurations are prepared as two datasets, so the Québec French of
``en_fr_ca`` has its own numbers: ``fr_ca`` says ``cell`` where ``fr`` says
``portable``. The corpus's own card describes it as built by verbalizing text
and synthesizing the audio with a TTS voice, one voice per speaker column, and
its own framing is French with English embedded; every clip is therefore
announced and normalized as French here. The source labels each word with its
language (``words`` and ``word_languages``); the reference text of an item is
the corpus's transcript of the sentence, with casing and punctuation.

The card carries no license. The owner approved its use for evaluation on
2026-11-08 (decision 0011): it is never trained on and never
redistributed.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import download
from audiolab.data.parquet import audio_of, rows

REPO = "ServiceNow-AI/asr_codeswitched"
# Pinned so the files a panel was built from cannot change under it.
REVISION = "82bf4b49499bcaa6f4cf6a67bebfab3ee2affcbb"
REPO_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
SPLIT = "test"
# One configuration per dataset: the French of France and the French of Québec.
CONFIGS = {
    "servicenow-codeswitched-en-fr": "en_fr",
    "servicenow-codeswitched-en-fr-ca": "en_fr_ca",
}
# French is the framing language of every utterance, in both configurations.
LANGUAGE = "fr"
LICENSE = "no license on the card: evaluation only, never trained on (decision 0011)"
FILE_NAME = "test-00000-of-00001.parquet"
COLUMNS = ["ID", "audio", "utterance", "voice"]
# sha256 of each file at that revision, as the Hub publishes it; recorded from
# the download that built the panels.
PINNED_SHA256 = {
    "en_fr": "bc05dc0404d8209e05ea7041c8557e773bf9c59127a52c8784c8c760ec11284d",
    "en_fr_ca": "1ba4b634d11621834bd4184d6455e6e4a7a9a33d8a9bad2c7853e58837966ad6",
}
# The published size of each configuration; tests patch this.
EXPECTED_CLIPS = {"en_fr": 298, "en_fr_ca": 188}


def fetch_servicenow_codeswitched(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool
) -> int:
    """Prepare one configuration, returning the number of clips written."""
    config = CONFIGS[name]
    path = download(
        f"{REPO_URL}/{config}/{FILE_NAME}",
        cache_dir / name / FILE_NAME,
        sha256=PINNED_SHA256.get(config, ""),
    )
    with ManifestBuilder(dataset_dir, verbose=verbose) as builder:
        for row in rows(path, COLUMNS):
            # Every sentence is different, so the clip is the group. The corpus has
            # only two synthetic voices per configuration: as groups they would give
            # an interval resampled from two values and a sealed panel holding one
            # whole voice. The voice is kept as the category instead, so a report can
            # still tell them apart.
            builder.add(
                f"{name}:{row['ID']}",
                partial(audio_of, row["audio"]),
                language=LANGUAGE,
                text=row["utterance"],
                text_has_casing_punct=True,
                category=f"voice/{row['voice']}",
                group=f"{name}:{row['ID']}",
                group_kind="clip",
                source_split=SPLIT,
                license=LICENSE,
            )
        require_clips(name, len(builder.clips), EXPECTED_CLIPS[config])
        builder.write()
    return len(builder.clips)
