"""The talk sources, and the one preparation they all share.

Every entry of :data:`SOURCES` is a dataset ``audiolab data fetch`` can prepare.
It says where the pinned listing is, in which language the talks are labelled,
which licences the listing may hold, how a talk's audio is found, and what a
seeded selection spreads over. Adding a talk source is a module that lists talks
and an entry here: the labelling passes, the clips and the caches belong to all
of them.

The licence of the dataset itself is not repeated here: it lives in
:data:`audiolab.data.datasets.DATASETS`, with every other dataset's (D-018).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import httpx

from audiolab.data import debconf, fosdem, peertube, peertube_edu
from audiolab.data.datasets import DATASETS
from audiolab.data.talks import PreparedAudio, Talk, TalkSource, fetch_talks, prepare_audio

DATA = Path(__file__).parent
# The Capitole du Libre instance: free-software talks in French, the first talk
# source of the repository.
CAPITOLE_DU_LIBRE = "https://videos.capitoledulibre.org"

# The licences a listing of the PeerTube sources may hold: CC BY and CC0, the
# two PeerTube ids whose videos may train a published model (D-019).
PEERTUBE_LICENSES = frozenset(peertube.LICENSES.values())

SOURCES: dict[str, TalkSource] = {
    "peertube-fr-train": TalkSource(
        name="peertube-fr-train",
        language="fr",
        allowed_licenses=PEERTUBE_LICENSES,
        split="listing",
        listing_path=DATA / "peertube_listing.json",
        listing_filter={"licence_ids": [1, 7], "language": "fr", "local": True},
        lister=partial(
            peertube.list_instance,
            instance=CAPITOLE_DU_LIBRE,
            filter=peertube.Filter(language="fr"),
        ),
        audio_url=peertube.smallest_file_url,
    ),
    "peertube-edu-fr-train": TalkSource(
        name="peertube-edu-fr-train",
        language="fr",
        allowed_licenses=PEERTUBE_LICENSES,
        split="listing",
        listing_path=DATA / "peertube_edu_listing.json",
        listing_filter={
            "instances": list(peertube_edu.INSTANCES),
            "licence_ids": list(peertube_edu.FILTER.licence_ids),
            "language": peertube_edu.FILTER.language,
            "local": True,
            "min_duration_s": peertube_edu.MIN_TALK_SECONDS,
            "channel_words_left_out": list(peertube_edu.NON_SPEECH_WORDS),
        },
        lister=peertube_edu.list_talks,
        audio_url=peertube.smallest_file_url,
        spread_by=peertube_edu.spread_key,
    ),
    "debconf-en-train": TalkSource(
        name="debconf-en-train",
        language="en",
        allowed_licenses=PEERTUBE_LICENSES,
        split="listing",
        listing_path=DATA / "debconf_listing.json",
        listing_filter={
            "instance": debconf.INSTANCE,
            "licence_ids": list(debconf.FILTER.licence_ids),
            "language": ["en", "untagged"],
            "local": True,
            "min_duration_s": debconf.MIN_TALK_SECONDS,
            "channel_or_account_must_hold": debconf.FILTER.name_must_hold,
            "editions_left_out": list(debconf.NON_ENGLISH_EDITIONS),
        },
        lister=debconf.list_talks,
        audio_url=peertube.smallest_file_url,
        spread_by=debconf.spread_key,
    ),
    "fosdem-en-train": TalkSource(
        name="fosdem-en-train",
        language="en",
        allowed_licenses=frozenset({fosdem.LICENSE}),
        split="listing",
        listing_path=DATA / "fosdem_listing.json",
        listing_filter={
            "schedule": fosdem.SCHEDULE,
            "video_tree": fosdem.VIDEO_TREE,
            "years": list(fosdem.YEARS),
            "language": ["en", "untagged"],
            "license": fosdem.LICENSE,
            "min_duration_s": fosdem.MIN_TALK_SECONDS,
            "join": "the schedule's video link must be a file the tree serves",
        },
        lister=fosdem.list_talks,
        audio_url=fosdem.file_url,
        spread_by=fosdem.spread_key,
    ),
}


def fetch_talk_dataset(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare one talk dataset, whatever its source is."""
    return fetch_talks(
        SOURCES[name], cache_dir, dataset_dir, license=DATASETS[name].license, verbose=verbose
    )


def prepare_talk_audio(
    name: str, cache_dir: Path, hours: float, *, verbose: bool = True
) -> PreparedAudio:
    """Download the audio of the first talks of a talk source, without labelling them."""
    return prepare_audio(SOURCES[name], cache_dir, hours, verbose=verbose)


def list_talks(name: str, http: httpx.Client) -> list[Talk]:
    """Read the live catalogue of a talk source: what ``audiolab data list-talks`` pins."""
    return SOURCES[name].lister(http)
