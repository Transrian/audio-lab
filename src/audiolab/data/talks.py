"""Talks from a talk source: the pinned list, the audio behind it, and their labels.

A *talk source* is a place that publishes recordings of talks: a PeerTube
instance, a conference's file server. Everything the pipeline needs from one is
what :class:`TalkSource` holds — the pinned list of talks it may use, the
licences that list may hold, how to find a talk's audio, the language the models
listen in, and the attribution a clip must carry. The modules that read a
particular catalogue (one per source, in :data:`audiolab.data.talk_sources.SOURCES`)
only produce lists of :class:`Talk`; everything after that is here and is shared.

A catalogue is live: videos are published, edited and deleted, and a training
source has to be reproducible. So the list of usable talks is made once, by
``audiolab data list-talks``, and committed as a JSON file: metadata only, with
the date it was listed on and the filter it was made with. A preparation reads
that file and never the catalogue.

Nobody transcribed those talks, so the labels are made here by served models, in
three passes:

1. ``stt-c-whisper-large-v3`` transcribes each whole talk in chunks and answers
   with timestamped segments, which :mod:`audiolab.data.talk_clips` joins into
   sentences: the candidate clips.
2. ``stt-a-qwen3-asr-1.7B`` transcribes every candidate clip on its own. A clip
   is kept when the two texts agree (:data:`MAX_DISAGREEMENT`), and its label is
   always the Qwen text, which is cased and punctuated. The agreement test
   checks the cut as well as the words: a clip whose timestamps miss a word is
   heard differently by the model that only gets the clip.
3. ``stt-c-parakeet-tdt-0.6B-v3``, the base model of the fine-tune, transcribes
   every kept clip, and the manifest records its word error rate against the
   label (``base_wer``). It does not vote: a clip the base model already gets
   right teaches it nothing, so the clips it gets wrong are the ones a training
   mix wants, and the field lets the mix find them.

The server holds one model at a time and swaps it in on demand, so the passes
run over a batch of talks: every Whisper request of the batch, then every Qwen
request, then every Parakeet request, and the next batch repeats the three
passes. The models never alternate clip by clip (D-020). :func:`batch_size` says
how many talks a batch reads. Talks are read in a seeded order
(:func:`reading_order`), and every answer is cached per talk under
``$AUDIOLAB_CACHE/<name>``, in files named after the model, so an interrupted
preparation resumes where it stopped. A talk that cannot be downloaded and a
request that fails are counted and skipped, and asked again by the next
preparation.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from itertools import islice
from pathlib import Path

import httpx

from audiolab.core.audio import SAMPLE_RATE, load_audio
from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, download
from audiolab.data.talk_clips import (
    CHUNK_SECONDS,
    MAX_CLIP_SECONDS,
    Answers,
    Segment,
    Transcribe,
    candidates,
    clip_text,
    cut,
    segment_audio,
)
from audiolab.data.train import CLIP_CAP_PER_GROUP, MIN_SECONDS, TRAIN_CLIP_CAP, Take
from audiolab.eval.client import (
    ModelConfig,
    TranscriptionClient,
    TranscriptionError,
    api_key_from_env,
    base_url_from_env,
)
from audiolab.eval.metrics import counts

SEED = 1
SEGMENT_MODEL = "stt-c-whisper-large-v3"
LABEL_MODEL = "stt-a-qwen3-asr-1.7B"
BASE_MODEL = "stt-c-parakeet-tdt-0.6B-v3"
MODEL_CONFIGS = Path("configs/models")
CONFIG_FILES = {
    SEGMENT_MODEL: "whisper-large-v3-crispasr.yaml",
    LABEL_MODEL: "qwen3-asr-1.7b-audiocpp.yaml",
    BASE_MODEL: "parakeet-tdt-0.6b-v3-crispasr.yaml",
}
REQUEST_TIMEOUT_S = 900.0
# How long a catalogue may take to answer: a listing asks an instance for
# hundreds of pages, a download for one large file.
HTTP_TIMEOUT_S = 120.0
# The share of words the two labels may differ by after normalization. One
# word in a clip of seven passes; two do not. Above that, one of the models
# misheard, or the cut missed a word.
MAX_DISAGREEMENT = 0.15
# What the first batch counts on: the first six talks of peertube-fr-train gave
# 142 kept clips each on average. Another source starts from the same guess, and
# its later batches count on the yield it actually shows.
EXPECTED_CLIPS_PER_TALK = 140
# Later batches count on the yield seen so far, times this margin.
BATCH_MARGIN = 1.3
# How many hours of talk the 12 000-clip cap of one source needs: peertube-fr-train
# keeps about three clips per minute of talk (142 clips for a talk of 44 minutes
# on average), so 12 000 clips come from about 65 hours of talk.
HOURS_PER_CAP = 65.0
# Downloads are network-bound and the instances are public services, so three at
# a time: enough to keep a fast link busy without hammering anyone.
DOWNLOAD_WORKERS = 3
# The counts the report is made of.
LABELLED = "clips labelled"
DISAGREED = "clips where Whisper and Qwen disagree"
KEPT = "clips kept"
BASE_WRONG = "kept clips the base model gets wrong"


@dataclass(frozen=True)
class Talk:
    """One talk of a pinned listing: where it is, what it is, and its licence.

    ``channel`` is the channel, the room or the collection the source files it
    under, and is empty when the source has none. ``instance`` is the PeerTube
    instance whose API serves the video files, and is empty when ``url`` already
    is the audio file: only a PeerTube file URL changes under a talk (an
    instance transcodes again), which is why it is asked for at download time
    rather than pinned.
    """

    id: str
    url: str
    title: str
    duration_s: int
    license: str
    author: str
    channel: str = ""
    instance: str = ""


@dataclass(frozen=True)
class TalkSource:
    """One source of talks, with what the pipeline must know about it.

    ``lister`` reads the live catalogue and returns the talks a listing may hold;
    it is used by ``audiolab data list-talks`` only. ``audio_url`` says how to
    find the file to download for one listed talk. ``listing_filter`` is the
    record of what ``lister`` keeps, written into the listing so a reader can see
    what was asked for and when. ``spread_by`` is what a seeded selection spreads
    over (a conference, an instance, a room), and is what keeps the first talks
    of a source from all coming from the same corner of it.
    """

    name: str
    language: str
    allowed_licenses: frozenset[str]
    listing_path: Path
    listing_filter: dict[str, object]
    lister: Callable[[httpx.Client], list[Talk]]
    audio_url: Callable[[httpx.Client, Talk], str]
    spread_by: Callable[[Talk], str] | None = None


@dataclass
class TalkInProgress:
    """One talk on its way through the passes: its audio, segments and agreed clips.

    A talk that could not be downloaded or heard has no audio or no segments,
    and goes through the later passes without a request.
    """

    talk: Talk
    audio_path: Path | None
    segments: list[Segment] = field(default_factory=list)
    agreed: list[tuple[Segment, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Models:
    """The three models, each as a function from a WAV to its JSON answer."""

    whisper: Transcribe
    qwen: Transcribe
    parakeet: Transcribe


def write_listing(source: TalkSource, listed_on: str, talks: list[Talk]) -> None:
    """Pin the usable talks of one source, with the date and the filter they came from."""
    listing = {
        "source": source.name,
        "listed_on": listed_on,
        "filter": source.listing_filter,
        "talks": [asdict(talk) for talk in talks],
    }
    source.listing_path.write_text(
        json.dumps(listing, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def read_listing(source: TalkSource) -> tuple[str, list[Talk]]:
    """The date of the pinned listing of ``source`` and its talks.

    A talk whose licence is not one this source may train on is refused rather
    than skipped: the file was written by :func:`write_listing` from talks that
    passed the filter, so a licence that does not is a wrong file (D-019).
    """
    listing = json.loads(source.listing_path.read_text(encoding="utf-8"))
    talks = [Talk(**row) for row in listing["talks"]]
    unusable = [talk.id for talk in talks if talk.license not in source.allowed_licenses]
    if unusable:
        raise DownloadError(f"{source.listing_path}: talks under another licence: {unusable[:3]}")
    return listing["listed_on"], talks


def reading_order(
    talks: list[Talk], seed: int, spread_by: Callable[[Talk], str] | None = None
) -> list[Talk]:
    """The listed talks in a seeded order, the same on every machine.

    A catalogue publishes one conference, instance or year after another, so the
    order of the listing would spend the clip cap on a handful of them. The
    talks are shuffled with a fixed seed, which is reproducible, and then taken
    one per group in turn (:func:`spread`), so that the first talks read cover
    every group the source has.
    """
    shuffled = sorted(talks, key=lambda talk: talk.id)
    random.Random(seed).shuffle(shuffled)
    return spread(shuffled, spread_by) if spread_by else shuffled


def spread(talks: list[Talk], group_of: Callable[[Talk], str]) -> list[Talk]:
    """The talks reordered so that each group appears once before any appears twice.

    The groups are the ones of the shuffle's first appearance, and the talks
    inside a group keep their shuffled order, so the result is as reproducible as
    the shuffle it is made from.
    """
    groups: dict[str, list[Talk]] = {}
    for talk in talks:
        groups.setdefault(group_of(talk), []).append(talk)
    longest = max((len(group) for group in groups.values()), default=0)
    return [
        group[position]
        for position in range(longest)
        for group in groups.values()
        if position < len(group)
    ]


def fetch_audio(http: httpx.Client, talk: Talk, destination: Path, source: TalkSource) -> Path:
    """Download one talk and keep only its audio, as a 16 kHz mono WAV.

    The video file is downloaded first (reading it remotely through ffmpeg is
    much slower) and deleted as soon as its audio is out: most of its bytes are
    pixels.
    """
    if destination.is_file():
        return destination
    video_file = destination.with_suffix(".video")
    download(source.audio_url(http, talk), video_file, client=http)
    partial_wav = destination.with_suffix(".partial.wav")
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video_file),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(partial_wav),
    ]  # fmt: skip
    try:
        finished = subprocess.run(command, capture_output=True, text=True)
    finally:
        video_file.unlink(missing_ok=True)
    if finished.returncode != 0:
        partial_wav.unlink(missing_ok=True)
        raise DownloadError(f"ffmpeg failed on {talk.url}: {finished.stderr.strip()[:200]}")
    partial_wav.rename(destination)
    return destination


def http_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT_S)


def fetch_talks(
    source: TalkSource, cache_dir: Path, dataset_dir: Path, *, license: str, verbose: bool
) -> int:
    """Prepare the pinned talks of ``source`` up to the clip cap, returning the clip count.

    ``license`` is what the dataset's registry entry states (D-018); every clip
    is recorded under the licence of its own talk all the same, which is what a
    listing of several licences needs.
    """
    listed_on, talks = read_listing(source)
    cache = cache_dir / source.name
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    take = Take(
        name=source.name,
        builder=builder,
        language=source.language,
        source_split="listing",
        license=license,
        # A talk is one recording, and its labels are what Qwen wrote, casing
        # and punctuation included: no talk source differs here.
        text_has_casing_punct=True,
        group_kind="recording",
        cap=TRAIN_CLIP_CAP,
        cap_per_group=CLIP_CAP_PER_GROUP,
    )
    tally: Counter[str] = Counter()
    if verbose:
        print(f"  listing of {listed_on}: {len(talks)} talks")
    order = reading_order(talks, SEED, source.spread_by)
    with (
        http_client() as http,
        _client(SEGMENT_MODEL, source.language, {"response_format": "verbose_json"}) as whisper,
        _client(LABEL_MODEL, source.language) as qwen,
        _client(BASE_MODEL, source.language) as parakeet,
        closing(prefetched(order, partial(_audio_file, http, cache, source))) as audio_files,
    ):
        models = Models(
            _answer(whisper, source.language),
            _answer(qwen, source.language),
            _answer(parakeet, source.language),
        )
        talks_read = 0
        while not take.full():
            batch = islice(audio_files, batch_size(take, talks_read))
            read = prepare_batch(batch, cache, models, take, source, tally, verbose=verbose)
            if not read:
                break
            talks_read += read
    if verbose:
        print(f"  {report(tally)}")
        print(f"  {take.report()}")
    builder.write()
    return take.kept


@dataclass(frozen=True)
class PreparedAudio:
    """What preparing the audio of a source produced, before any model heard it."""

    talks: int
    hours: float
    failed: int


def prepare_audio(
    source: TalkSource, cache_dir: Path, hours: float = HOURS_PER_CAP, *, verbose: bool = True
) -> PreparedAudio:
    """Download the audio of the talks a labelling run reads first, and nothing else.

    The labelling passes need no network beyond the audio, so this step runs
    while the speech server is busy with another dataset (D-020); the labelling
    run then finds every file in its cache and starts listening at once. The
    talks are the first ones of the reading order, so the audio prepared here is
    the audio the next run reads.
    """
    listed_on, talks = read_listing(source)
    order = reading_order(talks, SEED, source.spread_by)
    wanted = first_talks_over(order, hours)
    cache = cache_dir / source.name
    cache.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  listing of {listed_on}: {len(talks)} talks; preparing {len(wanted)} of them")
    done: list[Talk] = []
    failed = 0
    with http_client() as http, ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as downloads:
        heard = downloads.map(partial(_audio_file, http, cache, source), wanted)
        for position, (talk, path) in enumerate(zip(wanted, heard, strict=True), start=1):
            if path is None:
                failed += 1
            else:
                done.append(talk)
            if verbose and position % 25 == 0:
                print(f"  ... {position} talks read, {len(done)} with audio")
    prepared = PreparedAudio(len(done), sum(talk.duration_s for talk in done) / 3600, failed)
    if verbose:
        print(
            f"  {prepared.talks} talks ({prepared.hours:.1f} h) ready in {cache}, "
            f"{prepared.failed} not downloadable"
        )
    return prepared


def first_talks_over(order: list[Talk], hours: float) -> list[Talk]:
    """The shortest prefix of a reading order that holds at least ``hours`` of talk."""
    wanted: list[Talk] = []
    total = 0.0
    for talk in order:
        wanted.append(talk)
        total += talk.duration_s / 3600
        if total >= hours:
            break
    return wanted


def batch_size(take: Take, talks_read: int) -> int:
    """How many talks the next batch reads, to fill the cap at the yield expected.

    A batch too small costs one more round of three model loads, a few seconds
    each; a batch too large costs minutes of Whisper on every talk the cap
    cannot use. So the first batch counts on :data:`EXPECTED_CLIPS_PER_TALK`
    without a margin, and the later, small batches count on the yield seen so
    far with :data:`BATCH_MARGIN`, so that they rarely fall short again.
    """
    remaining = take.cap - take.kept
    if not talks_read:
        return math.ceil(remaining / EXPECTED_CLIPS_PER_TALK)
    clips_per_talk = max(1.0, take.kept / talks_read)
    return math.ceil(remaining / clips_per_talk * BATCH_MARGIN)


def prepare_batch(
    batch: Iterator[tuple[Talk, Path | None]],
    cache: Path,
    models: Models,
    take: Take,
    source: TalkSource,
    tally: Counter[str],
    *,
    verbose: bool,
) -> int:
    """Run the three passes over a batch of talks, one model after the other.

    The batch is read as the Whisper pass goes, so the next talk downloads while
    this one is transcribed. Returns the number of talks read.
    """
    started = time.monotonic()
    talks = [heard_talk(talk, path, cache, models.whisper, tally) for talk, path in batch]
    segmented = time.monotonic()
    room = take.cap - take.kept
    for talk in talks:
        answers = Answers(cache / clips_file(talk.talk, LABEL_MODEL))
        talk.agreed = agreeing(
            talk, answers, models.qwen, source.language, min(take.cap_per_group, room), tally
        )
        room -= len(talk.agreed)
    labelled = time.monotonic()
    for talk in talks:
        answers = Answers(cache / clips_file(talk.talk, BASE_MODEL))
        offer_talk(talk, answers, models.parakeet, take, source.language, tally)
    if verbose and talks:
        hours = sum(talk.talk.duration_s for talk in talks) / 3600
        print(
            f"  {len(talks)} talks ({hours:.1f} h): segments in {segmented - started:.0f} s, "
            f"labels in {labelled - segmented:.0f} s, base model in "
            f"{time.monotonic() - labelled:.0f} s; {take.kept} clips so far"
        )
    return len(talks)


def prefetched(
    talks: list[Talk], fetch: Callable[[Talk], Path | None]
) -> Iterator[tuple[Talk, Path | None]]:
    """Each talk with its audio file, the next one downloading while this one is used."""
    with ThreadPoolExecutor(max_workers=1) as downloads:
        upcoming = downloads.submit(fetch, talks[0]) if talks else None
        for position, talk in enumerate(talks):
            path = upcoming.result()
            if position + 1 < len(talks):
                upcoming = downloads.submit(fetch, talks[position + 1])
            yield talk, path


def heard_talk(
    talk: Talk, audio_path: Path | None, cache: Path, whisper: Transcribe, tally: Counter[str]
) -> TalkInProgress:
    """The Whisper pass on one talk: its segments, or none if it could not be heard."""
    if audio_path is None:
        tally["talks that could not be downloaded"] += 1
        return TalkInProgress(talk, None)
    answers = Answers(cache / f"{talk.id}.{SEGMENT_MODEL}.{CHUNK_SECONDS:g}s.jsonl")
    try:
        segments = segment_audio(load_audio(audio_path), whisper, answers)
    except TranscriptionError as failure:
        print(f"  warning: {SEGMENT_MODEL} failed on {talk.url}: {failure}")
        tally["talks Whisper failed on"] += 1
        return TalkInProgress(talk, audio_path)
    return TalkInProgress(talk, audio_path, segments)


def agreeing(
    talk: TalkInProgress,
    answers: Answers,
    qwen: Transcribe,
    language: str,
    share: int,
    tally: Counter[str],
) -> list[tuple[Segment, str]]:
    """The Qwen pass on one talk: label its candidate clips, keeping those the models agree on.

    Stops at ``share`` agreed clips, so the end of a long talk is not labelled
    for nothing. Returns each agreed clip with its label.
    """
    if not talk.segments:
        return []
    audio = load_audio(talk.audio_path)
    agreed: list[tuple[Segment, str]] = []
    for candidate in candidates(talk.segments, language, tally):
        if len(agreed) >= share:
            break
        duration = candidate.end_s - candidate.start_s
        if not MIN_SECONDS <= duration <= MAX_CLIP_SECONDS:
            tally[f"clips outside {MIN_SECONDS}-{MAX_CLIP_SECONDS:g} s"] += 1
            continue
        try:
            label = clip_text(answers, candidate, audio, qwen)
        except TranscriptionError:
            tally["clips Qwen failed on"] += 1
            continue
        tally[LABELLED] += 1
        if disagreement(candidate.text, label, language) > MAX_DISAGREEMENT:
            tally[DISAGREED] += 1
            continue
        agreed.append((candidate, label))
    return agreed


def offer_talk(
    talk: TalkInProgress,
    answers: Answers,
    parakeet: Transcribe,
    take: Take,
    language: str,
    tally: Counter[str],
) -> int:
    """The Parakeet pass on one talk: offer its agreed clips, each with the base model's WER."""
    if not talk.agreed:
        return 0
    audio = load_audio(talk.audio_path)
    source_talk = talk.talk
    kept = 0
    for clip, label in talk.agreed:
        try:
            heard = clip_text(answers, clip, audio, parakeet)
            base_wer = round(counts(label, heard, language, "normalized").wer, 4)
        except TranscriptionError:
            tally["kept clips Parakeet failed on (no base_wer)"] += 1
            base_wer = None
        offered = take.offer(
            f"{source_talk.id}-{round(clip.start_s * 100):07d}",
            cut(audio, clip),
            text=label,
            group=source_talk.id,
            license=source_talk.license,
            source_url=source_talk.url,
            source_title=source_talk.title,
            source_author=source_talk.author,
            base_wer=base_wer,
        )
        if offered:
            kept += 1
            if base_wer is not None and base_wer > 0:
                tally[BASE_WRONG] += 1
    tally[KEPT] += kept
    return kept


def disagreement(whisper_text: str, label: str, language: str) -> float:
    """The word error rate of the label against Whisper's text, in the normalized view.

    The normalized view ignores casing and punctuation and reads numbers as
    words, so ``2024`` and ``deux mille vingt-quatre`` agree: the test is about
    what was heard, not how it was written.
    """
    return counts(whisper_text, label, language, "normalized").wer


def clips_file(talk: Talk, model: str) -> str:
    return f"{talk.id}.{model}.clips.jsonl"


def report(tally: Counter[str]) -> str:
    """One line on the labelling: the rejection rate, the base model, the other counts."""
    labelled, rejected = tally[LABELLED], tally[DISAGREED]
    kept, wrong = tally[KEPT], tally[BASE_WRONG]
    shown = {LABELLED, DISAGREED, KEPT, BASE_WRONG}
    others = [f"{count} {reason}" for reason, count in sorted(tally.items()) if reason not in shown]
    return (
        f"{labelled} clips labelled, {rejected} ({rejected / max(labelled, 1):.1%}) rejected "
        f"because Whisper and Qwen disagree; the base model gets {wrong} of the {kept} kept "
        f"clips wrong ({wrong / max(kept, 1):.1%}); also {', '.join(others) or 'nothing else'}"
    )


def _audio_file(http: httpx.Client, cache: Path, source: TalkSource, talk: Talk) -> Path | None:
    """A talk's audio, or None when it cannot be downloaded: the run goes on without it."""
    try:
        return fetch_audio(http, talk, cache / f"{talk.id}.wav", source)
    except DownloadError as failure:
        print(f"  warning: skipping {talk.url}: {failure}")
        return None


def _client(model: str, language: str, request: dict | None = None) -> TranscriptionClient:
    """A client for one model, from its config in ``configs/models``.

    The language is part of the request, so it is part of the model's answers:
    an English talk is labelled by a client that was told English.
    """
    config = ModelConfig.from_file(MODEL_CONFIGS / CONFIG_FILES[model])
    config = replace(config, request={**config.request, **(request or {})})
    return TranscriptionClient(
        config, base_url_from_env(config), api_key_from_env(config), timeout_s=REQUEST_TIMEOUT_S
    )


def _answer(client: TranscriptionClient, language: str) -> Transcribe:
    return lambda wav: client.transcribe(wav, language).answer
