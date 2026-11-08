"""PeerTube French talks (``peertube-fr-train``): conference speech with model labels.

The videos come from the pinned listing of :mod:`audiolab.data.peertube`:
free-software talks in French, full of English technical terms, under CC BY or
CC0. Nobody transcribed them, so the labels are made here by served models, in
three passes:

1. ``stt-c-whisper-large-v3`` transcribes each whole talk in chunks and answers
   with timestamped segments, which :mod:`audiolab.data.peertube_clips` joins
   into sentences: the candidate clips.
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
passes. The models never alternate clip by clip (D-020). :func:`batch_size`
says how many talks a batch reads. Videos are read in a seeded order
(:func:`~audiolab.data.peertube.reading_order`), and every answer is cached per
video under ``$AUDIOLAB_CACHE/<name>``, in files named after the model, so an
interrupted preparation resumes where it stopped. A video that cannot be
downloaded and a request that fails are counted and skipped, and asked again by
the next preparation.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import islice
from pathlib import Path

import httpx

from audiolab.core.audio import load_audio
from audiolab.data import peertube
from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError
from audiolab.data.peertube import Video
from audiolab.data.peertube_clips import (
    CHUNK_SECONDS,
    LANGUAGE,
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

SPLIT = "listing"
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
# The share of words the two labels may differ by after normalization. One
# word in a clip of seven passes; two do not. Above that, one of the models
# misheard, or the cut missed a word.
MAX_DISAGREEMENT = 0.15
# What the first batch counts on: the first six talks of the reading order
# gave 142 kept clips each on average.
EXPECTED_CLIPS_PER_TALK = 140
# Later batches count on the yield seen so far, times this margin.
BATCH_MARGIN = 1.3
# The counts the report is made of.
LABELLED = "clips labelled"
DISAGREED = "clips where Whisper and Qwen disagree"
KEPT = "clips kept"
BASE_WRONG = "kept clips the base model gets wrong"


@dataclass
class Talk:
    """One video on its way through the passes: its audio, segments and agreed clips.

    A talk that could not be downloaded or heard has no audio or no segments,
    and goes through the later passes without a request.
    """

    video: Video
    audio_path: Path | None
    segments: list[Segment] = field(default_factory=list)
    agreed: list[tuple[Segment, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Models:
    """The three models, each as a function from a WAV to its JSON answer."""

    whisper: Transcribe
    qwen: Transcribe
    parakeet: Transcribe


def fetch_peertube_fr_train(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the pinned PeerTube talks up to the cap, returning the clip count."""
    listed_on, videos = peertube.read_listing()
    cache = cache_dir / name
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    take = Take(
        name=name,
        builder=builder,
        language=LANGUAGE,
        source_split=SPLIT,
        license="CC-BY",
        text_has_casing_punct=True,
        group_kind="recording",
        cap=TRAIN_CLIP_CAP,
        cap_per_group=CLIP_CAP_PER_GROUP,
    )
    tally: Counter[str] = Counter()
    if verbose:
        print(f"  listing of {listed_on}: {len(videos)} videos")
    order = peertube.reading_order(videos, SEED)
    with (
        peertube.http_client() as http,
        _client(SEGMENT_MODEL, {"response_format": "verbose_json"}) as whisper,
        _client(LABEL_MODEL) as qwen,
        _client(BASE_MODEL) as parakeet,
        closing(prefetched(order, partial(_audio_file, http, cache))) as audio_files,
    ):
        models = Models(_answer(whisper), _answer(qwen), _answer(parakeet))
        talks_read = 0
        while not take.full():
            batch = islice(audio_files, batch_size(take, talks_read))
            read = prepare_batch(batch, cache, models, take, tally, verbose=verbose)
            if not read:
                break
            talks_read += read
    if verbose:
        print(f"  {report(tally)}")
        print(f"  {take.report()}")
    builder.write()
    return take.kept


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
    batch: Iterator[tuple[Video, Path | None]],
    cache: Path,
    models: Models,
    take: Take,
    tally: Counter[str],
    *,
    verbose: bool,
) -> int:
    """Run the three passes over a batch of talks, one model after the other.

    The batch is read as the Whisper pass goes, so the next talk downloads while
    this one is transcribed. Returns the number of talks read.
    """
    started = time.monotonic()
    talks = [heard_talk(video, path, cache, models.whisper, tally) for video, path in batch]
    segmented = time.monotonic()
    room = take.cap - take.kept
    for talk in talks:
        answers = Answers(cache / clips_file(talk.video, LABEL_MODEL))
        talk.agreed = agreeing(talk, answers, models.qwen, min(take.cap_per_group, room), tally)
        room -= len(talk.agreed)
    labelled = time.monotonic()
    for talk in talks:
        answers = Answers(cache / clips_file(talk.video, BASE_MODEL))
        offer_talk(talk, answers, models.parakeet, take, tally)
    if verbose and talks:
        hours = sum(talk.video.duration_s for talk in talks) / 3600
        print(
            f"  {len(talks)} talks ({hours:.1f} h): segments in {segmented - started:.0f} s, "
            f"labels in {labelled - segmented:.0f} s, base model in "
            f"{time.monotonic() - labelled:.0f} s; {take.kept} clips so far"
        )
    return len(talks)


def prefetched(
    videos: list[Video], fetch: Callable[[Video], Path | None]
) -> Iterator[tuple[Video, Path | None]]:
    """Each video with its audio file, the next one downloading while this one is used."""
    with ThreadPoolExecutor(max_workers=1) as downloads:
        upcoming = downloads.submit(fetch, videos[0]) if videos else None
        for position, video in enumerate(videos):
            path = upcoming.result()
            if position + 1 < len(videos):
                upcoming = downloads.submit(fetch, videos[position + 1])
            yield video, path


def heard_talk(
    video: Video, audio_path: Path | None, cache: Path, whisper: Transcribe, tally: Counter[str]
) -> Talk:
    """The Whisper pass on one talk: its segments, or none if it could not be heard."""
    if audio_path is None:
        tally["talks that could not be downloaded"] += 1
        return Talk(video, None)
    answers = Answers(cache / f"{video.uuid}.{SEGMENT_MODEL}.{CHUNK_SECONDS:g}s.jsonl")
    try:
        segments = segment_audio(load_audio(audio_path), whisper, answers)
    except TranscriptionError as failure:
        print(f"  warning: {SEGMENT_MODEL} failed on {video.url}: {failure}")
        tally["talks Whisper failed on"] += 1
        return Talk(video, audio_path)
    return Talk(video, audio_path, segments)


def agreeing(
    talk: Talk, answers: Answers, qwen: Transcribe, share: int, tally: Counter[str]
) -> list[tuple[Segment, str]]:
    """The Qwen pass on one talk: label its candidate clips, keeping those the models agree on.

    Stops at ``share`` agreed clips, so the end of a long talk is not labelled
    for nothing. Returns each agreed clip with its label.
    """
    if not talk.segments:
        return []
    audio = load_audio(talk.audio_path)
    agreed: list[tuple[Segment, str]] = []
    for candidate in candidates(talk.segments, tally):
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
        if disagreement(candidate.text, label) > MAX_DISAGREEMENT:
            tally[DISAGREED] += 1
            continue
        agreed.append((candidate, label))
    return agreed


def offer_talk(
    talk: Talk, answers: Answers, parakeet: Transcribe, take: Take, tally: Counter[str]
) -> int:
    """The Parakeet pass on one talk: offer its agreed clips, each with the base model's WER."""
    if not talk.agreed:
        return 0
    audio = load_audio(talk.audio_path)
    video = talk.video
    kept = 0
    for clip, label in talk.agreed:
        try:
            heard = clip_text(answers, clip, audio, parakeet)
            base_wer = round(counts(label, heard, LANGUAGE, "normalized").wer, 4)
        except TranscriptionError:
            tally["kept clips Parakeet failed on (no base_wer)"] += 1
            base_wer = None
        offered = take.offer(
            f"{video.uuid}-{round(clip.start_s * 100):07d}",
            cut(audio, clip),
            text=label,
            group=video.uuid,
            license=video.license,
            source_url=video.url,
            source_title=video.title,
            source_author=video.account,
            base_wer=base_wer,
        )
        if offered:
            kept += 1
            if base_wer is not None and base_wer > 0:
                tally[BASE_WRONG] += 1
    tally[KEPT] += kept
    return kept


def disagreement(whisper_text: str, label: str) -> float:
    """The word error rate of the label against Whisper's text, in the normalized view.

    The normalized view ignores casing and punctuation and reads numbers as
    words, so ``2024`` and ``deux mille vingt-quatre`` agree: the test is about
    what was heard, not how it was written.
    """
    return counts(whisper_text, label, LANGUAGE, "normalized").wer


def clips_file(video: Video, model: str) -> str:
    return f"{video.uuid}.{model}.clips.jsonl"


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


def _audio_file(http: httpx.Client, cache: Path, video: Video) -> Path | None:
    """A video's audio, or None when it cannot be downloaded: the run goes on without it."""
    try:
        return peertube.fetch_audio(http, video, cache / f"{video.uuid}.wav")
    except DownloadError as failure:
        print(f"  warning: skipping {video.url}: {failure}")
        return None


def _client(model: str, request: dict | None = None) -> TranscriptionClient:
    """A client for one model, from its config in ``configs/models``."""
    config = ModelConfig.from_file(MODEL_CONFIGS / CONFIG_FILES[model])
    config = replace(config, request={**config.request, **(request or {})})
    return TranscriptionClient(
        config, base_url_from_env(config), api_key_from_env(config), timeout_s=REQUEST_TIMEOUT_S
    )


def _answer(client: TranscriptionClient) -> Transcribe:
    return lambda wav: client.transcribe(wav, LANGUAGE).answer
