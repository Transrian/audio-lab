"""Talks from a talk source: the pinned list, the audio behind it, and their labels.

A *talk source* is a place that publishes recordings of talks: a PeerTube
instance, a conference's file server. Everything the pipeline needs from one is
what :class:`TalkSource` holds — the pinned list of talks it may use, the
licences that list may hold, how to find a talk's audio, the language the models
listen in, and the attribution a clip must carry. The modules that read a
particular catalogue (one per source, in :data:`audiolab.data.talk_sources.SOURCES`)
only produce lists of :class:`Talk`; everything after that is here and is shared.

A preparation reads the pinned listing of its source and never the live
catalogue (:mod:`audiolab.data.talk_listing`).

Nobody transcribed those talks, so the labels are made here by served models, in
three passes:

1. ``stt-v-whisper-large-v3`` transcribes each whole talk in chunks and answers
   with timestamped segments, which :mod:`audiolab.data.talk_clips` joins into
   sentences: the candidate clips.
2. ``stt-v-qwen3-asr-1.7B`` transcribes every candidate clip on its own. A clip
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
how many talks a batch reads. A pass sends as many requests at once as its
model takes (:attr:`Served.concurrency`, one unless the environment says more;
D-022) — several talks for Whisper, several clips of one talk for Qwen and
Parakeet — and reads the answers in the order a pass sending one at a time
would, so the clips kept do not depend on which answer arrives first. Its
first request waits for the model to answer a warmup (:meth:`Served.one_pass`).
Talks are read in a seeded order (:func:`reading_order`), and every answer is
cached per talk under ``$AUDIOLAB_CACHE/<name>``, in files named after the
model, so an interrupted preparation resumes where it stopped. The cache is also
what keeps a resumed preparation consistent: vLLM does not answer bit for bit
the same when the requests sent beside one change (a comma, a capital letter).
A talk that cannot be downloaded and a request that fails are counted and
skipped, and asked again by the next preparation.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import islice
from pathlib import Path

import httpx

from audiolab.core.audio import SAMPLE_RATE, load_audio
from audiolab.core.parallel import map_requests
from audiolab.core.progress import progress, say
from audiolab.core.stop import Stopped, raise_if_stopped, stop_requested
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
from audiolab.data.talk_listing import Talk, TalkSource, read_listing, reading_order
from audiolab.data.train import CLIP_CAP_PER_GROUP, MIN_SECONDS, TRAIN_CLIP_CAP, Take
from audiolab.eval.client import (
    ModelConfig,
    TranscriptionClient,
    TranscriptionError,
    api_key_from_env,
    base_url_from_env,
    concurrency_from_env,
)
from audiolab.eval.metrics import counts

SEED = 1
SEGMENT_MODEL = "stt-v-whisper-large-v3"
LABEL_MODEL = "stt-v-qwen3-asr-1.7B"
BASE_MODEL = "stt-c-parakeet-tdt-0.6B-v3"
MODEL_CONFIGS = Path("configs/models")
CONFIG_FILES = {
    SEGMENT_MODEL: "whisper-large-v3-vllm.yaml",
    LABEL_MODEL: "qwen3-asr-1.7b-vllm.yaml",
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
class Served:
    """One served model as a pass asks it: a WAV in, its JSON answer out.

    ``concurrency`` is how many of its requests a pass sends at once, and
    ``warm_up`` waits until the model answers (D-022).
    """

    transcribe: Transcribe
    concurrency: int = 1
    warm_up: Callable[[], None] = lambda: None

    def one_pass(self) -> Served:
        """This model for one pass, warmed up before the first request the pass sends.

        A pass is the batch of one model (D-020). Since its last pass, the server
        may have unloaded the model, and loading it again can take minutes, so
        the first request of every pass waits for a warmup. A pass whose answers
        are all cached sends nothing and warms nothing up: it never makes the
        server load a model for nothing.
        """
        warming = threading.Lock()
        warm = False

        def transcribe(wav: bytes) -> dict:
            nonlocal warm
            with warming:
                if not warm:
                    self.warm_up()
                    warm = True
            return self.transcribe(wav)

        return replace(self, transcribe=transcribe)


@dataclass(frozen=True)
class Models:
    """The three models, as the passes ask them."""

    whisper: Served
    qwen: Served
    parakeet: Served


def fetch_audio(http: httpx.Client, talk: Talk, destination: Path, source: TalkSource) -> Path:
    """Download one talk and keep only its audio, as a 16 kHz mono WAV.

    The video file is downloaded first (reading it remotely through ffmpeg is
    much slower) and deleted as soon as its audio is out: most of its bytes are
    pixels. An ffmpeg that a stop interrupted (Ctrl-C reaches it too) is not a
    talk that failed: the video stays for the next run, which converts it again.
    """
    if destination.is_file():
        return destination
    raise_if_stopped(f"audio of {talk.url} stopped")  # before asking the host for its file
    video_file = destination.with_suffix(".video")
    download(source.audio_url(http, talk), video_file, client=http)
    partial_wav = destination.with_suffix(".partial.wav")
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video_file),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(partial_wav),
    ]  # fmt: skip
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        partial_wav.unlink(missing_ok=True)
        raise_if_stopped(f"audio of {talk.url} stopped")
        video_file.unlink(missing_ok=True)
        raise DownloadError(f"ffmpeg failed on {talk.url}: {finished.stderr.strip()[:200]}")
    video_file.unlink()
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
        client(SEGMENT_MODEL, {"response_format": "verbose_json"}) as whisper,
        client(LABEL_MODEL) as qwen,
        client(BASE_MODEL) as parakeet,
        closing(prefetched(order, partial(audio_file, http, cache, source))) as audio_files,
    ):
        models = Models(
            served(whisper, source.language),
            served(qwen, source.language),
            served(parakeet, source.language),
        )
        talks_read = 0
        bar = progress(desc=f"{source.name} talks", unit="talk")
        while not take.full():
            batch = until_stopped(islice(audio_files, batch_size(take, talks_read)), bar)
            read = prepare_batch(batch, cache, models, take, source, tally, verbose=verbose)
            if not read:
                break
            talks_read += read
        bar.close()
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
    with (
        http_client() as http,
        ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as downloads,
        progress(desc=f"{source.name} talks", unit="talk", total=len(wanted)) as bar,
    ):
        heard = downloads.map(partial(audio_file, http, cache, source), wanted)
        for position, (talk, path) in enumerate(zip(wanted, heard, strict=True), start=1):
            bar.update()
            if path is None:
                failed += 1
            else:
                done.append(talk)
            if verbose and position % 25 == 0:
                say(f"  ... {position} talks read, {len(done)} with audio")
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
    the ones before it are transcribed. Returns the number of talks read. A stop
    takes effect between two talks; every answer is cached, so the next run goes
    through what this one did without asking a model again.
    """
    whisper, qwen, parakeet = (
        models.whisper.one_pass(),
        models.qwen.one_pass(),
        models.parakeet.one_pass(),
    )
    started = time.monotonic()
    stopped = f"labelling of {source.name} stopped"
    talks = heard_talks(batch, cache, whisper.transcribe, tally, whisper.concurrency)
    raise_if_stopped(stopped)
    segmented = time.monotonic()
    room = take.cap - take.kept
    for talk in talks:
        raise_if_stopped(stopped)
        answers = Answers(cache / clips_file(talk.talk, LABEL_MODEL))
        share = min(take.cap_per_group, room)
        talk.agreed = agreeing(
            talk, answers, qwen.transcribe, source.language, share, tally, qwen.concurrency
        )
        room -= len(talk.agreed)
    labelled = time.monotonic()
    for talk in talks:
        raise_if_stopped(stopped)
        answers = Answers(cache / clips_file(talk.talk, BASE_MODEL))
        offer_talk(
            talk, answers, parakeet.transcribe, take, source.language, tally, parakeet.concurrency
        )
    if verbose and talks:
        hours = sum(talk.talk.duration_s for talk in talks) / 3600
        say(
            f"  {len(talks)} talks ({hours:.1f} h): segments in {segmented - started:.0f} s, "
            f"labels in {labelled - segmented:.0f} s, base model in "
            f"{time.monotonic() - labelled:.0f} s; {take.kept} clips so far"
        )
    return len(talks)


def until_stopped[T](items: Iterable[T], bar) -> Iterator[T]:
    """The items one after the other, counted on ``bar``, until a stop is asked for.

    The passes take a batch as it downloads, so a stop also stops the downloads
    of the talks after the one in hand.
    """
    for item in items:
        if stop_requested():
            return
        bar.update()
        yield item


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


def heard_talks(
    batch: Iterable[tuple[Talk, Path | None]],
    cache: Path,
    whisper: Transcribe,
    tally: Counter[str],
    concurrency: int = 1,
    should_stop: Callable[[], bool] = stop_requested,
) -> list[TalkInProgress]:
    """The Whisper pass on a batch: :func:`heard_talk` on ``concurrency`` talks at once.

    The talks come back in batch order. What runs in parallel is talks, not the
    chunks of one talk: a chunk starts where the last trusted segment of the
    one before it ends (:func:`audiolab.data.talk_clips.segment_audio`), so it
    cannot be sent first.

    Once ``should_stop()`` is true, by default once a stop was asked for
    (D-021), no talk starts, and the talks in flight end at their next chunk
    with :class:`~audiolab.core.stop.Stopped`, which is not a talk Whisper
    failed on. A failed talk ends the others the same way, and its failure is
    raised. Either way the answers already written stay cached.
    """
    failing = threading.Event()

    def whisper_until_stopped(wav: bytes) -> dict:
        if failing.is_set() or should_stop():
            raise Stopped("labelling stopped")
        return whisper(wav)

    def heard(pair: tuple[Talk, Path | None]) -> tuple[TalkInProgress, Counter[str]]:
        counted: Counter[str] = Counter()  # one per talk: threads must not share a Counter
        return heard_talk(*pair, cache, whisper_until_stopped, counted), counted

    results = map_requests(heard, batch, concurrency, should_stop=should_stop, failing=failing)
    for _, counted in results:
        tally.update(counted)
    return [talk for talk, _ in results]


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
    concurrency: int = 1,
) -> list[tuple[Segment, str]]:
    """The Qwen pass on one talk: label its candidate clips, keeping those the models agree on.

    Stops at ``share`` agreed clips, so the end of a long talk is not labelled
    for nothing. Sends ``concurrency`` clips at once. Returns each agreed clip
    with its label.
    """
    if not talk.segments:
        return []
    audio = load_audio(talk.audio_path)

    def ask_qwen(candidate: Segment) -> str | None:
        """Qwen's text for one clip; None when the request fails."""
        try:
            return clip_text(answers, candidate, audio, qwen)
        except TranscriptionError:
            return None

    clips = candidates(talk.segments, language, tally)
    fitting = [clip for clip in clips if fits(clip)]
    if len(fitting) < len(clips):
        tally[f"clips outside {MIN_SECONDS}-{MAX_CLIP_SECONDS:g} s"] += len(clips) - len(fitting)
    agreed: list[tuple[Segment, str]] = []
    pending = iter(fitting)
    while len(agreed) < share:
        # Never more at once than the clips still wanted: the talk is sent the
        # requests it would be sent one by one, and stops at the same clip.
        group = list(islice(pending, min(concurrency, share - len(agreed))))
        if not group:
            break
        labels = map_requests(ask_qwen, group, concurrency)
        raise_if_stopped("labelling stopped")
        for candidate, label in zip(group, labels, strict=True):
            if label is None:
                tally["clips Qwen failed on"] += 1
            else:
                tally[LABELLED] += 1
                if disagreement(candidate.text, label, language) > MAX_DISAGREEMENT:
                    tally[DISAGREED] += 1
                else:
                    agreed.append((candidate, label))
    return agreed


def fits(clip: Segment) -> bool:
    """Whether a clip lasts as long as a clip of this source may.

    Its length is rounded as its edges are, so a clip of exactly 0.5 s or 20 s
    fits whatever the float subtraction leaves (32.02 - 12.02 > 20.0).
    """
    return MIN_SECONDS <= round(clip.end_s - clip.start_s, 2) <= MAX_CLIP_SECONDS


def offer_talk(
    talk: TalkInProgress,
    answers: Answers,
    parakeet: Transcribe,
    take: Take,
    language: str,
    tally: Counter[str],
    concurrency: int = 1,
) -> int:
    """The Parakeet pass on one talk: offer its agreed clips, each with the base model's WER.

    Sends ``concurrency`` clips at once, and offers them in the talk's order.
    """
    if not talk.agreed:
        return 0
    audio = load_audio(talk.audio_path)

    def ask_parakeet(agreed: tuple[Segment, str]) -> float | None:
        """The base model's WER on one clip; None when the request fails."""
        clip, label = agreed
        try:
            heard = clip_text(answers, clip, audio, parakeet)
        except TranscriptionError:
            return None
        return round(counts(label, heard, language, "normalized").wer, 4)

    base_wers = map_requests(ask_parakeet, talk.agreed, concurrency)
    raise_if_stopped("labelling stopped")
    source_talk = talk.talk
    kept = 0
    for (clip, label), base_wer in zip(talk.agreed, base_wers, strict=True):
        if base_wer is None:
            tally["kept clips Parakeet failed on (no base_wer)"] += 1
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


def audio_file(http: httpx.Client, cache: Path, source: TalkSource, talk: Talk) -> Path | None:
    """A talk's audio, or None when it cannot be downloaded: the run goes on without it."""
    try:
        return fetch_audio(http, talk, cache / f"{talk.id}.wav", source)
    except DownloadError as failure:
        print(f"  warning: skipping {talk.url}: {failure}")
        return None


def client(model: str, request: dict | None = None) -> TranscriptionClient:
    """A client for one model, from its config in ``configs/models``.

    ``request`` adds fields to every request (Whisper's ``response_format``).
    The language is not the client's: :func:`served` passes the source's
    language with every request, so an English talk is heard in English. One
    client serves the parallel requests of a pass.
    """
    config = ModelConfig.from_file(MODEL_CONFIGS / CONFIG_FILES[model])
    config = replace(config, request={**config.request, **(request or {})})
    return TranscriptionClient(
        config, base_url_from_env(config), api_key_from_env(config), timeout_s=REQUEST_TIMEOUT_S
    )


def served(model: TranscriptionClient, language: str) -> Served:
    """A model as the passes ask it, in ``language``, as many requests at once as it takes."""
    return Served(
        transcribe=lambda wav: model.transcribe(wav, language).answer,
        concurrency=concurrency_from_env(model.config),
        warm_up=model.warm_up,
    )
