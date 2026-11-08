"""PeerTube French talks (``peertube-fr-train``): conference speech with model labels.

The videos come from the pinned listing of :mod:`audiolab.data.peertube`:
free-software talks in French, full of English technical terms, under CC BY or
CC0. Nobody transcribed them, so the labels are made here by served models, in
three passes:

1. ``stt-c-whisper-large-v3`` transcribes each whole talk in chunks and answers
   with timestamped segments, which are joined into sentences: the candidate
   clips. Segments Whisper itself marks as doubtful — a repetition loop, or
   silence it filled with text — are left out.
2. ``stt-a-qwen3-asr-1.7B`` transcribes every candidate clip on its own. A clip
   is kept when the two texts agree (:data:`MAX_DISAGREEMENT`), and its label is
   the Qwen text, which is cased and punctuated. The agreement test checks the
   cut as well as the words: a clip whose timestamps miss a word is heard
   differently by the model that only gets the clip.
3. ``stt-c-parakeet-tdt-0.6B-v3``, the base model of the fine-tune, transcribes
   every kept clip, and the manifest records its word error rate against the
   label (``base_wer``). It does not vote: a clip the base model already gets
   right teaches it nothing, so the clips it gets wrong are the ones a training
   mix wants, and the field lets the mix find them.

The server holds one model at a time and swaps it in on demand, so a pass sends
all its requests for a batch of talks before the next pass starts; the models
never alternate clip by clip. A batch is as many talks as could still fill the
cap if each gave its full share (``CLIP_CAP_PER_GROUP``), so no talk is
transcribed that the cap could not use. Videos are read in a seeded order
(:func:`~audiolab.data.peertube.reading_order`), and each pass caches its answers
per video under ``$AUDIOLAB_CACHE/<name>``, in files named after the model, so
an interrupted preparation resumes where it stopped.
"""

from __future__ import annotations

import io
import json
import math
import time
import zlib
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path

import httpx
import numpy as np
import soundfile

from audiolab.core.audio import SAMPLE_RATE, Audio, load_audio, to_int16
from audiolab.data import peertube
from audiolab.data.build import ManifestBuilder
from audiolab.data.peertube import Video
from audiolab.data.train import CLIP_CAP_PER_GROUP, MAX_SECONDS, MIN_SECONDS, TRAIN_CLIP_CAP, Take
from audiolab.eval.client import (
    ModelConfig,
    TranscriptionClient,
    api_key_from_env,
    base_url_from_env,
)
from audiolab.eval.metrics import counts
from audiolab.eval.normalize import normalize

LANGUAGE = "fr"
SPLIT = "listing"
SEED = 1
SEGMENT_MODEL = "stt-c-whisper-large-v3"
LABEL_MODEL = "stt-a-qwen3-asr-1.7B"
BASE_MODEL = "stt-c-parakeet-tdt-0.6B-v3"
# Whisper is sent five minutes at a time: the server takes fifteen-minute
# requests, and a short chunk is cheaper to retry.
CHUNK_SECONDS = 300.0
REQUEST_TIMEOUT_S = 900.0
# The label model's server answers HTTP 500 to audio longer than about 19 s,
# so this source's clips stop there rather than at the 20 s of every source.
MAX_CLIP_SECONDS = min(MAX_SECONDS, 19.0)
# The share of words the two labels may differ by after normalization. One
# word in a clip of seven passes; two do not. Above that, one of the models
# misheard, or the cut missed a word.
MAX_DISAGREEMENT = 0.15
# Whisper's own hallucination rules, the defaults of openai-whisper: a text that
# compresses better than 2.4 to 1 is a loop, and a segment that is probably not
# speech and was decoded with low confidence is silence filled with text.
MAX_COMPRESSION_RATIO = 2.4
NO_SPEECH_PROB = 0.6
LOW_LOGPROB = -1.0
# Segments are joined across pauses up to this long; a longer pause ends the
# clip. A full stop ends it too, once it lasts CLIP_TARGET_S: a dictated
# request is a sentence or two, and a talk is full of one-word replies ("Oui.")
# that would otherwise make most of the clips.
MAX_PAUSE_S = 1.0
CLIP_TARGET_S = 5.0
SENTENCE_ENDS = (".", "?", "!", "…")
# The counts the report is made of.
LABELLED = "clips labelled"
DISAGREED = "clips where Whisper and Qwen disagree"
KEPT = "clips kept"
BASE_WRONG = "kept clips the base model gets wrong"

# A Python callable that sends one WAV to one model and returns its JSON answer.
Transcribe = Callable[[bytes], dict]


@dataclass(frozen=True)
class Segment:
    """One timestamped segment of Whisper's answer, in seconds from the video start."""

    start_s: float
    end_s: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0

    @property
    def key(self) -> str:
        """What a cached label belongs to: the stretch of audio it was heard in."""
        return f"{self.start_s:.2f}-{self.end_s:.2f}"


@dataclass
class Talk:
    """One video on its way through the passes: its audio, segments and kept clips."""

    video: Video
    audio_path: Path
    segments: list[Segment]
    agreed: list[tuple[Segment, str]] = field(default_factory=list)


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
        while not take.full():
            # As many talks as could still fill the cap, if each gave its full share.
            needed = math.ceil((take.cap - take.kept) / take.cap_per_group)
            batch = list(islice(audio_files, needed))
            if not batch:
                break
            timing = prepare_batch(
                batch, cache, _answer(whisper), _answer(qwen), _answer(parakeet), take, tally
            )
            if verbose:
                print(f"  {timing}; {take.kept} clips so far")
    if verbose:
        print(f"  {report(tally)}")
        print(f"  {take.report()}")
    builder.write()
    return take.kept


def prepare_batch(
    batch: list[tuple[Video, Path]],
    cache: Path,
    whisper: Transcribe,
    qwen: Transcribe,
    parakeet: Transcribe,
    take: Take,
    tally: Counter[str],
) -> str:
    """Run the three passes over a batch of talks, one model after the other.

    Returns a line saying how long each pass took.
    """
    started = time.monotonic()
    talks = [
        Talk(video, path, cached_segments(path, whisper, cache / segments_file(video)))
        for video, path in batch
    ]
    segmented = time.monotonic()
    room = take.cap - take.kept
    for talk in talks:
        labels = ClipTexts(cache / clip_texts_file(talk.video, LABEL_MODEL))
        talk.agreed = agreeing(talk, labels, qwen, min(take.cap_per_group, room), tally)
        room -= len(talk.agreed)
    labelled = time.monotonic()
    for talk in talks:
        offer_talk(
            talk, ClipTexts(cache / clip_texts_file(talk.video, BASE_MODEL)), parakeet, take, tally
        )
    hours = sum(talk.video.duration_s for talk in talks) / 3600
    return (
        f"{len(talks)} talks ({hours:.1f} h): segments in {segmented - started:.0f} s, "
        f"labels in {labelled - segmented:.0f} s, base model in {time.monotonic() - labelled:.0f} s"
    )


def prefetched(videos: list[Video], fetch: Callable[[Video], Path]) -> Iterator[tuple[Video, Path]]:
    """Each video with its audio file, the next one downloading while this one is used."""
    with ThreadPoolExecutor(max_workers=1) as downloads:
        upcoming = downloads.submit(fetch, videos[0]) if videos else None
        for position, video in enumerate(videos):
            path = upcoming.result()
            if position + 1 < len(videos):
                upcoming = downloads.submit(fetch, videos[position + 1])
            yield video, path


def agreeing(
    talk: Talk, labels: ClipTexts, transcribe: Transcribe, share: int, tally: Counter[str]
) -> list[tuple[Segment, str]]:
    """Label the candidate clips of one talk in order, keeping those the models agree on.

    Stops at ``share`` kept clips, so the end of a long talk is not labelled for
    nothing. Returns each kept clip with its label.
    """
    audio = load_audio(talk.audio_path)
    agreed: list[tuple[Segment, str]] = []
    for candidate in candidates(talk.segments, tally):
        if len(agreed) >= share:
            break
        duration = candidate.end_s - candidate.start_s
        if not MIN_SECONDS <= duration <= MAX_CLIP_SECONDS:
            tally[f"clips outside {MIN_SECONDS}-{MAX_CLIP_SECONDS:g} s"] += 1
            continue
        label = labels.text(candidate, audio, transcribe)
        tally[LABELLED] += 1
        if disagreement(candidate.text, label) > MAX_DISAGREEMENT:
            tally[DISAGREED] += 1
            continue
        agreed.append((candidate, label))
    return agreed


def offer_talk(
    talk: Talk, base: ClipTexts, transcribe: Transcribe, take: Take, tally: Counter[str]
) -> int:
    """Offer the kept clips of one talk, each with the base model's WER on it."""
    audio = load_audio(talk.audio_path)
    video = talk.video
    kept = 0
    for candidate, label in talk.agreed:
        heard = base.text(candidate, audio, transcribe)
        base_wer = counts(label, heard, LANGUAGE, "normalized").wer
        offered = take.offer(
            f"{video.uuid}-{round(candidate.start_s * 100):07d}",
            cut(audio, candidate),
            text=label,
            group=video.uuid,
            license=video.license,
            source_url=video.url,
            source_title=video.title,
            source_author=video.account,
            base_wer=round(base_wer, 4),
        )
        kept += offered
        tally[BASE_WRONG] += offered and base_wer > 0
    tally[KEPT] += kept
    return kept


def candidates(segments: list[Segment], tally: Counter[str]) -> list[Segment]:
    """Whisper's segments joined into sentences: the clips that may be labelled.

    Whisper answers in phrases (a median of 1.4 s on a talk), and a phrase cut
    out of a sentence is a poor clip. Consecutive segments are joined until a
    full stop past :data:`CLIP_TARGET_S`, a pause longer than
    :data:`MAX_PAUSE_S`, or the training length limit. A doubtful segment is
    left out and ends the clip it was in.
    """
    joined: list[Segment] = []
    sentence: list[Segment] = []

    def close() -> None:
        if sentence:
            joined.append(_joined(sentence))
            sentence.clear()

    for segment in segments:
        reason = doubt(segment)
        if reason:
            tally[f"segments {reason}"] += 1
            close()
            continue
        if sentence and (
            segment.start_s - sentence[-1].end_s > MAX_PAUSE_S
            or segment.end_s - sentence[0].start_s > MAX_CLIP_SECONDS
        ):
            close()
        sentence.append(segment)
        lasted = segment.end_s - sentence[0].start_s
        if segment.text.endswith(SENTENCE_ENDS) and lasted >= CLIP_TARGET_S:
            close()
    close()
    return joined


def _joined(segments: list[Segment]) -> Segment:
    text = " ".join(segment.text for segment in segments)
    return Segment(start_s=segments[0].start_s, end_s=segments[-1].end_s, text=text)


def segments_file(video: Video) -> str:
    return f"{video.uuid}.{SEGMENT_MODEL}.{CHUNK_SECONDS:g}s.json"


def cached_segments(audio_path: Path, transcribe: Transcribe, path: Path) -> list[Segment]:
    """The segments of one video, asked once and kept in the cache at ``path``."""
    if path.is_file():
        return [Segment(**row) for row in json.loads(path.read_text(encoding="utf-8"))]
    segments = segment_audio(load_audio(audio_path), transcribe)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(segment) for segment in segments]
    path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    return segments


def segment_audio(audio: Audio, transcribe: Transcribe) -> list[Segment]:
    """Every segment Whisper hears in the audio, sent in chunks of ``CHUNK_SECONDS``.

    A chunk ends at a fixed time, which may fall inside a word, so its last
    segment is not trusted: the next chunk starts where the segment before it
    ended, and hears that sentence again whole.
    """
    segments: list[Segment] = []
    start = 0.0
    while audio.duration_s - start >= MIN_SECONDS:
        end = min(start + CHUNK_SECONDS, audio.duration_s)
        chunk = audio.samples[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)]
        heard = [_segment(row, start) for row in transcribe(wav_bytes(chunk))["segments"]]
        if end >= audio.duration_s:
            return segments + heard
        heard = heard[:-1]
        segments.extend(heard)
        start = heard[-1].end_s if heard and heard[-1].end_s > start else end
    return segments


def _segment(row: dict, offset_s: float) -> Segment:
    return Segment(
        start_s=round(offset_s + float(row["start"]), 2),
        end_s=round(offset_s + float(row["end"]), 2),
        text=" ".join(str(row["text"]).split()),
        no_speech_prob=float(row.get("no_speech_prob", 0.0)),
        avg_logprob=float(row.get("avg_logprob", 0.0)),
    )


def doubt(segment: Segment) -> str:
    """Why Whisper's own answer says a segment is not speech, or an empty string."""
    if not normalize(segment.text, LANGUAGE):
        return "without words"
    if compression_ratio(segment.text) > MAX_COMPRESSION_RATIO:
        return "in a repetition loop"
    if segment.no_speech_prob > NO_SPEECH_PROB and segment.avg_logprob < LOW_LOGPROB:
        return "not speech"
    return ""


def compression_ratio(text: str) -> float:
    """How much better than plain text a text compresses: high for a loop."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data))


def disagreement(whisper_text: str, label: str) -> float:
    """The word error rate of the label against Whisper's text, in the normalized view.

    The normalized view ignores casing and punctuation and reads numbers as
    words, so ``2024`` and ``deux mille vingt-quatre`` agree: the test is about
    what was heard, not how it was written.
    """
    return counts(whisper_text, label, LANGUAGE, "normalized").wer


def cut(audio: Audio, segment: Segment) -> Audio:
    """The samples of one segment, copied so the whole talk is not kept alive."""
    start, end = round(segment.start_s * SAMPLE_RATE), round(segment.end_s * SAMPLE_RATE)
    return Audio(samples=audio.samples[start:end].copy(), sample_rate=SAMPLE_RATE)


def wav_bytes(samples: np.ndarray) -> bytes:
    """16 kHz mono samples as a 16-bit WAV in memory: what a request sends."""
    buffer = io.BytesIO()
    soundfile.write(buffer, to_int16(samples), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def clip_texts_file(video: Video, model: str) -> str:
    return f"{video.uuid}.{model}.jsonl"


class ClipTexts:
    """What one model heard in the clips of one video, appended as it arrives.

    One JSON line per clip, keyed by the stretch of audio the text was heard in,
    so a clip cut differently is transcribed again.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.texts: dict[str, str] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.texts[row["segment"]] = row["text"]

    def text(self, clip: Segment, audio: Audio, transcribe: Transcribe) -> str:
        """The cached text of a clip, or the model's answer, cached on the way."""
        if clip.key not in self.texts:
            answer = transcribe(wav_bytes(cut(audio, clip).samples))
            self.texts[clip.key] = " ".join(str(answer["text"]).split())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {"segment": clip.key, "text": self.texts[clip.key]}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self.texts[clip.key]


def report(tally: Counter[str]) -> str:
    """One line on the labelling: the rejection rate, the base model, the rest."""
    labelled, rejected = tally[LABELLED], tally[DISAGREED]
    kept, wrong = tally[KEPT], tally[BASE_WRONG]
    others = [f"{count} {reason}" for reason, count in sorted(tally.items())]
    return (
        f"{labelled} clips labelled, {rejected} ({rejected / max(labelled, 1):.1%}) rejected "
        f"because Whisper and Qwen disagree; the base model gets {wrong} of the {kept} kept "
        f"clips wrong ({wrong / max(kept, 1):.1%}); counts: {', '.join(others)}"
    )


def _audio_file(http: httpx.Client, cache: Path, video: Video) -> Path:
    return peertube.fetch_audio(http, video, cache / f"{video.uuid}.wav")


def _client(model: str, request: dict | None = None) -> TranscriptionClient:
    """A client of this machine's server for one model (``.env`` names it)."""
    config = ModelConfig(name=model, model=model, api_key_env="STT_API_KEY", request=request or {})
    return TranscriptionClient(
        config, base_url_from_env(config), api_key_from_env(config), timeout_s=REQUEST_TIMEOUT_S
    )


def _answer(client: TranscriptionClient) -> Transcribe:
    return lambda wav: client.transcribe(wav, LANGUAGE).answer
