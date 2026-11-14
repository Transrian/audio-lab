"""A prepared set with the labels of one model on every clip: ``nasa-en-train-qwen``.

The label-style test (plan 06, Part B) asks what the audio of ``nasa-en-train``
is worth when the labels are a model's instead of NASA's. The test changes only
the labels: the same clips, the same audio, and the text of every clip is the
Qwen3-ASR answer, without the agreement filter — a clip set that were not
identical would mix the label change with a data change.

``audiolab data fetch nasa-en-train-qwen`` (:func:`fetch_nasa_en_train_qwen`)
reads the prepared manifest of ``nasa-en-train`` (it must exist; nothing is
downloaded) and, for every clip in id order:

- keeps the clip's fields — group, duration, ``base_wer``, the attribution, the
  license — and renames the clip and its group under the dataset's own name, so
  the id of a clip is ``nasa-en-train-qwen:<the source clip's suffix>``;
- links the clip's WAV from the source into the new set, with ``os.link``: the
  audio is the same file, not a copy. Only when the link fails (the two
  directories on different filesystems) the file is copied, and the copies are
  counted and reported;
- sets the text to ``plain_label`` of the Qwen answer. The answers come from
  the label audit's cache (``audiolab.data.label_audit``: the same per-clip
  files the audit of the same clips read, so the audit's clips are never asked
  again), and the rest are asked the same way: one model, as many at once as
  the model takes, every answer cached (D-017, D-020, D-022).

A clip whose Qwen answer is empty keeps an empty text: it is not dropped, it
is skipped by the training draw like any empty clip, so the clip set stays the
source's. The empties are counted, and a preparation with more than one per
cent of them is refused: that share would mean the server answered nothing for
a real share of the clips, not that the speech is empty.

Nothing here decides what the relabelled set is worth; it only makes it
comparable to ``nasa-en-train`` — same clips, same audio, different labels.
"""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path

from audiolab.core.audio import load_audio
from audiolab.core.parallel import map_requests
from audiolab.core.progress import say
from audiolab.core.stop import Stopped, raise_if_stopped, stop_requested
from audiolab.data.build import ManifestBuilder, file_name_for
from audiolab.data.download import DownloadError
from audiolab.data.label_audit import answer_path, read_answer, write_answer
from audiolab.data.store import Clip, read_manifest
from audiolab.data.talk_clips import wav_bytes
from audiolab.data.talks import LABEL_MODEL, client, served
from audiolab.data.train import plain_label

# The source this set relabels: the clips, the audio and every field but the
# text of the source's manifest.
SOURCE = "nasa-en-train"
# A share of empty Qwen answers above this refuses the preparation: the server
# answered nothing for a real share of the clips, and the label-style test
# would compare against silence.
EMPTY_ANSWER_LIMIT = 0.01


def fetch_nasa_en_train_qwen(
    name: str, cache: Path, dataset_dir: Path, *, verbose: bool = False
) -> int:
    """Prepare the Qwen-labelled copy of ``nasa-en-train``; the clip count, as a fetcher returns."""
    source_dir = dataset_dir.parent / SOURCE
    if not (source_dir / "manifest.jsonl").is_file():
        raise DownloadError(
            f"{source_dir} holds no manifest: prepare {SOURCE} first (audiolab data fetch {SOURCE})"
        )
    source_clips = read_manifest(source_dir)
    language = source_clips[0].language
    dataset_dir.mkdir(parents=True, exist_ok=True)

    answers = _qwen_answers(
        source_clips, language, cache, source_dir, name=dataset_dir.name, verbose=verbose
    )
    texts = {clip.id: plain_label(answers[clip.id]) for clip in source_clips}
    empty = sum(1 for text in texts.values() if not text.strip())
    if empty > EMPTY_ANSWER_LIMIT * len(source_clips):
        raise DownloadError(
            f"{name}: {empty} of {len(source_clips)} Qwen answers are empty "
            f"({empty / len(source_clips):.1%} > {EMPTY_ANSWER_LIMIT:.0%}): the server did not "
            "answer the clips, the preparation is not a label-style test"
        )
    if verbose:
        say(f"  {name}: {empty} clips with an empty Qwen answer ({empty / len(source_clips):.2%})")

    linked, copied = 0, 0
    with ManifestBuilder(dataset_dir, verbose=verbose, total=len(source_clips)) as builder:
        for clip in source_clips:
            new_id = f"{name}:{clip.id.split(':', 1)[1]}"
            # The path the builder writes the clip to, so the link is the clip's
            # own file: the builder's verification write is in place on the same
            # file (the source is already 16 kHz mono PCM_16, as it writes).
            wav = dataset_dir / "audio" / file_name_for(new_id)
            if not wav.is_file():
                wav.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_dir / clip.audio, wav)
                    linked += 1
                except OSError:
                    shutil.copy2(source_dir / clip.audio, wav)
                    copied += 1
            text = texts[clip.id]
            builder.add(
                new_id,
                partial(load_audio, wav),
                language=clip.language,
                text=text,
                text_has_casing_punct=bool(text),
                group=f"{name}:{clip.group.split(':', 1)[1]}",
                group_kind=clip.group_kind,
                source_split=clip.source_split,
                license=clip.license,
                category=clip.category,
                source_url=clip.source_url,
                source_title=clip.source_title,
                source_author=clip.source_author,
                base_wer=clip.base_wer,
            )
        builder.write()
    if verbose:
        say(f"  {name}: {linked} WAVs linked, {copied} copied")
    return len(builder.clips)


def _qwen_answers(
    source_clips: list[Clip],
    language: str,
    cache: Path,
    source_dir: Path,
    *,
    name: str,
    verbose: bool,
) -> dict[str, str]:
    """The Qwen answer of every clip, from the label audit's cache or asked once."""
    answers: dict[str, str] = {}
    missing: list[Clip] = []
    for clip in source_clips:
        text = read_answer(
            answer_path(cache, LABEL_MODEL, clip.id),
            model=LABEL_MODEL,
            language=language,
            audio_sha256=clip.sha256,
            request={},
        )
        if text is None:
            missing.append(clip)
        else:
            answers[clip.id] = text
    if missing:
        if verbose:
            say(
                f"  {name}: {len(missing)} Qwen answers to ask, "
                f"{len(source_clips) - len(missing)} in cache"
            )
        transcribe_ = client(LABEL_MODEL)
        failing = threading.Event()
        try:
            one_pass = served(transcribe_, language).one_pass()
            map_requests(
                partial(_ask, cache, source_dir, language, one_pass.transcribe, failing),
                missing,
                one_pass.concurrency,
                should_stop=lambda f=failing: stop_requested() or f.is_set(),
                failing=failing,
            )
            raise_if_stopped(f"the Qwen pass of the {name} preparation stopped")
        finally:
            transcribe_.close()
        for clip in missing:
            text = read_answer(
                answer_path(cache, LABEL_MODEL, clip.id),
                model=LABEL_MODEL,
                language=language,
                audio_sha256=clip.sha256,
                request={},
            )
            if text is None:
                raise DownloadError(f"{clip.id}: the Qwen answer is missing after the pass")
            answers[clip.id] = text
    return answers


def _ask(
    cache: Path,
    source_dir: Path,
    language: str,
    transcribe_: Callable[[bytes], dict],
    failing: threading.Event,
    clip: Clip,
) -> None:
    """One clip's Qwen answer from the server, cached on the way."""
    if failing.is_set() or stop_requested():
        raise Stopped("the preparation stopped")
    wav = wav_bytes(load_audio(source_dir / clip.audio).samples)
    text = " ".join(str(transcribe_(wav)["text"]).split())
    write_answer(
        answer_path(cache, LABEL_MODEL, clip.id),
        model=LABEL_MODEL,
        language=language,
        audio_sha256=clip.sha256,
        request={},
        text=text,
    )
