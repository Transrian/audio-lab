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
  counted and reported. The linked file is never written: a write to a hard
  link lands on the source's file, on the same inode, and a re-encoding is not
  the same bytes anyway (the int16 to float32 round trip moves a sample near
  full scale by one). The clip keeps the source's ``sha256`` and
  ``duration_s``: they describe the linked bytes exactly. Nothing is decoded,
  so the preparation has no partial state to resume — a stopped run is re-run,
  and a link that is already there is kept;
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
from audiolab.data.build import file_name_for
from audiolab.data.download import DownloadError
from audiolab.data.label_audit import answer_path, read_answer, write_answer
from audiolab.data.store import Clip, read_manifest, write_manifest
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

    linked, copied, kept = 0, 0, 0
    clips: list[Clip] = []
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for clip in source_clips:
        new_id = f"{name}:{clip.id.split(':', 1)[1]}"
        wav = audio_dir / file_name_for(new_id)
        state = _link_clip(source_dir / clip.audio, wav)
        if state == "linked":
            linked += 1
        elif state == "copied":
            copied += 1
        else:
            kept += 1
        text = texts[clip.id]
        clips.append(
            Clip(
                id=new_id,
                audio=str(Path("audio") / file_name_for(new_id)),
                sha256=clip.sha256,
                duration_s=clip.duration_s,
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
        )
        if verbose and len(clips) % 500 == 0:
            say(f"  ... {len(clips)} clips prepared")
        raise_if_stopped(f"preparation of {dataset_dir.name} stopped")
    write_manifest(dataset_dir, clips)
    if verbose:
        say(f"  {name}: {linked} WAVs linked, {copied} copied, {kept} already linked")
    return len(clips)


def _link_clip(source: Path, target: Path) -> str:
    """One clip's audio in the new set: the source's file, not its bytes.

    A hard link makes the two names one file; only a filesystem that refuses
    the link gets a copy. A target that already is the source's file is kept
    as it is — writing through a link would alter the source, on the same
    inode — and a target that is another file is replaced by the link.
    """
    if target.is_file() or target.is_symlink():
        if os.path.samefile(target, source):
            return "kept"
        target.unlink()
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


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
