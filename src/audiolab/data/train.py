"""Turning the train split of a source into a capped set of clips.

A training source is prepared like an evaluation set — 16 kHz mono WAV files
plus a manifest under `$AUDIOLAB_DATA/<name>` — with two differences:

- only part of the source is taken. Every source is prepared with the same
  number of clips, so a screening can compare what each source brings at the
  same number of training updates; the source is read in its own stable order
  and the first clips that pass the filters are kept, which needs nothing more
  than the rows already read and gives the same set on every machine;
- no panel is built. Panels are the frozen id lists a model is scored on
  (decision 0002), and a training source is not scored on.

Two filters, chosen for the screening (not owner directives): a clip must
carry text (a clip without a reference teaches nothing), and it must last 0.5
to 20 seconds, the length of a dictated sentence and what the training recipe
batches comfortably.
Nothing here decides what a source is worth — it only makes the prepared set
comparable between sources.

Every label is written in the characters the base model can write
(:func:`plain_label`), and a clip whose label still holds another one is left
out: the tokenizer of Parakeet TDT 0.6B v3 has no piece for ``’`` or ``"`` and
no byte fallback, so such a character becomes ``<unk>``, and a model trained on
it learns to write ``<unk>`` ("I don ⁇ t know").
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from audiolab.core.audio import Audio
from audiolab.core.paths import data_dir
from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError
from audiolab.data.store import MANIFEST_NAME, Clip, read_manifest

# The most clips one training source is prepared with. The screening fine-tunes
# once per source with the same number of updates, so this is the size every
# source is cut to; a source with fewer usable clips is prepared as it is and
# says so.
TRAIN_CLIP_CAP = 12_000
# The cap is spread over at least this many groups. Several corpora publish
# their train split one speaker after another, so taking the first 12 000 clips
# in the published order spends the whole cap on a handful of readers (MLS: 19
# speakers); no more than this share of the cap comes from one group, and the
# preparation reads on until it has covered enough groups.
MIN_GROUPS = 60
# What one group contributes at most: the cap spread over at least MIN_GROUPS
# groups. A fetcher passes it explicitly, so a test can lower it.
CLIP_CAP_PER_GROUP = TRAIN_CLIP_CAP // MIN_GROUPS
MIN_SECONDS = 0.5
MAX_SECONDS = 20.0
# How far past the end of a recording an annotation may reach before the
# preparation refuses it: a segment that ends slightly after the file is a
# rounding away from its end, one that ends well past it means the file is not
# the recording the annotation describes.
END_TOLERANCE_S = 1.0
# The characters of the base model's vocabulary: those of the pieces of the
# tokenizer.vocab inside parakeet-tdt-0.6b-v3.nemo, without "▁" and the special
# tokens. Any other character is <unk> to the tokenizer ("…" too, except that its
# normalization reads it as "...", which plain_label writes anyway).
VOCAB_CHARACTERS = frozenset(
    string.ascii_letters
    + string.digits
    + "!$%',-./:?¡£¿€"
    + "ÀÁÂÃÄÅÆÇÈÉÊÌÍÎÓÔÕÖØÚÜÝßàáâãäåæçèéêëìíîïñòóôõöøùúûüý"
    + "ĀāĂăĄąĆćĊċČčĎďđĒēĖėĘęĚěĠġĢģĦħĪīĮįĶķĺĻļĽľŁłŃńŅņŇňŐőŒœŕŘřŚśŠšŤťŪūŮůŰűŲųŹźŻżŽžȘșȚț"
    + "ΆΈΉΌΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩάέήίαβγδεζηθικλμνξοπρστυφχψωϊόύώ"
    + "ЄІЇАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяєіїҐґ"
)
# What a label writes for punctuation the vocabulary has no piece for: the plain
# apostrophe for the curly ones, nothing for quotation marks, parentheses and
# brackets (nobody says them), a comma for a semicolon, a space for an em dash (a
# break between words, as the scorer reads it), a hyphen for an en dash
# ("2014–2020"), and three points for an ellipsis.
PLAIN_PUNCTUATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": None,
        "”": None,
        '"': None,
        "«": None,
        "»": None,
        "(": None,
        ")": None,
        "[": None,
        "]": None,
        ";": ",",
        "—": " ",
        "–": "-",
        "…": "...",
    }
)
# A dash between spaces ("word – word") is a break, which the model writes as nothing.
LONE_DASH = re.compile(r"(?<!\S)-+(?!\S)")


def plain_label(text: str) -> str:
    """A label in the punctuation of the base model's vocabulary, one space between words.

    French sets a semicolon off with a space ("mot ; mot"), which a comma never
    is in either language, so the comma that replaces it closes up.

    >>> plain_label("I don’t know — “maybe” – later…")
    "I don't know maybe later..."
    """
    words = LONE_DASH.sub(" ", text.translate(PLAIN_PUNCTUATION)).split()
    return " ".join(words).replace(" ,", ",")


def unwritable(label: str) -> str:
    """The characters of a label the base model cannot write, in the order they come."""
    return "".join(dict.fromkeys(c for c in label if c != " " and c not in VOCAB_CHARACTERS))


@dataclass
class Take:
    """Collects clips until the cap, counting why the other rows were left out.

    The counts are the report of a preparation: they say how much of the source
    was needed, and whether the cap was reached at all. A fetcher stops reading
    its source as soon as :meth:`full` is true, so `rows_read` is the number of
    rows it looked at, not the size of the whole split.
    """

    name: str
    builder: ManifestBuilder
    language: str
    source_split: str
    license: str
    text_has_casing_punct: bool
    group_kind: str = "speaker"
    excluded: frozenset[str] = field(default_factory=frozenset)
    cap: int = TRAIN_CLIP_CAP
    cap_per_group: int = CLIP_CAP_PER_GROUP
    kept: int = 0
    rows_read: int = 0
    too_short: int = 0
    too_long: int = 0
    without_text: int = 0
    unwritable_text: int = 0
    in_evaluation_set: int = 0
    beyond_group_cap: int = 0
    per_group: dict[str, int] = field(default_factory=dict)

    def full(self) -> bool:
        """Whether the cap is reached and the source can stop being read."""
        return self.kept >= self.cap

    def group_full(self, group: str) -> bool:
        """Whether one group already holds its share of the cap.

        A fetcher asks this before reading a whole archive or shard, so that a
        source published speaker by speaker is not downloaded for a reader that
        is already fully represented.
        """
        return self.per_group.get(group, 0) >= self.cap_per_group

    def decide(self, *, text: str, group: str, seconds: float) -> str | None:
        """Count one row against the filters of the cap, and say why it is not a clip.

        ``group`` is the id the source itself uses for what the row belongs to
        (a speaker, a recording), without the dataset name: it is what the
        evaluation sets are checked against. ``None`` means the row is a clip:
        the clip and its group are counted for it, and the caller writes its
        audio with :meth:`write_clip` without deciding anything again. The
        filters live here alone, so a preparation that reads its rows twice — the
        annotation first, then the audio — decides once and counts once.
        """
        self.rows_read += 1
        label = plain_label(text)
        if not label:
            self.without_text += 1
            return "without text"
        if unwritable(label):
            self.unwritable_text += 1
            return "with a character the base model cannot write"
        if group in self.excluded:
            self.in_evaluation_set += 1
            return "in an evaluation set"
        if self.group_full(group):
            self.beyond_group_cap += 1
            return "beyond the share of its group"
        if seconds < MIN_SECONDS:
            self.too_short += 1
            return f"shorter than {MIN_SECONDS} s"
        if seconds > MAX_SECONDS:
            self.too_long += 1
            return f"longer than {MAX_SECONDS:g} s"
        self.kept += 1
        self.per_group[group] = self.per_group.get(group, 0) + 1
        return None

    def write_clip(
        self,
        source_id: str,
        audio: Audio | Callable[[], Audio],
        *,
        text: str,
        group: str,
        **extra_fields,
    ) -> None:
        """Write the clip of a row :meth:`decide` took, its label as :func:`plain_label` writes it.

        ``extra_fields`` are manifest fields only some sources fill, such as the
        attribution a licence requires to travel with the clip, or a license
        that is not the same for every row of the source. ``audio`` can be a
        function that reads it, called only if the clip is not prepared yet.
        """
        clip_id, fields = self._clip(source_id, text, group, extra_fields)
        self.builder.add(clip_id, audio, **fields)

    def prepared(self, source_id: str, *, text: str, group: str, **extra_fields) -> Clip | None:
        """The clip of this row a stopped preparation already wrote, if any."""
        clip_id, fields = self._clip(source_id, text, group, extra_fields)
        return self.builder.prepared(clip_id, **fields)

    def _clip(self, source_id: str, text: str, group: str, extra_fields: dict) -> tuple:
        """The id and the manifest fields of the clip of one row."""
        fields = {
            "language": self.language,
            "text": plain_label(text),
            "text_has_casing_punct": self.text_has_casing_punct,
            "group": f"{self.name}:{group}",
            "group_kind": self.group_kind,
            "source_split": self.source_split,
            "license": self.license,
            **extra_fields,
        }
        return f"{self.name}:{source_id}", fields

    def offer(
        self,
        source_id: str,
        audio: Audio | Callable[[], Audio],
        *,
        text: str,
        group: str,
        **extra_fields,
    ) -> bool:
        """Offer one row of a source whose audio is on hand; it becomes a clip if it passes.

        Whose rows are read as the source is read, one after another, this is the
        whole path: decide, and write when the decision takes the row. ``audio``
        can be a function that reads it: a row a stopped preparation already
        wrote is then decided on the length it was written with, without being
        read again.
        """
        if self.full():
            return False
        done = self.prepared(source_id, text=text, group=group, **extra_fields)
        if done is None and callable(audio):
            audio = audio()
        seconds = done.duration_s if done else audio.duration_s
        if self.decide(text=text, group=group, seconds=seconds) is not None:
            return False
        self.write_clip(source_id, audio, text=text, group=group, **extra_fields)
        return True

    def report(self) -> str:
        """One line saying what was taken and what was left out."""
        state = "cap reached" if self.full() else f"below the cap of {self.cap}"
        return (
            f"{self.name}: {self.kept} clips ({state}), {self.rows_read} rows read; left out: "
            f"{self.too_short} shorter than {MIN_SECONDS} s, {self.too_long} longer than "
            f"{MAX_SECONDS:g} s, {self.without_text} without text, "
            f"{self.unwritable_text} with a character the base model cannot write, "
            f"{self.in_evaluation_set} already in an evaluation set, "
            f"{self.beyond_group_cap} beyond the {self.cap_per_group}-clip share of a group"
        )


def evaluation_groups(*datasets: str, root: Path | None = None) -> frozenset[str]:
    """The source-side group ids of prepared evaluation datasets.

    Nothing a model is scored on may be trained on (decision 0002), so a
    training source leaves out every row whose group appears in an evaluation
    set. A group in a manifest is ``<dataset>:<source group>`` and the part
    after the first colon is the id the source itself uses — a speaker for MLS
    and LibriSpeech, a sentence for FLEURS — which is what this returns.

    An evaluation dataset that is not prepared on this machine contributes
    nothing: it is a set of clips, not a list of ids, and the repository holds
    no audio. Which evaluation sets were available is part of what a
    preparation reports.
    """
    root = root or data_dir()
    groups: set[str] = set()
    for dataset in datasets:
        directory = root / dataset
        if not (directory / MANIFEST_NAME).is_file():
            continue
        groups.update(clip.group.partition(":")[2] for clip in read_manifest(directory))
    return frozenset(groups)


class Plannable(Protocol):
    """What the clip machinery reads of one row of an annotation."""

    id: str
    text: str
    start: float
    end: float
    speaker: str


def plan_clips[Row: Plannable](take: Take, rows: Iterable[Row]) -> list[Row]:
    """The rows of an annotation that become clips, counted as they are decided.

    A source that serves one file per microphone must decide with the annotation
    alone which rows it will read, so that it downloads only what it keeps; and
    the decision is :meth:`Take.decide`, the one a source whose audio is already
    on hand makes, counted on the same :class:`Take`. The rows this returns are
    written afterwards by :func:`write_clips` and are not decided again.
    """
    planned: list[Row] = []
    for row in rows:
        if take.full():
            break
        if take.decide(text=row.text, group=row.speaker, seconds=row.end - row.start) is None:
            planned.append(row)
    return planned


def write_clips[Row: Plannable](
    take: Take,
    recording: str,
    audio: Audio | Callable[[], Audio],
    rows: Iterable[Row],
    *,
    tolerance_s: float = END_TOLERANCE_S,
) -> None:
    """Write the clips one recording holds, from the rows a plan took on it.

    ``audio`` is the whole recording — a channel of a meeting, the close-talking
    recording of one participant — and every row is a span of it, already decided
    and counted by :func:`plan_clips`. A row that reaches past the end of the
    recording is refused rather than clamped: times that do not describe this
    file mean the wrong file was read, and clamping would train on the wrong
    speech under the right words. ``audio`` can be a function that reads the
    recording, called only if one of its clips is not prepared yet.
    """
    rows = list(rows)
    if callable(audio):
        if all(take.prepared(row.id, text=row.text, group=row.speaker) for row in rows):
            for row in rows:
                take.write_clip(row.id, audio, text=row.text, group=row.speaker)
            return
        audio = audio()
    for row in rows:
        if row.end > audio.duration_s + tolerance_s:
            raise DownloadError(
                f"{row.id} ends at {row.end:.2f} s but {recording} is {audio.duration_s:.2f} s "
                "long: the annotation is not of this recording"
            )
        take.write_clip(row.id, audio.cut(row.start, row.end), text=row.text, group=row.speaker)
