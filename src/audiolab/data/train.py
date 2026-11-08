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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from audiolab.core.audio import Audio
from audiolab.core.paths import data_dir
from audiolab.data.build import ManifestBuilder
from audiolab.data.store import MANIFEST_NAME, read_manifest

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

    def offer(self, source_id: str, audio: Audio, *, text: str, group: str) -> bool:
        """Offer one row of the source; it becomes a clip when it passes the filters.

        ``group`` is the id the source itself uses for what the row belongs to
        (a speaker, a recording), without the dataset name: it is what the
        evaluation sets are checked against.
        """
        if self.full():
            return False
        self.rows_read += 1
        if not text.strip():
            self.without_text += 1
            return False
        if group in self.excluded:
            self.in_evaluation_set += 1
            return False
        if self.group_full(group):
            self.beyond_group_cap += 1
            return False
        if audio.duration_s < MIN_SECONDS:
            self.too_short += 1
            return False
        if audio.duration_s > MAX_SECONDS:
            self.too_long += 1
            return False
        self.builder.add(
            f"{self.name}:{source_id}",
            audio,
            language=self.language,
            text=text,
            text_has_casing_punct=self.text_has_casing_punct,
            group=f"{self.name}:{group}",
            group_kind=self.group_kind,
            source_split=self.source_split,
            license=self.license,
        )
        self.kept += 1
        self.per_group[group] = self.per_group.get(group, 0) + 1
        return True

    def report(self) -> str:
        """One line saying what was taken and what was left out."""
        state = "cap reached" if self.full() else f"below the cap of {self.cap}"
        return (
            f"{self.name}: {self.kept} clips ({state}), {self.rows_read} rows read; left out: "
            f"{self.too_short} shorter than {MIN_SECONDS} s, {self.too_long} longer than "
            f"{MAX_SECONDS:g} s, {self.without_text} without text, "
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
