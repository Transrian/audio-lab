"""Text files that come with a dataset.

FLEURS ships a TSV table with no header, MLS and LibriSpeech ship space- or
tab-separated records. Each format has one reader here, and both stop on a file
that does not look the way preparation expects, instead of building clips with
the wrong text.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.data.download import DownloadError


def read_fields(
    path: Path, wanted: dict[int, str], key_index: int, min_fields: int
) -> dict[str, dict[str, str]]:
    """Read a headerless tab-separated file, keeping fields by position.

    FLEURS ships its tables without a header line, so positions are the only
    thing to go by. A line with too few fields stops preparation rather than
    producing a clip with the wrong text.
    """
    records: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < min_fields:
            raise DownloadError(f"{path.name} has an unexpected line: {line[:60]!r}")
        records[fields[key_index]] = {name: fields[index] for index, name in wanted.items()}
    return records


def read_records(path: Path, max_fields: int = 2) -> dict[str, str]:
    """Read ``<key> <value>`` lines into a dictionary.

    Only the first field is split off, so a value may contain separators. MLS
    transcript lines are free text, and MLS segment lines are not; both are read
    here with the check that no line is shorter than expected.
    """
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < max_fields:
            raise DownloadError(f"{path.name} has an unexpected line: {line[:60]!r}")
        records[fields[0]] = line.strip().split(maxsplit=1)[1] if max_fields > 1 else ""
    return records
