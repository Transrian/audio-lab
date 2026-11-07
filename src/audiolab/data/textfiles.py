"""Text files that come with a dataset.

FLEURS ships TSV tables, MLS and LibriSpeech ship space- or tab-separated
records. All three are read here, by column name rather than by position, so
that a reordered upstream file fails loudly instead of silently mixing up
transcriptions.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.data.download import DownloadError


def read_table(path: Path, columns: tuple[str, ...], key: str) -> dict[str, dict[str, str]]:
    """Read a delimited text file with a header line, indexed by one column.

    Fields are split on tabs when the header contains a tab, else on whitespace.
    A table that no longer has the expected columns stops preparation instead of
    silently mixing up transcriptions.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise DownloadError(f"{path.name} is empty")
    separator = "\t" if "\t" in lines[0] else None
    header = lines[0].split(separator)
    missing = [name for name in (*columns, key) if name not in header]
    if missing:
        raise DownloadError(f"{path.name} has no column {missing}; header is {header}")
    rows = (dict(zip(header, line.split(separator), strict=False)) for line in lines[1:] if line)
    return {row[key]: row for row in rows}


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
