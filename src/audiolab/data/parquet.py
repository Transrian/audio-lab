"""Reading a Parquet file whose rows carry the audio itself.

Some corpora are published as Parquet with one utterance per row and the
encoded audio in a column, next to the reference text. Reading one back is the
only thing this module does, so a preparation script states which columns it
wants and gets one dictionary per row.

A shard can be several gigabytes, so rows are decoded batch by batch: the whole
file in memory would work and would cost that much memory for no reason.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq

from audiolab.core.audio import Audio, load_audio
from audiolab.data.download import DownloadError

BATCH_ROWS = 64


def rows(path: Path, columns: list[str]) -> Iterator[dict]:
    """Yield one dictionary per row of ``path``, holding the named columns."""
    try:
        batches = pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns)
        for batch in batches:
            yield from batch.to_pylist()
    except (pyarrow.ArrowException, OSError) as failure:
        raise DownloadError(f"cannot read {path}: {failure}") from failure


def audio_of(cell: object) -> Audio:
    """The audio of one row, as 16 kHz mono float32.

    A Parquet column of audio holds the encoded file itself (WAV here, several
    sample rates), not samples, so it is decoded like any other file.
    """
    data = cell.get("bytes") if isinstance(cell, dict) else None
    if not data:
        raise DownloadError("a Parquet row holds no audio")
    return load_audio(io.BytesIO(data))
