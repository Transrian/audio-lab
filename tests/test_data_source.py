import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import write_tone  # noqa: E402

from audiolab.data.download import DownloadError, audio_from_archive, download, open_archive  # noqa: E402
from audiolab.data.textfiles import read_records, read_table  # noqa: E402


def make_archive(path: Path, members: dict[str, Path]) -> Path:
    """A tar.gz holding the given files, built the way a dataset archive is organised."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as handle:
        for name, source in members.items():
            handle.add(str(source), arcname=name)
    return path


def test_download_copies_a_local_source_and_is_idempotent(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    destination = tmp_path / "cache" / "nested" / "source.txt"
    assert download(str(source), destination).read_text(encoding="utf-8") == "hello"

    destination.write_text("already here", encoding="utf-8")
    download(str(source), destination)
    assert destination.read_text(encoding="utf-8") == "already here"


def test_download_raises_through_a_usable_error(tmp_path):
    with pytest.raises(DownloadError, match="cannot download"):
        download("http://127.0.0.1:1/nothing.wav", tmp_path / "nothing.wav")


def test_audio_from_archive_reads_a_wav_member(tmp_path):
    tone = write_tone(tmp_path / "clip.wav", seconds=0.5)
    archive = make_archive(tmp_path / "audio.tar.gz", {"audio/clip.wav": tone})
    audio = None
    with open_archive(archive) as handle:
        for member in handle:  # a streaming archive is read once, in order
            audio = audio_from_archive(handle, member)
    assert audio is not None
    assert audio.sample_rate == 16_000
    assert audio.duration_s == pytest.approx(0.5, abs=1e-3)


def test_read_table_uses_the_header(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text(
        "id\tfilename\ttranscription\n1\ta\tBonjour.\n2\tb\tAu revoir.\n", encoding="utf-8"
    )
    rows = read_table(path, ("id", "filename", "transcription"), key="filename")
    assert rows["a"]["transcription"] == "Bonjour."
    assert rows["b"]["id"] == "2"


def test_read_table_refuses_a_changed_header(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text("id\taudio\ttext\n1\ta\thi\n", encoding="utf-8")
    with pytest.raises(DownloadError, match="no column"):
        read_table(path, ("filename", "transcription"), key="filename")


def test_read_records_keeps_the_rest_of_the_line_as_value(tmp_path):
    path = tmp_path / "segments.txt"
    path.write_text("seg1 1234 5678\nseg2 42 99\n\n", encoding="utf-8")
    assert read_records(path, 3) == {"seg1": "1234 5678", "seg2": "42 99"}


def test_read_records_of_transcripts_keeps_spaces(tmp_path):
    path = tmp_path / "transcripts.txt"
    path.write_text("a text with spaces\nb another one\n", encoding="utf-8")
    assert read_records(path, 2) == {"a": "text with spaces", "b": "another one"}


def test_read_records_refuses_an_unexpected_line(tmp_path):
    path = tmp_path / "segments.txt"
    path.write_text("only-one-field\n", encoding="utf-8")
    with pytest.raises(DownloadError, match="unexpected line"):
        read_records(path, 3)
