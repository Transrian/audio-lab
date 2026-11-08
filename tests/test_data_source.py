import hashlib
import json
import sys
import tarfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import audio_cell, write_parquet, write_tone  # noqa: E402

from audiolab.data.download import (  # noqa: E402
    DownloadError,
    audio_from_archive,
    download,
    huggingface_sha256_map,
    open_archive,
    sha256_file,
)
from audiolab.data.parquet import audio_of, rows  # noqa: E402
from audiolab.data.textfiles import read_fields, read_records  # noqa: E402


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


def test_read_fields_reads_a_headerless_table_by_position(tmp_path):
    """FLEURS tables have no header, so positions are the only thing to go by."""
    path = tmp_path / "test.tsv"
    path.write_text(
        "1\ta.wav\tBonjour, monde.\tnormalized\tx\t3.0\tFEMALE\n"
        "2\tb.wav\tAu revoir.\tnormalized\tx\t2.0\tMALE\n",
        encoding="utf-8",
    )
    rows = read_fields(path, {0: "id", 2: "transcription"}, key_index=1, min_fields=3)
    assert rows["a.wav"]["transcription"] == "Bonjour, monde."
    assert rows["b.wav"]["id"] == "2"


def test_read_fields_stops_on_a_short_line(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text("1\ta.wav\n", encoding="utf-8")
    with pytest.raises(DownloadError, match="unexpected line"):
        read_fields(path, {0: "id", 2: "transcription"}, key_index=1, min_fields=3)


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


def test_sha256_file_hashes_the_bytes_on_disk(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"audio-lab")
    assert sha256_file(path) == hashlib.sha256(b"audio-lab").hexdigest()


def test_download_checks_the_recorded_sha256_of_a_fresh_file(tmp_path):
    """A source that changed under a pinned dataset must stop preparation."""
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"first version")
    good = hashlib.sha256(b"first version").hexdigest()
    destination = tmp_path / "cache" / "source.tar.gz"

    assert download(str(source), destination, sha256=good) == destination

    source.write_bytes(b"second version")
    destination.unlink()
    with pytest.raises(DownloadError, match="does not have the expected sha256"):
        download(str(source), destination, sha256=good)


def test_download_checks_a_recorded_md5_when_that_is_what_is_published(tmp_path):
    """Zenodo publishes md5 checksums, and one of the sources is pinned by it."""
    source = tmp_path / "source.zip"
    source.write_bytes(b"a corpus")
    good = hashlib.md5(b"a corpus").hexdigest()
    destination = tmp_path / "cache" / "source.zip"

    assert download(str(source), destination, md5=good) == destination

    source.write_bytes(b"another corpus")
    destination.unlink()
    with pytest.raises(DownloadError, match="does not have the expected md5"):
        download(str(source), destination, md5=good)


def test_download_checks_the_checksum_of_a_file_it_fetched_over_http(tmp_path):
    """The fetch path has to check too: a copied file and a fetched one are two paths."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"a corpus")

    destination = tmp_path / "cache" / "source.zip"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert (
            download(
                "https://example.invalid/source.zip",
                destination,
                client=client,
                md5=hashlib.md5(b"a corpus").hexdigest(),
            )
            == destination
        )

    destination.unlink()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(DownloadError, match="does not have the expected sha256"):
            download(
                "https://example.invalid/source.zip",
                destination,
                client=client,
                sha256="0" * 64,
            )


def test_huggingface_sha256_map_reads_the_hub_file_listing():
    """The Hub publishes the sha256 of every large file at a pinned revision."""

    def respond(request: httpx.Request) -> httpx.Response:
        assert (
            request.url.path == "/api/datasets/facebook/multilingual_librispeech/tree/abc123/data/x"
        )
        return httpx.Response(
            200,
            json=[
                {"path": "data/x/one.tar.gz", "lfs": {"oid": "a" * 64}},
                {"path": "data/x/two.tar.gz", "lfs": {"oid": "b" * 64}},
            ],
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        hashes = huggingface_sha256_map(
            "facebook/multilingual_librispeech", "abc123", "data/x", client=client
        )
    assert hashes == {"one.tar.gz": "a" * 64, "two.tar.gz": "b" * 64}


def test_huggingface_sha256_map_reports_an_unusable_answer():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"error": "no"}).encode())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(DownloadError, match="did not answer with a file list"):
            huggingface_sha256_map(
                "facebook/multilingual_librispeech", "abc", "data/x", client=client
            )


def test_huggingface_sha256_map_follows_a_long_listing_page_by_page():
    """One answer holds a thousand files: the rest is behind a Link header."""

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("cursor") == "page2":
            return httpx.Response(
                200, json=[{"path": "data/x/two.tar.gz", "lfs": {"oid": "b" * 64}}]
            )
        return httpx.Response(
            200,
            headers={"link": '<https://hub.example/tree/abc?cursor=page2>; rel="next"'},
            json=[{"path": "data/x/one.tar.gz", "lfs": {"oid": "a" * 64}}],
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        hashes = huggingface_sha256_map("byan/cs-fleurs", "abc", "data/x", client=client)
    assert hashes == {"one.tar.gz": "a" * 64, "two.tar.gz": "b" * 64}


def test_parquet_rows_come_back_one_dictionary_per_row(tmp_path):
    tone = write_tone(tmp_path / "tone.wav", seconds=0.2)
    path = write_parquet(
        tmp_path / "test.parquet",
        {"ID": ["a", "b"], "audio": [audio_cell(tone), audio_cell(tone, "other.wav")]},
    )
    read = list(rows(path, ["ID", "audio"]))
    assert [row["ID"] for row in read] == ["a", "b"]
    assert read[0]["audio"]["path"] == "clip.wav"


def test_parquet_audio_is_decoded_like_any_other_file(tmp_path):
    """The Hub stores the encoded file in the row, not samples."""
    tone = write_tone(tmp_path / "tone.wav", seconds=0.5)
    path = write_parquet(
        tmp_path / "test.parquet", {"audio": [audio_cell(tone)], "accent": ["en_fr"]}
    )
    row = next(iter(rows(path, ["audio", "accent"])))
    audio = audio_of(row["audio"])
    assert audio.sample_rate == 16_000
    assert audio.duration_s == pytest.approx(0.5, abs=1e-3)


def test_parquet_audio_refuses_a_row_without_audio():
    with pytest.raises(DownloadError, match="holds no audio"):
        audio_of({"bytes": b"", "path": "clip.wav"})


def test_parquet_rows_report_a_file_that_is_not_parquet(tmp_path):
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises(DownloadError, match="cannot read"):
        list(rows(path, ["audio"]))
