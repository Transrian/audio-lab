import hashlib
import json
import sys
import tarfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import audio_cell, write_parquet, write_tone  # noqa: E402

from audiolab.core.stop import Stopped, request_stop  # noqa: E402
from audiolab.data import download as download_module  # noqa: E402
from audiolab.data.download import (  # noqa: E402
    DownloadError,
    audio_from_archive,
    download,
    git_blob_sha1_file,
    huggingface_entries,
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


def test_download_leaves_nothing_behind_when_the_checksum_does_not_match(tmp_path):
    """A file that failed its checksum must not be in the cache for the next call.

    Keeping it under its final name would make the next preparation take it as a
    finished download and prepare a corpus the pin says changed.
    """
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"another version")
    good = hashlib.sha256(b"the pinned version").hexdigest()
    destination = tmp_path / "cache" / "source.tar.gz"

    with pytest.raises(DownloadError, match="does not have the expected sha256"):
        download(str(source), destination, sha256=good)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []

    with pytest.raises(DownloadError, match="does not have the expected sha256"):
        download(str(source), destination, sha256=good)


def test_download_fetches_again_when_the_fetched_checksum_did_not_match(tmp_path):
    """The fetched path leaves nothing behind either, so the next call re-fetches."""
    attempts = []

    def respond(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url)
        return httpx.Response(200, content=b"a corpus")

    destination = tmp_path / "cache" / "source.zip"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(DownloadError, match="does not have the expected md5"):
            download(
                "https://example.invalid/source.zip",
                destination,
                client=client,
                md5=hashlib.md5(b"another corpus").hexdigest(),
            )
        assert not destination.exists()
        with pytest.raises(DownloadError, match="does not have the expected md5"):
            download(
                "https://example.invalid/source.zip",
                destination,
                client=client,
                md5=hashlib.md5(b"another corpus").hexdigest(),
            )
    assert len(attempts) == 2


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


def test_huggingface_entries_keeps_the_path_of_every_file_of_a_directory():
    """A listing of many meetings is one answer with the files of all of them."""

    def respond(request: httpx.Request) -> httpx.Response:
        # The listing is asked for recursively: a directory of a corpus is not
        # the files of one meeting but the files of every meeting under it.
        assert request.url.params.get("recursive") == "true"
        return httpx.Response(
            200,
            json=[
                {"type": "directory", "path": "data/one"},
                {"type": "file", "path": "data/one/a.wav", "lfs": {"oid": "a" * 64}},
                {"type": "file", "path": "data/two/b.wav", "oid": "b" * 40},
            ],
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        entries = huggingface_entries("microsoft/NOTSOFAR", "abc", "data", client=client)
    assert [entry["path"] for entry in entries] == [
        "data/one",
        "data/one/a.wav",
        "data/two/b.wav",
    ]


def test_git_blob_sha1_is_the_digest_of_the_object_stored_not_only_of_the_bytes(tmp_path):
    """A Hub names a file it keeps as a git object by the sha1 of ``blob <size>\0…``."""
    path = tmp_path / "gt_transcription.json"
    path.write_text("[]", encoding="utf-8")

    digest = git_blob_sha1_file(path)
    assert digest == hashlib.sha1(b"blob 2\0[]").hexdigest()
    # A file that grew by a byte has another digest, which the bytes alone miss.
    path.write_text("[] ", encoding="utf-8")
    assert git_blob_sha1_file(path) != digest


def test_download_refuses_a_file_whose_git_blob_digest_is_not_the_pinned_one(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("[]", encoding="utf-8")
    destination = tmp_path / "cache" / "source.json"

    download(str(source), destination, git_blob_sha1=git_blob_sha1_file(source))
    assert destination.read_text(encoding="utf-8") == "[]"

    other = tmp_path / "other.json"
    other.write_text("[] ", encoding="utf-8")
    with pytest.raises(DownloadError, match="expected git blob sha1"):
        download(str(other), tmp_path / "cache" / "other.json", git_blob_sha1="0" * 40)


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


def ranged_server(content: bytes, *, honours_range: bool = True):
    """A mock server for one file that answers range requests like a file server does."""
    asked = []

    def respond(request: httpx.Request) -> httpx.Response:
        wanted = request.headers.get("range", "")
        asked.append(wanted)
        if not wanted or not honours_range:
            return httpx.Response(200, content=content)
        first = int(wanted.removeprefix("bytes=").rstrip("-"))
        if first >= len(content):
            return httpx.Response(416, headers={"content-range": f"bytes */{len(content)}"})
        return httpx.Response(
            206,
            content=content[first:],
            headers={"content-range": f"bytes {first}-{len(content) - 1}/{len(content)}"},
        )

    return httpx.MockTransport(respond), asked


def test_a_stopped_download_keeps_its_part_and_the_next_call_fetches_only_the_rest(tmp_path):
    content = bytes(range(256)) * 16_384  # 4 MiB: four blocks
    transport, asked = ranged_server(content)
    destination = tmp_path / "cache" / "archive.tar.gz"
    part = destination.with_name("archive.tar.gz.part")
    pinned = hashlib.sha256(content).hexdigest()

    part.parent.mkdir(parents=True)
    part.write_bytes(content[: 3 << 20])  # what a stop three blocks in leaves
    with httpx.Client(transport=transport) as client:
        download("https://example.invalid/a.tar.gz", destination, client=client, sha256=pinned)
    assert asked == [f"bytes={3 << 20}-"]
    assert destination.read_bytes() == content
    assert not part.exists()


def test_a_download_asked_to_stop_before_it_starts_sends_nothing(tmp_path):
    transport, asked = ranged_server(b"a corpus")
    request_stop()
    with httpx.Client(transport=transport) as client:
        with pytest.raises(Stopped):
            download("https://example.invalid/c", tmp_path / "corpus.zip", client=client)
    assert asked == []


def test_a_download_stops_between_two_blocks_and_keeps_them(tmp_path, monkeypatch):
    content = b"x" * (3 << 20)
    transport, _ = ranged_server(content)
    destination = tmp_path / "archive.tar.gz"
    blocks = []

    def stop_after_the_first_block(what="stopped"):
        blocks.append(what)
        if len(blocks) == 2:  # the first call is the check before the download starts
            raise Stopped(what)

    monkeypatch.setattr(download_module, "raise_if_stopped", stop_after_the_first_block)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(Stopped, match="download of archive.tar.gz stopped"):
            download("https://example.invalid/a", destination, client=client, sha256="0" * 64)
    assert destination.with_name("archive.tar.gz.part").stat().st_size == 1 << 20
    assert not destination.exists()


def test_a_server_that_ignores_the_range_sends_the_whole_file_again(tmp_path):
    content = b"the whole corpus"
    transport, asked = ranged_server(content, honours_range=False)
    destination = tmp_path / "corpus.zip"
    destination.with_name("corpus.zip.part").write_bytes(b"the who")
    with httpx.Client(transport=transport) as client:
        download("https://example.invalid/c", destination, client=client, md5=md5(content))
    assert asked == ["bytes=7-"]
    assert destination.read_bytes() == content


def test_a_part_longer_than_the_file_is_dropped_and_fetched_again(tmp_path):
    content = b"a shorter file now"
    transport, asked = ranged_server(content)
    destination = tmp_path / "corpus.zip"
    destination.with_name("corpus.zip.part").write_bytes(b"x" * 100)
    with httpx.Client(transport=transport) as client:
        download("https://example.invalid/c", destination, client=client, md5=md5(content))
    assert asked == ["bytes=100-", ""]
    assert destination.read_bytes() == content


def test_a_part_that_is_the_whole_file_is_kept_and_checked(tmp_path):
    """A stop after the last block, or a kill before the rename, leaves the whole file."""
    content = b"the whole corpus"
    transport, asked = ranged_server(content)
    destination = tmp_path / "corpus.zip"
    destination.with_name("corpus.zip.part").write_bytes(content)
    with httpx.Client(transport=transport) as client:
        download("https://example.invalid/c", destination, client=client, md5=md5(content))
    assert asked == ["bytes=16-"], "nothing is fetched again"
    assert destination.read_bytes() == content

    other = tmp_path / "other.zip"
    other.with_name("other.zip.part").write_bytes(b"x" * len(content))  # as long, not the file
    with httpx.Client(transport=transport) as client, pytest.raises(DownloadError):
        download("https://example.invalid/c", other, client=client, md5=md5(content))
    assert not other.with_name("other.zip.part").exists(), "the checksum drops it"


def test_an_unpinned_download_never_continues_a_part(tmp_path):
    """Without a checksum nothing proves that the part and the rest are one file."""
    content = b"a talk"
    transport, asked = ranged_server(content)
    destination = tmp_path / "talk.mp4"
    destination.with_name("talk.mp4.part").write_bytes(b"an older ta")
    with httpx.Client(transport=transport) as client:
        download("https://example.invalid/t", destination, client=client)
    assert asked == [""]
    assert destination.read_bytes() == content


def md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()
