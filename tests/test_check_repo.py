import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_repo.py"
spec = importlib.util.spec_from_file_location("check_repo", SCRIPT)
check_repo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_repo)

# Built at runtime so that this file does not trip the checks itself.
PRIVATE_IP = "10." + "1.2.3"
FAKE_TOKEN = "hf_" + "a" * 34


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "protected-paths.txt").write_text("# comment\ndocs/owner/\nLICENSE\n")
    return tmp_path


def commit(repo: Path, path: str, message: str) -> None:
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(f"{message}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def test_clean_files_pass(repo):
    (repo / "notes.md").write_text("Nothing to see here.\n")
    assert check_repo.check_files(repo, ["notes.md"]) == []


def test_audio_file_is_rejected(repo):
    (repo / "voice.WAV").write_bytes(b"RIFF")
    assert "never belong" in check_repo.check_files(repo, ["voice.WAV"])[0]


def test_large_file_is_rejected(repo):
    (repo / "big.txt").write_text("x" * (check_repo.MAX_FILE_BYTES + 1))
    assert "larger than" in check_repo.check_files(repo, ["big.txt"])[0]


@pytest.mark.parametrize(
    "line, label",
    [(f"host = {PRIVATE_IP}", "private IP address"), (f"token={FAKE_TOKEN}", "Hugging Face")],
)
def test_secrets_and_private_ips_are_rejected(repo, line, label):
    (repo / "config.txt").write_text(f"ok\n{line}\n")
    errors = check_repo.check_files(repo, ["config.txt"])
    assert len(errors) == 1
    assert errors[0].startswith("config.txt:2:")
    assert label in errors[0]


def test_allow_marker_skips_a_line(repo):
    (repo / "doc.md").write_text(f"example {PRIVATE_IP}  check-repo: allow\n")
    assert check_repo.check_files(repo, ["doc.md"]) == []


def test_public_addresses_and_versions_pass(repo):
    (repo / "doc.md").write_text("Use 127.0.0.1 or 8.8.8.8; version 1.10.2.3 is fine.\n")
    assert check_repo.check_files(repo, ["doc.md"]) == []


def test_agents_budget(repo):
    assert check_repo.check_agents_budget(repo) == ["AGENTS.md is missing"]
    (repo / "AGENTS.md").write_text("word " * check_repo.AGENTS_MAX_WORDS)
    assert check_repo.check_agents_budget(repo) == []
    (repo / "AGENTS.md").write_text("word " * (check_repo.AGENTS_MAX_WORDS + 1))
    assert "limit" in check_repo.check_agents_budget(repo)[0]


def test_is_protected():
    prefixes = ["docs/owner/", "LICENSE"]
    assert check_repo.is_protected("docs/owner/directives.md", prefixes)
    assert check_repo.is_protected("LICENSE", prefixes)
    assert not check_repo.is_protected("LICENSE.md", prefixes)
    assert not check_repo.is_protected("docs/roadmap.md", prefixes)


def test_commits_without_protected_changes_pass(repo):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init\n\nOwner-Approved: owner asked for setup, 2026-11-07")
    commit(repo, "src/code.py", "add code")
    assert check_repo.check_commits(repo, "all") == []


def test_protected_change_without_trailer_fails(repo):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init\n\nOwner-Approved: owner asked for setup, 2026-11-07")
    commit(repo, "docs/owner/directives.md", "edit directives")
    errors = check_repo.check_commits(repo, "HEAD~1..HEAD")
    assert len(errors) == 1
    assert "docs/owner/directives.md" in errors[0]


def test_protected_change_with_trailer_passes(repo):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init\n\nOwner-Approved: owner asked for setup, 2026-11-07")
    commit(
        repo, "docs/owner/directives.md", "docs: add D-015\n\nOwner-Approved: owner asked, today"
    )
    assert check_repo.check_commits(repo, "all") == []


def test_editing_the_protected_list_is_always_protected(repo):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init\n\nOwner-Approved: owner asked for setup, 2026-11-07")
    # The list does not mention itself, and the edit even removes an entry.
    (repo / ".github" / "protected-paths.txt").write_text("docs/owner/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "loosen protections")
    errors = check_repo.check_commits(repo, "HEAD~1..HEAD")
    assert len(errors) == 1
    assert ".github/protected-paths.txt" in errors[0]


def test_repository_without_commits_passes(repo):
    assert check_repo.check_commits(repo, "all") == []
