"""Repository hygiene checks, run by CI on every push and by scripts/check.sh.

Checks:
1. no audio, model weight or oversized file is tracked;
2. no secret token or private IP address appears in a text file;
3. AGENTS.md stays under its word limit;
4. with --commits: every commit touching a protected path (listed in
   .github/protected-paths.txt) carries an "Owner-Approved:" trailer.

A line may opt out of check 2 by containing the marker "check-repo: allow".
Uses the standard library only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_SUFFIXES = {
    # audio
    ".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".webm",
    # model weights
    ".nemo", ".ckpt", ".pt", ".pth", ".safetensors", ".gguf", ".onnx", ".bin",
}  # fmt: skip
MAX_FILE_BYTES = 2 * 1024 * 1024
SIZE_EXEMPT = {"uv.lock"}
AGENTS_MAX_WORDS = 750
ALLOW_MARKER = "check-repo: allow"
TRAILER = re.compile(r"^Owner-Approved:\s*\S", re.MULTILINE)

SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|\bgithub_pat_\w{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private IP address": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
}


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def repo_files(root: Path = ROOT) -> list[str]:
    """Tracked files plus untracked files that are not ignored."""
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", cwd=root)
    return sorted({name for name in listed.splitlines() if (root / name).is_file()})


def check_files(root: Path, files: list[str]) -> list[str]:
    errors = []
    for name in files:
        path = root / name
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"{name}: audio or model files never belong in the repository")
            continue
        if path.stat().st_size > MAX_FILE_BYTES and name not in SIZE_EXEMPT:
            errors.append(f"{name}: larger than {MAX_FILE_BYTES // 1024 // 1024} MiB")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if ALLOW_MARKER in line:
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    errors.append(f"{name}:{number}: looks like a {label}")
    return errors


def check_agents_budget(root: Path) -> list[str]:
    agents = root / "AGENTS.md"
    if not agents.is_file():
        return ["AGENTS.md is missing"]
    words = len(agents.read_text(encoding="utf-8").split())
    if words > AGENTS_MAX_WORDS:
        return [f"AGENTS.md has {words} words; the limit is {AGENTS_MAX_WORDS}"]
    return []


PROTECTED_LIST = ".github/protected-paths.txt"


def protected_prefixes(root: Path) -> list[str]:
    """Paths listed in the protected list. The list itself is always protected."""
    lines = (root / PROTECTED_LIST).read_text(encoding="utf-8").splitlines()
    listed = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return [PROTECTED_LIST, *listed]


def is_protected(path: str, prefixes: list[str]) -> bool:
    return any(
        path.startswith(prefix) if prefix.endswith("/") else path == prefix for prefix in prefixes
    )


def check_commits(root: Path, commit_range: str) -> list[str]:
    """Check the Owner-Approved trailer on every commit of ``commit_range`` ("all" = HEAD)."""
    if subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=root,
                      capture_output=True).returncode != 0:  # fmt: skip
        return []  # no commit yet
    prefixes = protected_prefixes(root)
    revisions = git("rev-list", "HEAD" if commit_range == "all" else commit_range, cwd=root)
    errors = []
    for sha in revisions.split():
        changed = git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", sha, cwd=root)
        touched = [path for path in changed.splitlines() if is_protected(path, prefixes)]
        if touched and not TRAILER.search(git("log", "-1", "--format=%B", sha, cwd=root)):
            errors.append(
                f"commit {sha[:10]} changes protected files ({', '.join(touched)}) "
                "without an 'Owner-Approved:' trailer"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--commits",
        metavar="RANGE",
        help="also check Owner-Approved trailers on these commits (a git range, or 'all')",
    )
    args = parser.parse_args(argv)

    errors = check_files(ROOT, repo_files()) + check_agents_budget(ROOT)
    if args.commits:
        errors += check_commits(ROOT, args.commits)

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print("check_repo: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
