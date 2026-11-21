#!/usr/bin/env python3
"""Diagnostic re-scoring of finished screening runs: official normalizer plus two style rules.

This re-scores the per-clip answer files of finished runs (under $AUDIOLAB_RUNS)
with the frozen normalizer (version 1, decision 0002) and, on top of it, the
extra style rules of proposed decision 0013 as the published diagnostic ran
them, on both sides (reference and hypothesis):

- English contractions, mapped to a canonical expanded form from a fixed table,
  with ``'s`` always read as ``is`` after a pronoun;
- the French elision spacing: ``j' ai`` is scored as ``j'ai``.

Two things the official version 2 does are therefore **not** in this default,
and each has a switch: ``'s`` before ``been``/``got``/``gotten`` is ``has`` only
under ``--s-context`` (rule 2 of the proposal, which the published numbers
predate), and scoring a run under version 2 at all needs ``--normalizer 2``
*and* a run that records version 2.

Nothing here is an official score: the numbers go through the official pipeline
— tokenization and edit counts from ``audiolab.eval.metrics``, the per-group
bootstrap and the paired difference interval from ``audiolab.eval.stats`` — and
the diagnostic transform lives only in this script. By default, every
candidate's official column is cross-checked against the official
``audiolab.eval.report.compare_runs`` and the run stops if they diverge.

A run whose ``run.json`` records another normalizer version than the one
selected (``--normalizer``, default 1, the version the diagnostic was measured
with) is refused rather than scored under rules it was not produced with; the
report says which version was used. Since no screening run records 2,
``--normalizer 2`` needs a run directory whose own ``run.json`` says 2.

See the adjacent README for the question, the reproduction command, the
ambiguities of the rule tables and the limits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from audiolab.eval import metrics, stats
from audiolab.eval.normalize import TYPOGRAPHIC, V1
from audiolab.eval.records import (
    IDENTITY_NAME,
    RECORDS_NAME,
    latest_records,
    load_panel_items,
    read_records,
    with_records,
)
from audiolab.eval.report import FEW_GROUPS_CAUTION, compare_runs


class Problem(ValueError):
    """An input cannot be re-scored as a complete, comparable screening run."""


# ---------------------------------------------------------------------------
# The diagnostic rules. They live in this script only (decision 0007: an
# experiment is not a source of official scoring rules) and apply to both
# sides of a score, before the frozen normalizer.

# The official normalizer maps these to the ASCII apostrophe; the rules below
# need the ASCII form, so the same mapping is done here first (idempotent). Only
# the apostrophes: the rest of the normalizer's table maps characters that are
# not an elision at all (the em dash, the guillemets), and folding those here
# would move numbers for a reason that has nothing to do with the rules.
_APOSTROPHES = {character: "'" for character in ("\u2018", "\u2019", "\u02bc")}
assert set(_APOSTROPHES) <= set(TYPOGRAPHIC), "the normalizer's table must hold these"

# Fixed table of the common English contractions, in canonical expanded form,
# case-folded (the frozen normalizer case-folds anyway).
CONTRACTIONS: dict[str, str] = {
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "let's": "let us",
}

# ``'s`` is ambiguous (contraction or possessive): it is expanded only after
# these words, never for a possessive such as ``john's``. Which reading it takes
# is rule 2 of proposed decision 0013: ``has`` before a participle ("it's been",
# "he's got", "that's gotten"), ``is`` everywhere else. The published diagnostic
# predates that rule and read ``is`` everywhere, so rule 2 is the opt-in
# ``--s-context``; a verbatim ``has`` then still counts as an error without it.
S_PRONOUNS = ("it", "that", "there", "here", "what", "who", "he", "she", "where", "how")
S_HAS_FOLLOWERS = frozenset({"been", "got", "gotten"})
_S_PRONOUN_RE = re.compile(rf"({'|'.join(S_PRONOUNS)})'s")

# ``'ll`` and ``'ve`` are unambiguous and expand on any stem (they cover the
# combinations the table does not list, such as ``it'll`` or ``what'll``).
# ``'d`` is never expanded: ``he'd`` is had or would, with no canonical form.


def expand_contraction(token: str, following: str = "", s_context: bool = False) -> str:
    """The canonical expanded form of one case-folded token, or the token itself.

    ``following`` is the word after the token, and it is read only when
    ``s_context`` is set: the two readings of ``'s`` are told apart by it (rule 2
    of proposed decision 0013, ``has`` before ``been``/``got``/``gotten``), while
    the default of this diagnostic reads every pronominal ``'s`` as ``is``.
    """
    if token in CONTRACTIONS:
        return CONTRACTIONS[token]
    match = _S_PRONOUN_RE.fullmatch(token)
    if match:
        verb = "has" if s_context and following.casefold() in S_HAS_FOLLOWERS else "is"
        return f"{match.group(1)} {verb}"
    if token.endswith("'ll"):
        return f"{token[:-3]} will"
    if token.endswith("'ve"):
        return f"{token[:-3]} have"
    return token


_CONTRACTION_TOKEN = re.compile(r"\b[a-z]+(?:'[a-z]+)*\b", re.IGNORECASE)
# The word after a contraction, for the ``'s`` reading: a letter followed by a
# word boundary, which is what the rule needs and not a whole tokenizer.
_FOLLOWING_WORD = re.compile(r"[a-z]+", re.IGNORECASE)


def apply_contractions(text: str, s_context: bool = False) -> str:
    """Expand the contractions of ``text``; everything else stays as written."""

    def expand(match: re.Match) -> str:
        following = _FOLLOWING_WORD.search(text, match.end()) if s_context else None
        return expand_contraction(
            match.group(0).casefold(), following.group(0) if following else "", s_context
        )

    return _CONTRACTION_TOKEN.sub(expand, text)


# The elisions a model may write with a space after the apostrophe:
# ``j' ai`` means ``j'ai``, ``d' accord`` means ``d'accord``. Any whitespace the
# model wrote counts, including a no-break space, which ``\s`` matches directly:
# only the space goes, the elision stays two tokens. An apostrophe right before
# the letter means the letter is a word of its own in a quoted expression
# (English ``rock 'n' roll``), so it is left alone; the same lookbehind leaves an
# elision at the start of a single-quoted phrase unjoined (``'j' ai faim'``),
# because no rule here can tell its apostrophe from the quotation mark's.
_ELISION_RE = re.compile(r"(?<!')(\b[cdjlmnst]|qu|jusqu|lorsqu|puisqu)'\s+(\w)")


def apply_elision(text: str) -> str:
    """Drop the space after the apostrophe of a French elision."""
    return _ELISION_RE.sub(lambda m: m.group(1) + "'" + m.group(2), text)


def _ascii(text: str) -> str:
    """NFC with the typographic apostrophes the official normalizer would map.

    The set comes from the normalizer's own table, so the two cannot drift.
    """
    folded = unicodedata.normalize("NFC", text)
    for typographic, plain in _APOSTROPHES.items():
        folded = folded.replace(typographic, plain)
    return folded


def _contractions_only(text: str, s_context: bool = False) -> str:
    return apply_contractions(_ascii(text), s_context)


def _elision_only(text: str) -> str:
    return apply_elision(_ascii(text))


def diagnostic_text(text: str, s_context: bool = False) -> str:
    """The full diagnostic transform: both rules, on ASCII-folded text."""
    return apply_elision(apply_contractions(_ascii(text), s_context))


def transform_for(rules: str, s_context: bool = False) -> Callable[[str], str]:
    """The text transform the diagnostic column uses; ``none`` is the official view."""
    if rules == "both":
        return lambda text: diagnostic_text(text, s_context)
    if rules == "contractions":
        return lambda text: _contractions_only(text, s_context)
    if rules == "elision":
        return _elision_only
    if rules == "none":
        return _identity
    raise Problem(f"unknown rules {rules!r}; expected both, contractions, elision or none")


def _identity(text: str) -> str:
    """The official view: the text as the frozen normalizer reads it."""
    return text


# ---------------------------------------------------------------------------
# Reading runs and pairing the runs to compare.


def read_run(path: Path, normalizer: int = V1) -> tuple[dict, list[dict]]:
    """The identity and records of one finished run, or a Problem.

    The recorded normalizer version is checked against the one selected, not
    only against the other run's: re-scoring a frozen run under another rule set
    would otherwise happen silently, under a report still labelled with the
    version the run recorded.
    """
    identity_path, records_path = path / IDENTITY_NAME, path / RECORDS_NAME
    if not identity_path.is_file():
        raise Problem(f"{identity_path}: missing")
    if not records_path.is_file():
        raise Problem(f"{records_path}: missing")
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        records = read_records(records_path)
    except (OSError, ValueError) as failure:
        raise Problem(f"{path}: unreadable ({failure})") from failure
    recorded = identity.get("normalizer_version")
    if recorded != normalizer:
        raise Problem(
            f"{identity_path}: the run records normalizer version {recorded}, but this "
            f"script scores with version {normalizer}; these runs were not scored under "
            "the rules this diagnostic applies"
        )
    declared = identity.get("panels")
    if not isinstance(declared, dict) or not declared:
        raise Problem(f"{identity_path}: names no panels")
    if any(panel.endswith("-sealed") for panel in declared):
        raise Problem(f"{path}: sealed final panels are not for exploratory re-scoring")
    if any(
        not isinstance(record, dict) or not record.get("id") or not record.get("panel")
        for record in records
    ):
        raise Problem(f"{records_path}: every record needs an id and a panel")
    if {record["panel"] for record in records} != set(declared):
        raise Problem(f"{path}: records do not cover exactly the declared panels")
    return identity, records


def paired_items(
    panel: str,
    panels_dir: Path,
    data_dir: Path | None,
    records_base: list[dict],
    records_cand: list[dict],
    normalizer: int = V1,
) -> tuple[list[metrics.Item], list[metrics.Item]]:
    """The clips both runs answered, in panel order.

    The same pairing as ``audiolab.eval.report.compare_runs``: an id that
    failed in either run has no answer on both sides and leaves the comparison.
    """
    latest_base, latest_cand = latest_records(records_base), latest_records(records_cand)
    answered_base = {clip_id for clip_id, record in latest_base.items() if not record.get("error")}
    answered_cand = {clip_id for clip_id, record in latest_cand.items() if not record.get("error")}
    shared = answered_base & answered_cand
    if not shared:
        raise Problem(f"{panel}: the two runs share no answered ids, so nothing can be compared")
    items = [item for item in load_panel_items(panel, panels_dir, data_dir) if item.id in shared]
    return (
        with_records(items, records_base, normalizer=normalizer),
        with_records(items, records_cand, normalizer=normalizer),
    )


# ---------------------------------------------------------------------------
# Scoring: the official items, tokenization, edit counts and intervals, with the
# transform applied to both stored texts first. Nothing is re-derived here: the
# transform is the only difference from the official path.


def transformed(items: list[metrics.Item], transform: Callable[[str], str]) -> list[metrics.Item]:
    """The items as the diagnostic reads them: the transform on both stored texts."""
    return [
        replace(item, reference=transform(item.reference), hypothesis=transform(item.hypothesis))
        for item in items
    ]


def value_for(items: list[metrics.Item]):
    """The official headline value of a panel: WER if it has speech, hallucination if not."""
    speech = any(item.language != "none" for item in items)
    return metrics.wer_value("normalized") if speech else metrics.hallucination_value


def paired_delta(
    items_base: list[metrics.Item],
    items_cand: list[metrics.Item],
    transform: Callable[[str], str],
    *,
    draws: int,
    seed: int,
) -> dict:
    """Candidate minus base, with the official paired interval over groups.

    ``draws`` is also the distribution the interval is read from, so where a
    verdict sits on the zero line can be counted rather than guessed.
    """
    value = value_for(items_base)
    totals_base = metrics.totals_by_group(
        metrics.score_items(transformed(items_base, transform)), value
    )
    totals_cand = metrics.totals_by_group(
        metrics.score_items(transformed(items_cand, transform)), value
    )
    difference = stats.ratio(totals_cand) - stats.ratio(totals_base)
    draws_difference = stats.difference_draws(totals_base, totals_cand, draws=draws, seed=seed)
    return {
        "base": stats.ratio(totals_base),
        "candidate": stats.ratio(totals_cand),
        "difference": difference,
        "ci": stats.interval_of(draws_difference),
        "draws": {
            "below": int((draws_difference < 0).sum()),
            "on": int((draws_difference == 0).sum()),
            "total": len(draws_difference),
        },
        "groups": len(set(totals_base) & set(totals_cand)),
    }


def verdict(delta: dict) -> str:
    """The verdict one interval gives: harm, gain, or neutral.

    ``harm`` and ``gain`` are separable from the base (the interval stays away
    from zero); ``neutral`` is what the screening conclusions call no change.
    The reading of the interval itself is the official one, ``stats.verdict``,
    asked with this script's own names for the two sides: the base is the
    reference of the comparison, so naming it the first side and the run being
    judged the second keeps its wording ("base better") reading the same way as
    here, whatever column the delta came from.
    """
    low, high = delta["ci"]
    official = stats.verdict(low, high, "base", "candidate")
    return {"candidate better": "gain", "base better": "harm"}.get(official, "neutral")


def rescore(
    base_path: str | Path,
    candidate_paths: list[str | Path | tuple[str, str | Path]],
    panels_dir: str | Path,
    data_dir: str | Path | None = None,
    *,
    rules: str = "both",
    normalizer: int = V1,
    s_context: bool = False,
    draws: int = stats.DRAWS,
    seed: int = stats.SEED,
    verify_official: bool = True,
) -> dict:
    """Re-score every candidate against the base, official and diagnostic.

    The official column is the pipeline applied to the runs' records under
    ``normalizer``; the diagnostic column applies ``transform_for(rules,
    s_context)`` to both sides before it. ``verify_official`` cross-checks every
    candidate's official column against the official ``compare_runs`` and raises
    Problem on any divergence, which is what a wrong pairing would look like. A
    candidate may be a path or a ``(label, path)`` pair; the label is what the
    tables call it.

    ``s_context`` is rule 2 of proposed decision 0013 (``'s`` before
    ``been``/``got``/``gotten`` is ``has``); it is off by default because the
    numbers this diagnostic published were measured without it.

    Every run is scored under ``normalizer``, so a run whose recorded version is
    another one is refused by ``read_run``.
    """
    base_path, panels_dir = Path(base_path), Path(panels_dir)
    candidates = [_split_candidate(entry) for entry in candidate_paths]
    candidates = [(label, Path(path)) for label, path in candidates]
    if not candidates:
        raise Problem("no candidate run given")
    transform = transform_for(rules, s_context)
    identity, base_records = read_run(base_path, normalizer)
    result: dict = {
        "base": str(base_path),
        "base_name": identity.get("model", {}).get("name", str(base_path)),
        # What scored these numbers: the version selected here, checked against
        # every run's recorded one in read_run. Never the runs' claim on their own.
        "normalizer_version": normalizer,
        "rules": rules,
        "s_context": s_context,
        "draws": draws,
        "seed": seed,
        "panels": {},
        "changes": [],
    }
    for label, candidate_path in candidates:
        cand_identity, cand_records = read_run(candidate_path, normalizer)
        if cand_identity.get("panels") != identity.get("panels"):
            raise Problem(
                f"{candidate_path}: scored different panels than {base_path}; "
                "only runs over the same frozen panels compare"
            )
        name = label or display_label(candidate_path, cand_identity)
        for panel in sorted(identity["panels"]):
            items_base, items_cand = paired_items(
                panel, panels_dir, data_dir, base_records, cand_records, normalizer
            )
            official = paired_delta(items_base, items_cand, _identity, draws=draws, seed=seed)
            diagnostic = paired_delta(items_base, items_cand, transform, draws=draws, seed=seed)
            kind = "nonspeech" if all(item.language == "none" for item in items_base) else "speech"
            section = result["panels"].setdefault(
                panel, {"kind": kind, "caution": "", "candidates": {}}
            )
            if official["groups"] < stats.MINIMUM_GROUPS_FOR_INTERVALS:
                section["caution"] = FEW_GROUPS_CAUTION
            section["candidates"][name] = {"official": official, "diagnostic": diagnostic}
            if rules != "none":
                before, after = verdict(official), verdict(diagnostic)
                if before != after:
                    result["changes"].append(
                        {
                            "panel": panel,
                            "source": name,
                            "official": before,
                            "diagnostic": after,
                            "position": diagnostic["draws"],
                        }
                    )
        if verify_official:
            comparison = compare_runs(
                base_path,
                candidate_path,
                panels_dir,
                data_dir,
                draws=draws,
                seed=seed,
                normalizer=normalizer,
            )
            for panel, section in sorted(result["panels"].items()):
                theirs, mine = comparison["panels"][panel], section["candidates"][name]["official"]
                if (
                    abs(theirs["difference"] - mine["difference"]) > 1e-9
                    or abs(theirs["ci"]["low"] - mine["ci"][0]) > 1e-9
                    or abs(theirs["ci"]["high"] - mine["ci"][1]) > 1e-9
                ):
                    raise Problem(
                        f"{panel}: the official column diverges from compare_runs "
                        f"({mine['difference']:.6f} vs {theirs['difference']:.6f}); "
                        "the pairing is not the official one"
                    )
    return result


def _split_candidate(entry: str | Path | tuple[str, str | Path]) -> tuple[str | None, str | Path]:
    """A candidate as ``(label, path)``; a bare path carries no label.

    ``--candidate PATH`` arrives from ``parse_candidate`` as ``(None, path)``, so
    an empty label has to stay empty: ``str(None)`` is the four letters of the
    word "None", which would become the table row's name.
    """
    if isinstance(entry, tuple):
        label, path = entry
        text = "" if label is None else str(label).strip()
        return (text or None, path)
    return (None, entry)


# ---------------------------------------------------------------------------
# The presentation, in the order of docs/stt/screening.md.

CANDIDATE_ORDER: tuple[tuple[str, str], ...] = (
    ("screen-granary-fr", "Granary"),
    ("screen-voxpopuli-fr", "VoxPopuli FR"),
    ("screen-mls-fr", "MLS FR"),
    ("screen-african-accented-fr-uncleaned-2026-11-08", "African-accented FR (uncleaned)"),
    ("screen-librispeech-en", "LibriSpeech EN"),
    ("screen-icsi-en-train", "ICSI EN"),
    ("screen-notsofar-en-train", "NOTSOFAR EN"),
    ("screen-librispeech-pc-en", "LibriSpeech-PC EN"),
    ("screen-pxcorpus-fr", "PxCorpus FR"),
    ("screen-african-accented-fr", "African-accented FR (cleaned)"),
    ("screen-ami-fr-speakers", "AMI FR speakers"),
    ("screen-peertube-fr", "PeerTube FR"),
    ("screen-peertube-edu-fr", "PeerTube EDU FR"),
    ("screen-debconf-en", "DebConf EN"),
    ("screen-fosdem-en", "FOSDEM EN"),
    ("screen-nasa-en", "NASA EN"),
)

PANEL_ORDER: tuple[str, ...] = (
    "fleurs-fr-full",
    "mls-fr-full",
    "servicenow-codeswitched-en-fr-full",
    "servicenow-codeswitched-en-fr-ca-full",
    "cs-fleurs-fr-en-full",
    "voxpopuli-en-nonnative-full",
    "fleurs-en-full",
    "librispeech-en-full",
    "nonspeech-full",
)

RULES_LABEL = {
    "both": "English contractions to their expanded form + French elision spacing",
    "contractions": "English contractions to their expanded form only",
    "elision": "French elision spacing only",
    "none": "no extra rule (official view twice, a sanity check)",
}


def display_label(path: Path, identity: dict) -> str:
    """The screening name of a run directory, or its recorded model name."""
    directory = path.parent.name
    for name, label in CANDIDATE_ORDER:
        if name == directory:
            return label
    return identity.get("model", {}).get("name", directory)


def ordered_candidates(result: dict) -> list[str]:
    """Candidates in screening order; unknown ones last, alphabetically."""
    names = {name for section in result["panels"].values() for name in section["candidates"]}
    known = [label for directory, label in CANDIDATE_ORDER if label in names]
    return known + sorted(names - set(known))


def changed_rows(result: dict) -> list[dict]:
    """The changed conclusions in the tables' own order, whatever order they were found."""
    order = {label: index for index, (_, label) in enumerate(CANDIDATE_ORDER)}
    return sorted(
        result["changes"],
        key=lambda change: (
            change["panel"],
            order.get(change["source"], len(order)),
            change["source"],
        ),
    )


def ordered_panels(result: dict) -> list[str]:
    names = set(result["panels"])
    known = [panel for panel in PANEL_ORDER if panel in names]
    return known + sorted(names - set(known))


def _points(value: float) -> str:
    return f"{value * 100:+.2f}"


def _s_context_note(result: dict) -> str:
    """How the diagnostic read ``'s``, said only when a contraction rule is on."""
    if result["rules"] not in ("both", "contractions"):
        return ""
    if result["s_context"]:
        return ", 's read from the next word (rule 2 of proposed decision 0013)"
    return ", 's always read as is (the published measurement)"


def _cell(delta: dict) -> str:
    """One delta in points with its interval, bold when the interval excludes zero."""
    low, high = delta["ci"]
    text = f"{_points(delta['difference'])} [{_points(low)}, {_points(high)}]"
    return f"**{text}**" if low > 0 or high < 0 else text


def render(result: dict) -> str:
    """The diagnostic as one table per panel, plus the changed conclusions."""
    lines = [
        "# Normalization diagnostic: screening runs re-scored with extra rules",
        "",
        f"- base: `{result['base']}` (decision 0002 frozen)",
        f"- normalizer version used for these numbers: {result['normalizer_version']} "
        "(`audiolab.eval.normalize`), the version every run here records and the one "
        "selected with --normalizer",
        f"- diagnostic view: official normalizer plus {RULES_LABEL[result['rules']]}"
        f"{_s_context_note(result)}; the extra rules are applied in this script only, "
        "to both sides",
        f"- intervals: official paired bootstrap over groups, {result['draws']} draws, "
        f"seed {result['seed']} (`audiolab.eval.stats`)",
        "- pairing: the ids both runs answered, per panel, the official `compare_runs` "
        "pairing; the official column was verified against it for every candidate "
        "(skip with --no-verify-official)",
        "- deltas are candidate minus base, in percentage points; negative is better; "
        "bold means the interval excludes zero",
        "",
    ]
    for panel in ordered_panels(result):
        section = result["panels"][panel]
        lines += [f"## {panel}", ""]
        if section["caution"]:
            lines += [f"- **{section['caution']}**", ""]
        lines += [
            f"- {section['kind']}: candidate minus base, with the 95 % interval",
            "",
            "| source | official Δ vs base | diagnostic Δ vs base |",
            "| --- | --- | --- |",
        ]
        for name in ordered_candidates(result):
            entry = section["candidates"][name]
            lines.append(f"| {name} | {_cell(entry['official'])} | {_cell(entry['diagnostic'])} |")
        lines.append("")
    lines += ["## Conclusion changes", ""]
    if result["changes"]:
        lines += [
            "The verdict one panel/source has moves from separable to neutral, or flips side:",
            "",
            "| panel | source | official | diagnostic | draws on zero, below it |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {change['panel']} | {change['source']} | {change['official']} "
            f"| {change['diagnostic']} | {change['position']['on']}, "
            f"{change['position']['below']} of {change['position']['total']} |"
            for change in changed_rows(result)
        ]
    else:
        lines.append(
            "No conclusion changed: every panel/source verdict is the same under both views."
        )
    return "\n".join(lines) + "\n"


def recorded_panels(path: Path) -> dict:
    """The panels a run records, without ``read_run``'s version check."""
    return json.loads((path / IDENTITY_NAME).read_text(encoding="utf-8"))["panels"]


def default_candidates(runs_dir: Path, panels: dict | None = None) -> list[Path]:
    """The finished screen runs scored on the same panels as the base.

    That is the 16 rows of the screening tables. A later run over another panel
    set (the 17-panel NASA runs) is left out instead of failing the whole
    diagnostic on the first candidate.
    """
    paths = []
    for path in sorted(runs_dir.glob("screen-*/full")):
        identity_path = path / IDENTITY_NAME
        if not identity_path.is_file():
            continue
        if panels is not None:
            recorded = json.loads(identity_path.read_text(encoding="utf-8")).get("panels")
            if recorded != panels:
                continue
        paths.append(path)
    if not paths:
        raise Problem(f"{runs_dir}: no screen-*/full run on the base's panels found")
    return paths


def parse_candidate(text: str) -> tuple[str | None, Path]:
    """A candidate as ``label=path``, or a bare path that keeps the run's own name.

    Same convention as ``experiments/source-screening/table.py``: an explicit
    label is what orders and names the table row.
    """
    label, separator, path = text.partition("=")
    if not separator:
        return None, Path(text.strip())
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"{text!r} is not label=path")
    return label.strip(), Path(path.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score finished screening runs with a normalizer version plus the "
            "contraction and elision rules. Diagnostic only: it changes no official "
            "score and writes nothing to the run directories."
        ),
        epilog=(
            "example:\n"
            "  set -a; . ./.env; set +a\n"
            "  uv run python experiments/normalization-review/rescore.py\n"
            "A candidate is a run directory, or LABEL=PATH to name its table row "
            "(as in experiments/source-screening/table.py).\n"
            "Defaults: base $AUDIOLAB_RUNS/eval/parakeet-tdt-0.6b-v3-nemo/"
            "full-9panels-v1, candidates the $AUDIOLAB_RUNS/eval/screen-*/full runs "
            "scored on that base's nine panels (the 16 rows of the tables), panels "
            "configs/panels, data $AUDIOLAB_DATA. No model, audio or network request "
            "is used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--candidate", type=parse_candidate, action="append", default=None, metavar="[LABEL=]PATH"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="directory holding the run directories (default: $AUDIOLAB_RUNS/eval)",
    )
    parser.add_argument("--panels-dir", type=Path, default=Path("configs/panels"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="dataset manifests (default: $AUDIOLAB_DATA)",
    )
    parser.add_argument(
        "--rules",
        choices=("both", "contractions", "elision", "none"),
        default="both",
        help="which extra rules the diagnostic column uses (default: both)",
    )
    parser.add_argument(
        "--normalizer",
        type=int,
        choices=(V1, 2),
        default=V1,
        help="normalizer version the official column uses, and the version every run "
        "must record (default: 1, the version the diagnostic was measured with)",
    )
    parser.add_argument(
        "--s-context",
        action="store_true",
        help="read 's from the next word (has before been/got/gotten), rule 2 of "
        "proposed decision 0013; off by default, because the published numbers "
        "predate it and read 's as is everywhere",
    )
    parser.add_argument(
        "--no-verify-official",
        action="store_true",
        help="skip the cross-check of the official column against compare_runs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        env_runs = os.environ.get("AUDIOLAB_RUNS")
        runs_dir = args.runs_dir or (Path(env_runs) / "eval" if env_runs else None)
        if runs_dir is None:
            raise Problem("set AUDIOLAB_RUNS or pass --runs-dir")
        base = args.base or (runs_dir / "parakeet-tdt-0.6b-v3-nemo" / "full-9panels-v1")
        candidates = args.candidate or default_candidates(runs_dir, recorded_panels(base))
        result = rescore(
            base,
            candidates,
            args.panels_dir,
            args.data_dir,
            rules=args.rules,
            normalizer=args.normalizer,
            s_context=args.s_context,
            verify_official=not args.no_verify_official,
        )
    except (OSError, ValueError, KeyError, TypeError) as failure:
        print(f"rescore.py: {failure}", file=sys.stderr)
        return 2
    print(render(result), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
