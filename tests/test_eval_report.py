"""The report's gap list: what the current suite does not measure."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audiolab.eval import report  # noqa: E402


def test_gaps_describe_the_current_suite():
    # The suite measures noise, reverberation, music and the two accent panels;
    # a gap must say what those panels measure, not what an earlier suite did.
    assert "not measured in 1a" not in " ".join(report.GAPS)
    assert "local models on our own GPUs" not in " ".join(report.GAPS)
    assert any(
        gap.startswith("noise and reverberation: measured at the fixed conditions")
        for gap in report.GAPS
    )
    assert any(
        gap.startswith("music: measured as a hallucination rate on the nonspeech-music panel")
        for gap in report.GAPS
    )
    assert any(
        "african-accented-fr-test panel" in gap and "voxpopuli-en-nonnative panel" in gap
        for gap in report.GAPS
    )


def test_gaps_keep_what_still_is_not_measured():
    # The gaps that are still true keep their wording.
    assert "long-form audio (more than one utterance per request): not measured" in report.GAPS
    assert any(gap.startswith("dictation itself:") for gap in report.GAPS)
    assert any(gap.startswith("which word an accent or a code switch cost") for gap in report.GAPS)
    assert any(
        gap.startswith("these panels are small and were chosen by us") for gap in report.GAPS
    )
