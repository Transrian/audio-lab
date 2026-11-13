# NASA filters: the facts behind the Curious Universe and edge-gate questions

**Status:** the read-only analysis behind two open questions of the NASA
source — what NASA's Curious Universe show is worth in the prepared set, and
what the edge-word gate drops during a preparation. It reads the files the
preparation and the first screen already wrote; it never re-runs a model, a
fetch or a preparation.

The first [screen](../../docs/stt/screening.md#what-each-source-does-alone) of
`nasa-en-train` (2026-11-09) was mixed, and the labels were rewritten for the
characters the base model can write. Two questions stayed open, answered here
from existing files:

1. **Curious Universe** — the show whose episodes carry the third-party music
   credits and most of the played recordings that the listing leaves out: how
   much of the prepared set does it still hold?
2. **The edge-word gate** — a clip is kept only when the base model hears the
   label's first and last words at its edges: how many clips does the gate
   drop, per show, and what is recorded about them?

[`analyze.py`](analyze.py) reports five tables.

## Environment and inputs

From the repository root, use the default Python 3.12 environment (`uv sync`);
the script calls the core's manifest reader (`audiolab.data.store`) and the
label rule (`audiolab.data.train.unwritable`) and adds no dependency.

Inputs, all read-only, from the environment
(`set -a; . ./.env; set +a`):

- `$AUDIOLAB_DATA/nasa-en-train/manifest.jsonl` — the current preparation;
- `$AUDIOLAB_DATA/nasa-en-train.curly-2026-11-09/manifest.jsonl` — the first,
  typographic, preparation it was rewritten from;
- `$AUDIOLAB_DATA/nonspeech/manifest.jsonl` — the category of each
  non-speech clip;
- `$AUDIOLAB_CACHE/audiolab/nasa-en-train/*.stt-c-parakeet-tdt-0.6B-v3.clips.jsonl`
  — the base model's answer for every clip the gate asked about;
- `$AUDIOLAB_RUNS/eval/screen-nasa-en/full/records.jsonl` — the first screen's
  answers.

Help needs neither data nor environment variables:

```bash
uv run python experiments/nasa-filters/analyze.py --help
```

## Reproduce the tables

```bash
set -a; . ./.env; set +a
uv run python experiments/nasa-filters/analyze.py
```

Flags override every default: `--data-dir`, `--runs-dir`, `--cache-dir`,
`--dataset` (default `nasa-en-train`), `--old-dataset` (default
`nasa-en-train.curly-2026-11-09`), `--eval-run` (default
`<runs-dir>/eval/screen-nasa-en/full`), `--panel` (default `nonspeech-full`)
and `--base-model` (default `stt-c-parakeet-tdt-0.6B-v3`, the name its
per-clip cache files carry). The tables go to standard output; a missing
manifest, a missing run or a malformed cache file is a standard-error error
with exit `2`.

## Output

1. **The current set, per show** — episodes (its distinct groups), clips,
   hours and mean `base_wer`, one row per show.
2. **Curious Universe in the current set** — its episodes, its clips and its
   share of the set.
3. **The first screen's non-empty answers on the non-speech panel** — every
   non-empty answer with the category of the clip it was said on (silence,
   noise, or the ESC-50 class), and how many answers *look like podcast
   speech*: three or more words with two of them different ("So they use the
   two microphones." yes; "Oh", "Yeah, yeah, yeah." and "BIT! BIT!" no). The
   unknown token is not a word for that count.
4. **The edge-word gate, per show** — `asked`: the clips the base model was
   asked about at the preparation (its answers are in the per-clip cache, one
   line per clip, keyed by the clip's stretch of audio); `gate-dropped`: the
   asked clips the first set does not hold; `rewrite-dropped`: the clips the
   first set holds that the current set does not (the label rewrite lost them
   to a character still outside the vocabulary, not to the gate).
5. **The characters of the current set's labels** — the clips that hold a
   character the base model cannot write, the same check as
   [label-text](../label-text/); a clean set prints the zero line.

## Limits

- The match between a manifest clip and its cached answer is the clip's start:
  the id carries it in centiseconds, the cache key in seconds with two
  decimals, so the two are the same number and the match is exact.
- `gate-dropped` is the difference "in the cache but not in the first set",
  which the preparation's order guarantees to be the gate's own drops: every
  clip of the first set was asked, and no other rule drops an asked clip.
- The dropped clips' `base_wer` is **not recorded**: the cache holds the base
  model's raw text for each asked clip but not its label, so the error of a
  dropped clip cannot be computed from the existing files. Getting it would
  mean re-running the base model on audio the gate already rejected, which
  this analysis does not do.
- "Looks like podcast speech" is a shape rule on the answer's words, not a
  judgement about the audio; the table shows every answer with its category,
  so the rule's borderlines stay visible.
- The script opens no audio, sends no request and starts no server, and
  writes nothing under the data, runs or cache roots, not even a temporary
  file: re-running it changes nothing and says the same.

[Tests](../../tests/test_nasa_filters.py) run the tables on synthetic
manifests, a synthetic per-clip cache and synthetic run records: the per-show
rows, the share, the speech-likeness rule, the gate's arithmetic (asked,
dropped, rewrite, and the failure on a kept clip with no cached answer), the
zero line of the character check, and a snapshot assertion that nothing under
the input roots is created, written or renamed.
