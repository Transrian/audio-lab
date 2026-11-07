# 0010 — STT evaluation core (phase 1a)

- Scope: stt
- Derived from: D-005, D-009, D-015, decisions 0002, 0008, 0009
- Decided by: proposed by an agent, 2026-11-07; awaiting owner review

## Proposal

The first version of the evaluation measures, for any model behind an
OpenAI-compatible transcription API:

- **Panels**: frozen id lists per dataset, split into `fast` (100 items),
  `full` (up to 600) and `sealed` (20 % of speakers, never scored without an
  explicit owner request). First datasets: FLEURS (fr, en), MLS French,
  LibriSpeech test-clean, and a 300-clip non-speech battery (silence,
  low-level noise, ESC-50 sounds).
- **Three WER views**: formatted (case and punctuation kept), strict
  (lowercase, no punctuation), normalized (strict plus numbers spelled out and
  fillers removed). The normalized one is the headline.
- **Beyond the average**: sentence error rate, the worst 10 % and 5 %,
  share of utterances above 25 % and 50 % WER, repetition loops,
  truncations, empty outputs, failed requests.
- **Hallucinations**: share of non-speech clips for which the model writes
  something, and what it writes.
- **Statistics**: 95 % confidence intervals by bootstrap over speakers (or
  the closest correlated unit), and paired comparisons between two models with
  a verdict: better, worse, or not separable.

Sizing follows published guidance: about 100 utterances detect only large
regressions (around 5 points); 400–600 detect about 2 points; about 2 000 are
needed for 1 point.

## Why

Leaderboard averages on clean read speech hide the failures users notice:
inventions on silence, loops, truncations, and errors that normalization
erases. The previous private evaluation never measured them.

## Alternatives considered

- Per-utterance bootstrap (used in the earlier private work). Proposed
  replacement, because it gives intervals that are too narrow when one
  speaker's utterances are correlated.
