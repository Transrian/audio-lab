# Evaluating a speech-to-text model

This page explains how to score a model that is served through an
OpenAI-compatible transcription API, what each number means, and what the
numbers do not cover. The code lives in `src/audiolab/eval/`; the panels it
scores are committed under `configs/panels/`, the reference texts stay in the
data directory outside git ([decision 0009](../decisions/accepted/0009-non-commercial-datasets-for-testing-only.md),
[D-013](../owner/directives.md)).

## 1. Prepare the data (once)

Data never lives in the repository. Point `AUDIOLAB_DATA` at a directory with
room for a few gigabytes:

```bash
export AUDIOLAB_DATA=/mnt/data/audio-lab/data
audiolab data list                      # what can be prepared
audiolab data fetch                     # everything, or: audiolab data fetch fleurs-fr
audiolab data panels fleurs-fr          # freeze fast/full/sealed id lists
audiolab data panels fleurs-fr --check  # the panels still match this data
```

`data fetch` is idempotent: a prepared dataset whose audio still matches the
hashes in its manifest is left alone, and `--force` prepares it again. Each
prepared dataset is a `manifest.jsonl` plus 16 kHz mono 16-bit WAV files; the
manifest holds the reference text, the speaker (or sentence) a clip belongs to
and the license. `docs/stt/datasets.md` has one card per dataset.

`data panels` writes `configs/panels/<dataset>-{fast,full,sealed}.txt`. A panel
is a frozen list of ids, so it can be committed without committing any dataset
content. Building refuses to replace an existing file, because rebuilding a
panel from another download would silently change what a past score meant
([decision 0002](../decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)).

The sealed panel is a one-time measurement. The runner refuses it unless
`--sealed` is passed, and says so again when it is.

## 2. Describe the model

A model config names the endpoint's environment variables, never the URL or the
key ([D-015](../owner/directives.md)):

```yaml
# configs/models/example.yaml
name: example            # used in run paths and reports
api: transcriptions      # POST {base_url}/v1/audio/transcriptions
base_url_env: STT_BASE_URL
api_key_env: STT_API_KEY # optional
model: served-model-id   # the `model` form field
send_language: true      # set false for servers that reject `language`
request: {}              # extra form fields, for example {temperature: 0}
```

```bash
export STT_BASE_URL=https://server.example/v1   # or the server root
export STT_API_KEY=...                          # only if the server needs one
```

## 3. Run and read the result

```bash
audiolab eval --model configs/models/example.yaml --suite configs/suites/fast.yaml
audiolab eval --model configs/models/example.yaml --suite configs/suites/full.yaml --concurrency 4
```

Everything lands in `$AUDIOLAB_RUNS/eval/<model name>/<suite>/`:

- `records.jsonl`: one line per clip, appended and flushed as soon as the
  answer arrives (`id`, `panel`, `hypothesis`, `error`, `latency_s`, `cold`,
  `language_sent`, `timestamp`).
- `run.json`: the run identity — model config without secret values, a hash of
  the endpoint host (never the URL), the panel hashes, the audiolab version and
  the git commit.
- `summary.json`: every number at full precision. `audiolab score <run dir>`
  writes it again from the records alone, byte for byte.
- `report.md`: the same numbers rounded, plus what the run does not measure.

Rerunning the command resumes: recorded clips are skipped. If the model config,
the endpoint host, a panel or the normalizer changed, the identity differs and
the runner refuses instead of mixing two experiments; `--fresh` starts a new
directory named with a timestamp.

Compare two runs over the ids they share:

```bash
audiolab compare runs/eval/model-a/full runs/eval/model-b/full
```

A difference is only reported as a difference when the 95 % interval of `B - A`
stays away from zero; otherwise the verdict is `not separable`.

## 4. Text normalization (three views)

Model output and reference text never match character for character. Both sides
go through the same function, so a difference in a score always comes from the
text, never from the processing. `NORMALIZER_VERSION` is recorded in every
summary because changing a rule changes scores.

| view | rule | example: reference vs hypothesis | equal? |
| --- | --- | --- | --- |
| `formatted` | case kept, sentence punctuation split off words | `Bonjour, monde.` vs `Bonjour , monde .` | yes |
| `formatted` | — | `Bonjour, monde.` vs `bonjour monde` | no |
| `strict` | casefold, punctuation removed, `'` and `-` kept inside words | `l'homme` vs `l homme` | no |
| `strict` | — | `chat.` vs `chat` | yes |
| `normalized` | `strict` + numbers verbalized | `21 %` vs `vingt et un pour cent` | yes |
| `normalized` | — | `12,50 €` vs `douze euros cinquante` | yes |
| `normalized` | filler words dropped | `euh le chat` vs `le chat` | yes |
| `normalized` | homophones are **not** folded | `son` vs `sont` | no |

`formatted` is only computed when the reference has casing and punctuation
(`text_has_casing_punct` in the manifest); elsewhere the report shows `n/a`.

The `normalized` view verbalizes cardinals, decimals (either decimal mark),
ordinals, percentages, currencies, clock times and the abbreviations that are
unambiguous (`M.` → `monsieur`, `5 km` → `cinq kilomètres`), and it maps a small
list of British to American spellings. Non-speech clips carry no language, so
they are normalized in the neutral view only: enough to tell an invented word
from silence.

The behaviour is pinned by a trap set of 220 pairs in
`tests/data/normalizer_traps.jsonl` (`ref`, `hyp`, `language`, `view`,
`expect_equal`), 185 of which must compare equal. It is agent-written and has
not been reviewed by the owner; every pair is asserted by
`tests/test_eval_normalize.py`.

## 5. What each number means

Per utterance, the tokens of the hypothesis are aligned with the tokens of the
reference (Levenshtein), giving substitutions, deletions and insertions.
`WER = (S + D + I) / reference words`.

Speech panels:

| number | definition | example |
| --- | --- | --- |
| corpus WER | total errors over total reference words (micro), with a 95 % interval | 30 errors over 600 words = 5 % |
| sentence error rate | share of utterances with at least one error | 12 of 60 = 20 % |
| p90 / p95 | per-utterance WER quantiles: how bad the worst tenth or twentieth is | p95 = 41 % means 5 % of utterances are worse |
| share > 25 % / > 50 % | utterances a reader would call unusable or unusable twice over | 7 % / 2 % |
| loop rate | share of hypotheses repeating an n-gram (n = 1…4) three times in a row where the reference does not | `le chat le chat le chat` for `le chat` |
| truncation rate | reference of 5 words or more answered with fewer than 80 % of its words | 10-word reference, 3-word answer |
| empty output rate | share of hypotheses with no tokens after normalization | `""` or `euh` |
| failed requests | calls that never produced text; they are counted, never scored as empty | HTTP 500 after three attempts |
| latency | median and p95 of warm calls, cold excluded, plus mean and median RTF | 0.42 s for a 6 s clip = 0.07 |

Zero-reference panels have no WER, so `n/a` is reported rather than 0.

Non-speech panels (the hallucination battery):

| number | definition |
| --- | --- |
| hallucination rate | share of clips whose normalized hypothesis is not empty, with a 95 % interval |
| words per clip | mean, maximum, and share with 5 words or more |
| top hypotheses | the ten most frequent normalized outputs with their counts |
| by category | the same rate per `category` (silence, noise, ESC-50 class group) |

## 6. Confidence intervals

Intervals come from a bootstrap that resamples **groups**, not utterances
(`B = 10 000`, seed 20261107). Utterances of one speaker share an accent, a
recording and a topic, so their errors are correlated; resampling utterances
alone would narrow the interval and let noise look like a difference. The
`audiolab compare` interval is paired over the ids both runs recorded, with the
same group resampling.

## 7. What this does not measure

- noise and reverberation;
- long-form audio (more than one utterance per request);
- term recall, named entities and accents;
- a semantic score from a language model;
- local models on our own GPUs (phase 1b);
- music: the non-speech battery covers silence, noise and everyday ESC-50
  sounds only;
- the panels are small and were chosen by us, so absolute numbers use them as
  the population they are.
