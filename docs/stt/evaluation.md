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
key ([D-015](../owner/directives.md)). `server_label` is the one place a run can
say which server answered: it is free text in the config, it is recorded in the
run identity, and it must not be a URL or a host name.

```yaml
# configs/models/example.yaml
name: example            # used in run paths and reports
api: transcriptions      # POST {base_url}/v1/audio/transcriptions
base_url_env: STT_BASE_URL
api_key_env: STT_API_KEY # optional
model: served-model-id   # the `model` form field
send_language: true      # set false for servers that reject `language`
server_label: ""         # optional free text: which server answered ("our box")
request: {}              # extra form fields, for example {temperature: 0}
```

```bash
export STT_BASE_URL=https://server.example     # the server root
export STT_BASE_URL=https://server.example/v1  # ...or the version root: same request
export STT_API_KEY=...                         # only if the server needs one
```

`send_language: true` sends the item's language as the `language` form field. A
server that answers `400` when it does not know that field is asked **once**:
the refusal is remembered, the same clip is sent again without the field, and no
later request in the run carries a language. `records.jsonl` says per clip
whether a language was sent, so a run whose server refused it is recognisable
after the fact.

The variable holds the **server root**; a value that ends in `/v1` is accepted
too, because servers are usually documented that way, and a trailing slash is
ignored. Only one `/v1` ever ends up in the request path: a value that is only
`/v1` names no server and is refused with that message rather than being turned
into `/v1/v1/audio/transcriptions`.

## 3. Run and read the result

```bash
audiolab eval --model configs/models/example.yaml --suite configs/suites/fast.yaml
audiolab eval --model configs/models/example.yaml --suite configs/suites/full.yaml --concurrency 4
```

Everything lands in `$AUDIOLAB_RUNS/eval/<model name>/<suite>/`:

- `records.jsonl`: one line per clip, appended and flushed as soon as the
  answer arrives (`id`, `panel`, `hypothesis`, `error`, `latency_s`,
  `attempts`, `cold`, `language_sent`, `timestamp`).
- `run.json`: the run identity — model config without secret values, the panel
  hashes, the audiolab version, the git commit, and the endpoint. The URL itself
  and the API key are never stored: the endpoint is a **keyed fingerprint** of
  the server root, computed with HMAC-SHA256 and a random per-machine key in
  `<runs dir>/.identity-key` (created on first use, mode `0600`). A run
  directory can therefore be shared without naming a private address, and the
  fingerprint cannot be reversed by guessing the shape of a URL. It is only ever
  compared, never printed: `summary.json` and `report.md` do not contain it. The
  config's optional `server_label` is free text, so a run can also say in words
  which server answered.
- `summary.json`: every number at full precision, plus the run identity without
  the endpoint fingerprint. `audiolab score <run dir>` writes it again from the
  records alone, byte for byte.
- `report.md`: the same numbers rounded, plus what the run does not measure.

Rerunning the command resumes: clips already answered are skipped, and a clip
whose last record is an error is **sent again** — a failed request is not a
measurement. The failed record stays in the file (so a flaky endpoint is still
visible) and the newest record of a clip is the one that is scored; the summary
says how many lines and how many clips a run holds. If the model config, the
endpoint, a panel or the normalizer changed, the identity differs and the runner
refuses instead of mixing two experiments; the error names the field that
differs by its path (`model.model`, `panels.fleurs-fr-fast`), with its old and
new value, except for the endpoint fingerprint, which is reported only as "the
server URL is not the one recorded" so that no reading of the run directory ever
gets the fingerprint. `--fresh` starts a new directory named with a timestamp.

Compare two runs over the ids they share:

```bash
audiolab compare runs/eval/model-a/full runs/eval/model-b/full
```

A difference is only reported as a difference when the 95 % interval of `B - A`
stays away from zero; otherwise the verdict is `not separable`.

A comparison pairs only the clips **both** runs answered, because a clip without
a hypothesis cannot be paired with anything. Leaving those clips out silently
would flatter a model that fails: it would look better than a model that answers
everything and is sometimes wrong. The comparison therefore counts and lists the
excluded ids with their reason (failed in A, failed in B, not recorded in A or
B); the report shows the reasons and up to twenty ids, and `--json` holds every
id. A run whose panel holds nothing but failures cannot be compared at all, and
says so.

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

The `normalized` view verbalizes cardinals, decimals, ordinals, percentages,
currencies, clock times and the abbreviations that are unambiguous
(`Mme` → `madame`, `5 km` → `cinq kilomètres`), and it maps a small list of
British to American spellings.

Every item is scored in the `normalized` view, the non-speech battery included.
The battery has no language of its own, so its words go through the English
rules; what it asks is only whether the model said anything, and one view
everywhere keeps a word count comparable between panels.

Which mark is the decimal and which groups thousands depends on the language,
because the two conventions are the other way round:

- **English** groups thousands with a comma or a dot and writes the decimal with
  a dot: `1,500` and `1.500` are both `one thousand five hundred`, `1,000,000` is
  `one million`, and `1.5` is `one point five`. `1,5` is read as a decimal too,
  since a mark that does not group exactly three digits cannot be a thousands
  separator: reading it as fifteen would be the worse mistake;
- **French** writes the decimal with a comma and groups thousands with a dot:
  `1,5` is `un virgule cinq` and `1.500` is `mille cinq cents`. Both languages
  also write grouped thousands with spaces (`1 500`, `12 000`), which are joined
  into one number;
- `cent` and `quatre-vingt` lose their plural `s` before `mille` (`500 000` →
  `cinq cent mille`, `80 000` → `quatre-vingt mille`), because `mille` is a
  numeral and not a noun. Before `million` and `milliard`, which are nouns, the
  `s` stays (`200 000 000` → `deux cents millions`).

The rules have limits, and they are deliberate:

- four-digit English numbers from 1100 to 2099 read as a **year pair**
  (`1999` → `nineteen ninety nine`) only when the number stands **bare**: a
  number followed by a unit or a currency is a quantity, so `1999 dollars` is
  `one thousand nine hundred ninety nine dollars`. A number ending in `00` keeps
  its cardinal form (`1900` → `one thousand nine hundred`), which is why `1,500`
  reads as `one thousand five hundred` and not as `fifteen hundred`; the rule is
  English only, and French `1999` stays a cardinal;
- the decade 2000–2009 is read as a **cardinal** even on its own: `2005` is
  `two thousand five`. The spoken `twenty oh five` is a form this normalizer does
  not fold into it, so a hypothesis that says it scores a substitution; from 2010
  the pair reading applies again (`2010` → `twenty ten`);
- `M.` is **not** expanded: the dot is stripped by the strict rules, and the one
  letter left is not enough to tell an abbreviation from a word;
- `m` and `mm` are **not** expanded to metres and millimetres, although `5 km`
  is: in these panels a lone `m` is as likely to be a time unit or a filler as a
  length.

The behaviour is pinned by a trap set of 286 pairs in
`tests/data/normalizer_traps.jsonl` (`ref`, `hyp`, `language`, `view`,
`expect_equal`), 233 of which must compare equal: a pair that must compare
*different* is as important as one that must match, and the file records both.
It is agent-written and has not been reviewed by the owner; every pair is
asserted by `tests/test_eval_normalize.py`.

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
| latency | median and p95 of the warm calls' **successful request**, cold excluded, plus mean and median RTF | 0.42 s for a 6 s clip = 0.07 |
| requests | how many HTTP requests the run sent, retries included | 60 items, 2 retries = 62 |

Zero-reference panels have no WER, so `n/a` is reported rather than 0: an error
rate over zero reference words does not exist, and a model that talks over
silence has not scored 0 %. The errors it made are still counted in the battery.

Latency is the time of the request that answered, and nothing else. A backoff
sleep after a 503, or the second call a server forces after refusing the
`language` field, is not part of how long the model took to answer; the
`attempts` of an item say how many requests it cost.

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
`audiolab compare` interval is paired over the ids both runs answered, with the
same group resampling; the excluded ids are listed next to the result, so a
comparison over a third of a panel cannot pass for a comparison over all of it.
The draws are made with a fixed seed (`numpy.random.default_rng`), so the same
totals give the same interval on any machine, and they depend on the numbers
only: the order the groups happen to be stored in does not change them.

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
