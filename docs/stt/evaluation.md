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
audiolab data fetch                     # every dataset, or: audiolab data fetch fleurs-fr
audiolab data panels fleurs-fr          # freeze fast/full/sealed id lists
audiolab data panels fleurs-fr --check  # the panels still match this data
```

`data fetch` with no name prepares everything this version can prepare, the
large training sources of the per-source screening included, so naming what you
need is the cheaper habit. It is idempotent: a prepared dataset whose audio still
matches the hashes in its manifest is left alone, and `--force` prepares it
again. Each
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

## 3. Serve the checkpoint yourself

An endpoint needs a server. For a model this project trained or downloaded, the
config names one instead of a URL: `serve:` says which checkpoint to load, and
`audiolab eval` starts a small OpenAI-compatible server for it, waits until it
answers, runs the suite and stops it again. Nothing else changes — the suite, the
records and the report are the same as for a remote server.

```yaml
# configs/models/parakeet-tdt-0.6b-v3-nemo.yaml
name: parakeet-tdt-0.6b-v3-nemo
model: parakeet-tdt-0.6b-v3
api: transcriptions
send_language: true
server_label: local NeMo
serve:
  backend: nemo                            # the NeMo backend is the only one so far
  checkpoint: nvidia/parakeet-tdt-0.6b-v3  # as published: a Hub id, or a .nemo path
  checkpoint_env: AUDIOLAB_PARAKEET_NEMO   # optional: a variable holding a local copy
  gpu: "0"                                 # `audiolab eval --gpu 1` overrides it
  max_symbols: 10                          # greedy decoding stops after this many
```

```bash
export AUDIOLAB_NEMO_PYTHON=$PWD/.venv-nemo/bin/python   # the interpreter that has NeMo
export AUDIOLAB_PARAKEET_NEMO=/path/to/parakeet-tdt-0.6b-v3.nemo
audiolab eval --model configs/models/parakeet-tdt-0.6b-v3-nemo.yaml \
    --suite configs/suites/fast.yaml
```

The server needs NeMo and PyTorch, which the project itself does not depend on
(see [fine-tuning](training.md#1-the-environment-it-needs) for the environment
that has them). `audiolab eval` starts it with the interpreter
`AUDIOLAB_NEMO_PYTHON` names, and its own interpreter when that variable is
unset — so a light environment with no `.venv-nemo` fails with a message saying
which variable to set rather than with an import error.

`audiolab serve --model <config>` runs the same server in the foreground, on the
host and port you choose, for a client you drive yourself:

```bash
.venv-nemo/bin/audiolab serve --model configs/models/parakeet-tdt-0.6b-v3-nemo.yaml \
    --port 8080
curl -s localhost:8080/v1/models
curl -s -F file=@clip.wav -F model=parakeet-tdt-0.6b-v3 \
    localhost:8080/v1/audio/transcriptions
```

It answers `POST /v1/audio/transcriptions` with `{"text": "..."}` (multipart
`file`, plus `model`, an optional `language` and `response_format` of `json` or
`text`) and `GET /v1/models`. Requests are decoded one at a time, whatever the
suite's `--concurrency` is: one GPU decodes one clip at a time, and the server
queues the rest. The fields of the OpenAI request that this server does not use
are ignored, and a request it cannot serve is answered with an explicit error
rather than an empty transcript.

The decoding it imposes is recorded in the run identity: greedy, batched
(`greedy_batch`), `max_symbols` from the config, with the fused batch fast paths
that an archive carries for training turned off.

A server answers `GET /v1/models` with the identity of the checkpoint it holds —
its content, its backend and its decoding settings. `audiolab eval` reads it back
from the server it started and stops if it is not the checkpoint the config asks
for, so a port still answering in the name of another model can never fill the
cache with answers attributed to the wrong one.

With `--base-url`, or with a config that has no `serve:` block, the run talks to
an endpoint instead: the cache key is then that server's address, and the
checkpoint a config may also name is never read.

## 4. Run and read the result

```bash
audiolab eval --model configs/models/example.yaml --suite configs/suites/fast.yaml
audiolab eval --model configs/models/example.yaml --suite configs/suites/full.yaml --concurrency 4
```

**Requests at once**: a run sends one request at a time unless told otherwise,
because what a server supports varies ([D-022](../owner/directives.md)).
`--concurrency N` sends N at once. Without it, the run reads
`AUDIOLAB_CONCURRENCY`, the same variable the labelling of the talk sources
reads ([datasets](datasets.md#peertube-fr-train--capitole-du-libre-talks-in-french)):
comma-separated `<served model id>=<N>` entries, for example
`AUDIOLAB_CONCURRENCY=stt-v-whisper-large-v3=10,stt-v-qwen3-asr-1.7B=10`; a model
it does not name gets 1. Like the URL, it is a fact of the server, so it lives in
the environment (`.env`), not in a model config. It is not part of the cache
key: a server that batches requests (vLLM) may answer a clip slightly
differently depending on what was sent beside it, but that is timing, which no
key can hold.

**Warmup**: before its first request, a run sends half a second of silence and
waits until the model answers it ([D-022](../owner/directives.md)), because a
server that loads models on demand can take minutes to answer the first request
for one. While the connection is refused, times out or is dropped, or the server
answers 429, 502, 503 or 504, the silence is sent again every 5 s, and stderr
says `waiting for <model> to answer (<why>)...`, for example `(HTTP 503)`; after
about 15 minutes (an attempt under way then is not cut short) the run stops with
an error and records nothing. Any other answer, an error included, ends the
wait. A failure that waiting cannot cure, such as a URL without `http://` or a
proxy that refuses, stops the run at once. A Ctrl-C ends the wait within
seconds, even in the middle of an attempt. The warmup is not a clip: it is
neither cached nor recorded, and it is not counted in `requests`. A run that has
nothing to send sends no warmup. The first clip is still flagged `cold` and left
out of the latency: a model that answered the warmup has not necessarily
answered a clip of real speech yet.

**Stopping**: a run can be stopped with Ctrl-C or SIGTERM
([D-021](../owner/directives.md)). The requests in flight finish and no other is
sent; the run prints `audiolab: evaluation stopped; run the same command again to
resume`, exits with code 130 and writes and scores nothing. The answers it
received are in the cache, so the same command sends only the rest. A checkpoint
the run is still loading is stopped with it. On a terminal, a progress bar
counts the clips sent; the `[i/N] <id> ok` lines stay the run's printed record.

Sending audio to a model is the slow part; scoring takes seconds. So the
answers are kept once, in a **transcript cache** shared by every suite, and a
suite's run directory is rebuilt from it on every run.

**The cache**: `$AUDIOLAB_RUNS/transcripts/<key>.jsonl`, one line per answer,
appended and flushed as it arrives (`id`, `audio_sha256`, `language`, `panel`,
`hypothesis`, `error`, `latency_s`, `attempts`, `cold`, `language_sent`,
`timestamp`). A torn last line left by a crash is skipped with a warning and its
clip is asked again. An answer is reused when the same clip, with the same audio, is
asked the same way again. "The same way" is the cache key, stored next to the
file as `<key>.key.json`:

| in the key (a change asks again) | not in the key (a change reuses the answers) |
| --- | --- |
| served model id, `api`, `request` options, `send_language` | config `name`, `server_label`, environment variable names |
| the server, as a keyed fingerprint (below) | the API key |
| a served checkpoint's **content** (the sha256 of its file, or its Hub id), its backend and its decoding settings | the path the file sits at, the host and port the server listened on |
| the clip id, the sha256 of its audio, the language asked | the panel or suite the clip is in |
| `CACHE_FORMAT` of `eval/cache.py`, bumped when the client changes what it sends | the git commit, the normalizer version, the metrics |

So adding a dataset or a panel costs only the clips never asked; `fast` clips
are never asked again by `full`, which contains them; a new commit or a
normalizer change sends nothing and only re-scores. A failed request is not an
answer: it stays in the file, so a flaky endpoint remains visible, and the clip
is sent again on the next run, the newest line winning. `--fresh` sends every
clip of the suite again (the old lines stay in the file).

What the key cannot see: a server upgraded behind the same URL, or a server
that refused the language field (the client then asks without it, and those
answers are cached like any other; the report counts them per panel). After
such a change, rerun with `--fresh`.

For a config with a `serve:` block the fingerprint is not about a URL at all: the
server is started for the run and its port is new every time, so what the
fingerprint covers is the checkpoint itself — the sha256 of the `.nemo` file, or
the Hub id when there is no local file — together with the backend and the
decoding settings. Two runs of the same file therefore reuse each other's
answers, on whatever port they listened, and a different checkpoint never reuses
them. A path is not part of it: moving or copying the file changes nothing, and a
rewritten file is a different model. What this cannot see: a Hub id is resolved
when the model loads, so the same id can mean different weights on different
days; keep a local `.nemo` when a score has to be pinned to exact bytes.

Otherwise the server is recorded as a **keyed fingerprint** of its root URL: HMAC-SHA256
with a random per-machine key in `<runs dir>/.identity-key` (created on first
use, mode `0600`). The URL and the API key are never stored, and the
fingerprint cannot be reversed by guessing the shape of a private address. It
lives only in the cache's key file, never in a run directory.

**The run directory**: `$AUDIOLAB_RUNS/eval/<model name>/<suite>/`, rewritten on
every run so it always describes the current panels. If the same config name now
asks differently (another served id, option or server), the run says so and the
directory shows the new answers; the old ones stay in their cache file. To keep
two views side by side, for example before and after `--fresh`, copy the
directory first.

- `records.jsonl`: one line per panel item, the cached answer with the panel it
  is scored in.
- `run.json`: the model config without secret values, the panel hashes, the
  audiolab and normalizer versions, the git commit that scored it, and the
  name of the cache file the answers came from. The config's optional
  `server_label` says in words which server answered.
- `summary.json`: every number at full precision. `audiolab score <run dir>`
  writes it again from the records alone, byte for byte.
- `report.md`: the same numbers rounded, plus what the run does not measure.

A record never holds the API key: an endpoint that echoes the request headers in
an error body is common enough that the client replaces the key value with
`<redacted api key>` before any message is written.

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

## 5. Text normalization (three views)

Model output and reference text never match character for character. Both sides
go through the same function, so a difference in a score always comes from the
text, never from the processing.

`NORMALIZER_VERSION` is recorded in every summary because changing a rule
changes scores, and the rules are not frozen while the project moves: **version 1
is the first version any score was published with, and every later change to a
normalization rule bumps it, however small**, so two summaries that carry
different versions were not produced by the same measuring instrument and do not
compare. Bumping the version is the point at which a comparison across versions
becomes a new experiment rather than a diff.

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

- **English** groups thousands with a **comma only** and writes the decimal with
  a dot: `1,500` is `one thousand five hundred`, `1,000,000` is `one million`, and
  `1.5` and `1.500` are decimals (`one point five`, `one point five zero zero`).
  A comma is read as a decimal when it does not group exactly three digits —
  `1,5` is `one point five`, because reading it as fifteen would be the worse
  mistake — so in English thousands are grouped in one way only;
- **French** writes the decimal with a comma and groups thousands with a dot:
  `1,5` is `un virgule cinq` and `1.500` is `mille cinq cents`;
- a mark can only group three digits if it sits between a leading group of one to
  three digits and a group of exactly three, and that leading group cannot start
  with `0`: no number is written `0 125`. `0.125` is therefore a decimal in both
  languages (`zero point one two five` in English, `zéro virgule un deux cinq` in
  French) and `3.141` is pi, not three thousand one hundred forty one;
- both languages also write grouped thousands with spaces (`1 500`,
  `12 000`), which are joined into one number;
- a quantity written against its symbol is one token, and the symbol may stand on
  either side or both: `21%`, `12,50 €`, `$3.50`, `90°`, `20°C`. The degree sign
  reads `degrees`/`degrés` (`36°` → `trente six degrés`), a currency symbol in
  front makes the reading a money one (`$1,500,000` → `one million five hundred
  thousand dollars`), and a unit this table does not know keeps its own symbol in
  the output rather than stopping the run;
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

**Reading a number never raises.** The functions run on whatever a model answered,
which is not a controlled input, so a number or a symbol no rule expected is kept
as it was written instead of stopping the run, and a number of more than thirty
digits is read digit by digit rather than as a cardinal (a cardinal costs one
step per group of nine digits, and an invented number of a few hundred digits
would exhaust the stack). The internal stand-in characters the tokenizer uses for
a decimal mark, a clock colon and a plus sign never reach an output.

The behaviour is pinned by two tests. A trap set of 307 pairs in
`tests/data/normalizer_traps.jsonl` (`ref`, `hyp`, `language`, `view`,
`expect_equal`), 249 of which must compare equal: a pair that must compare
*different* is as important as one that must match, and the file records both. It
is agent-written and has not been reviewed by the owner; every pair is asserted
by `tests/test_eval_normalize.py`. On top of it, a seeded fuzz of 3000 strings
over the alphabet a transcript really uses (digits, `, . $ € £ % ° h : / -`,
letters and spaces, in three shapes: noise, words, and a written number) checks
that every view of both languages answers something for any input and that the
`normalized` view is idempotent — normalizing a text twice gives the same words
twice. A failure is reproducible from the seed alone.

## 6. What each number means

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
| scoring errors | hypotheses that came back but that scoring could not read; they are counted, named in the report, and left out of every other number | a number the reader cannot verbalize |
| latency | median and p95 of the warm calls' **successful request**, cold excluded, plus mean and median RTF | 0.42 s for a 6 s clip = 0.07 |
| requests | how many HTTP requests the run sent, retries included, the warmup not | 60 items, 2 retries = 62 |

A request that failed and a hypothesis that could not be scored are different
things, and both are counted rather than dropped: the first never came back, the
second came back and could not be read. A scoring error costs its own utterance
only — the panel's numbers are computed over the utterances that did score, and
the report names the ids that did not, because a scoring error is a bug in the
reader and a summary that hid it would look complete while covering less than it
claims.

Zero-reference panels have no WER, so `n/a` is reported rather than 0: an error
rate over zero reference words does not exist, and a model that talks over
silence has not scored 0 %. The errors it made are still counted in the battery.

An utterance whose reference normalizes to nothing is the same problem one level
down, and it is answered the other way round, because the tail quantiles are
built on per-utterance rates: insertions over an empty reference count one error
per invented word, so such an utterance reads as far above 50 % WER — which is
what the sentence error rate already says about it. An empty reference answered
with nothing stays 0 %. At panel level the invented words stay in the numerator
and no words are added to the denominator, so a panel whose references all
normalize away still reports `n/a` rather than a rate invented from nothing.

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

## 7. Confidence intervals

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

**The interval is only as good as the number of groups behind it.** Below
`stats.MINIMUM_GROUPS_FOR_INTERVALS` (20) a percentile bootstrap over groups has
fewer distinct resamples than the interval has meaning, so the report writes
`few groups (n < 20): interval approximate` next to that panel, in `report.md`
and in the comparison. Groups are counted per dataset and per panel in
[datasets.md](datasets.md#panels-and-their-groups); `mls-fr` is the one that
matters here, 4 groups on its sealed panel and 14 on the others, so a French
interval is an indication of spread between speakers rather than a precise
bound. The two `servicenow-codeswitched-*` datasets have only two synthetic
voices each, so their group is the clip: their intervals have enough groups but
are optimistic, since clips of one voice are not independent.

## 8. What this does not measure

- long-form audio (more than one utterance per request);
- which word a code switch or an accent cost, and term recall and named
  entities: the switches and the vocabulary are scored inside the sentence, not
  as a list of words that had to be right;
- dictation itself: the code-switched panels are text written first and read or
  spoken afterwards, so a person thinking out loud and switching language
  mid-sentence is still not measured;
- a semantic score from a language model;
- local models on our own GPUs (phase 1b);
- accents beyond the African accent of `african-accented-fr-test` and the
  non-native English of `voxpopuli-en-nonnative`;
- the panels are small and were chosen by us, so absolute numbers use them as
  the population they are.

Noise, reverberation, music and technical talks have their own panels since
2026-11-13 (`noisy-fleurs-fr`, `noisy-fleurs-en`,
`noisy-servicenow-codeswitched-en-fr`, `nonspeech-music`, `nasa-en-test`,
`peertube-edu-captions-fr`), listed in [datasets.md](datasets.md).
