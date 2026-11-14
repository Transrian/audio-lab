# Label audit: the error of the agreement pipeline's labels

**Status:** the measurement behind "Label error of the agreement pipeline" in
[`docs/stt/datasets.md`](../../docs/stt/datasets.md) (Training sources). The
script asks two served models for transcripts of prepared clips that also hold
a human label, applies the talk pipeline's own agreement rule, and scores the
result against the human text. It reads the prepared datasets and writes only
to its own cache directory.

## What it measures

The talk pipeline keeps a clip when Whisper large-v3 and Qwen3-ASR agree
within 15 % of the normalized words (`MAX_DISAGREEMENT` in
[`audiolab.data.talks`](../../src/audiolab/data/talks.py)), and its label is
the Qwen text. No talk source has a human label, so that error was never
measured. Two prepared sets do:

| arm | dataset | clips | human label |
|---|---|---|---|
| English | `nasa-en-train` | 3 000 of 11 956 | NASA's own near-verbatim transcripts |
| French | `peertube-edu-captions-fr` | 1 000 of 4 367 | the ministry's human captions |

For every sampled clip the script gets the two models' transcript of the
clip's own audio, applies the agreement rule — imported from the pipeline, not
reimplemented — and scores the kept and the rejected against the human text
with the project's scorer (`audiolab.eval.metrics.counts`, the normalized
view, the run's normalizer version).

## Environment and inputs

From the repository root, use the default Python 3.12 environment (`uv sync`),
and load the data environment:

```bash
set -a; . ./.env; set +a
```

Inputs are read-only: the two prepared datasets in `$AUDIOLAB_DATA`, and the
two home vLLM STT models (`stt-v-whisper-large-v3`, `stt-v-qwen3-asr-1.7B`)
through their OpenAI-compatible endpoints. Help needs neither data nor
environment variables:

```bash
uv run python experiments/label-audit/audit.py --help
```

## Reproduce

The sample is fixed once and for all: `random.Random(20261110).sample` over
the manifest's ids sorted, written to
`$AUDIOLAB_CACHE/label-audit/<dataset>-sample.json` with the dataset, the
seed, the count and the whole id set as its key. A dataset prepared again with
other clips draws again; a re-run on the same dataset reads the draw back.

```bash
# 1. Ask both models for every sampled clip (a few minutes per arm; one
#    model's requests before the other's, D-020). Answers are cached per clip
#    under $AUDIOLAB_CACHE/label-audit/<model>/<clip id>.json.
uv run python experiments/label-audit/audit.py transcribe nasa-en-train
uv run python experiments/label-audit/audit.py transcribe peertube-edu-captions-fr

# 2. The numbers: kept vs rejected, by base_wer bucket, per-clip rows written
#    to $AUDIOLAB_CACHE/label-audit/<dataset>-scored.jsonl.
uv run python experiments/label-audit/audit.py score nasa-en-train
uv run python experiments/label-audit/audit.py score peertube-edu-captions-fr

# 3. The 30 classified differences of the kept clips, for the caveat.
uv run python experiments/label-audit/audit.py examples nasa-en-train
```

Every step is idempotent: `transcribe` asks only for what is not cached, and
`score` and `examples` recompute from the cached answers and the manifests.

## Cache key (D-017)

One answer is one JSON file per clip and model, holding the model id, the
language sent, the request extras and the audio's `sha256`. A reuse is trusted
only when all of them match what the current caller would send: another model
id, a clip prepared again with other audio, or another language is asked
again. The shared helper is
[`audiolab.data.label_audit`](../../src/audiolab/data/label_audit.py); the
`nasa-en-train-qwen` fetcher
([`audiolab.data.relabel`](../../src/audiolab/data/relabel.py)) reads the same
Qwen answers, so it never asks for what this script already asked. The key
behaviour is tested in
[`tests/test_data_label_audit.py`](../../tests/test_data_label_audit.py).

## How the numbers are read

- **Kept WER** is the error of the labels the pipeline would keep, against the
  human text, corpus-wide (errors over reference words, as the panels report
  WER).
- **Rejected** is the share the agreement rule would have dropped, and the WER
  its Qwen labels would have had; a rejected clip's label is never used, the
  number says how wrong the rule is to drop them and to keep the rest.
- **The `base_wer` split** (English arm) asks whether the rule's error
  concentrates where the base model already does well — the clips a
  fine-tune learns least from.
- **The caveat:** the human transcripts are near-verbatim, not verbatim — they
  drop some fillers and repetitions, and they spell names and read numbers
  their own way. Part of the measured error is that style difference, not a
  word heard wrong. The `examples` step classifies the kept clips'
  differences (filler insertions, numbers, name spellings, other) and prints
  30 of them; the normalizer already drops the bare interjections (uh, um,
  hmm) and reads numbers as words, so those categories are what the WER still
  sees.
