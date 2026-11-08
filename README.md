# audio-lab

A lab for speech experiments, written entirely by LLM agents and meant to be
read by humans.

The first project is a **better French speech-to-text model**, fine-tuned from
NVIDIA Parakeet with a rigorous evaluation method: frozen test sets, per-dataset
results with confidence intervals, and train/test leakage checks. Other problem
families (text-to-speech, full-duplex speech) may follow.

**Status:** phase 0 (foundation) is done; phase 1 (evaluation) is in progress.
See the [roadmap](docs/roadmap.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/Transrian/audio-lab.git
cd audio-lab
uv sync
uv run audiolab --help
uv run audiolab gpus          # list the GPUs audiolab can use
```

Every command that uses a GPU takes `--gpu`, either an index (`--gpu 1`) or a GPU
UUID prefix (`--gpu GPU-1a2b`). The default is GPU 0; one GPU is always enough.

## Score a model

`audiolab` scores any model served through an OpenAI-compatible transcription
API on frozen panels of French, English, code-switched, accented and non-speech
audio, and reports honest numbers (grouped confidence intervals, a paired
comparison, a repeatable summary).

```bash
export AUDIOLAB_DATA=/mnt/data/audio-lab/data    # datasets live outside the repo
export AUDIOLAB_RUNS=/mnt/data/audio-lab/runs    # runs too
uv run audiolab data fetch                       # prepare the nine datasets
uv run audiolab data panels fleurs-fr            # freeze one dataset's panels

export STT_BASE_URL=https://server.example/v1    # never written to disk
export STT_API_KEY=...                           # only if the server needs one
uv run audiolab eval --model configs/models/example.yaml --suite configs/suites/fast.yaml
uv run audiolab score runs/eval/example/fast     # rewrite the summary from the records

uv run audiolab compare runs/eval/model-a/full runs/eval/model-b/full
```

[Evaluating a speech-to-text model](docs/stt/evaluation.md) explains the panels,
every metric and what they do not cover;
[datasets](docs/stt/datasets.md) has one card per dataset, with its license, and
[sources](docs/stt/sources.md) registers every data source considered, with its
status.

## How this repository is built

The owner writes no code. LLM agents write everything, under rules that keep the
owner in control:

- **[Owner directives](docs/owner/directives.md)**: what the owner explicitly
  asked for, in wording the owner validated. Agents cannot change them.
- **[Decisions](docs/decisions/README.md)**: everything an agent inferred or
  designed. A decision binds only once the owner accepts it, so the line between
  "what the owner said" and "what an agent understood" stays visible.
- **[Agent rules](AGENTS.md)**: what agents must read first, which tasks need the
  most capable model, and what may never be committed.
- **CI** checks every push: lint, tests, no audio or secrets in the repository,
  and an owner-approval marker on protected files.

## Layout

```
src/audiolab/     the Python package (one command: audiolab)
  core/           shared building blocks: configuration, GPU selection, audio
  data/           dataset preparation and frozen panels
  eval/           text normalization, metrics, statistics, runs and reports
configs/          one file per experiment: models, suites, frozen panels
tests/            fast CPU tests
docs/             roadmap, owner directives, decisions, how-to pages
scripts/          repository checks
```

Datasets, audio and model weights are never stored in the repository.

## License

Code: [MIT](LICENSE). Published models carry their own license, stated on their
model card.
