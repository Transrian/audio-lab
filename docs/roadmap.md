# Roadmap

The first project is a better French Parakeet speech-to-text model (D-005).
Each phase ends with a check that anyone can verify.

| Phase | Content | Done when | Status |
|---|---|---|---|
| 0. Foundation | Repository skeleton, README, agent rules, directives, decisions, CI. | CI is green on `main`; `uv run audiolab --help` works. | done |
| 1. Evaluation | Transcribe with a local model or any OpenAI-compatible endpoint; French/English text normalizer; WER with confidence intervals **per dataset**; a fast suite and a full suite, each sized for the smallest difference it must detect. | The base Parakeet model is scored on public French and English suites; running it twice gives identical numbers. | done |
| 2. Data | One card per dataset (license, hours, speakers); manifests; leakage audit in CI; sealed final test set ([decision 0002](decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md)). | Every dataset used has a card; the audit passes. | next |
| 3. Baseline training | Reproduce the known recipe on public data ([decision 0004](decisions/accepted/0004-parakeet-fr-baseline-recipe.md)); selectable GPU; safe stop and resume. | The model is compared with the base model on the fast and full suites, with confidence intervals. | in progress |
| 4. Dataset by dataset | Add one dataset at a time and measure its effect. | A table shows each dataset's effect; the best mix is chosen **before** the sealed set is opened. | — |
| 5. Publication | Only when the owner asks, after testing the model (D-006). | — | — |

Phase 3 is *in progress*, not done: `audiolab train` fine-tunes a checkpoint,
keeps one resume slot so a stopped run goes on, and writes a checkpoint a run can
be scored from ([fine-tuning](stt/training.md)); no model has been trained for
real yet, and nothing has been compared with the base model.
