# Owner directives

This file records what the repository owner has explicitly asked for. It is the
highest authority in the repository.

- Each directive is faithful to the owner's meaning, in neutral wording that the
  owner has validated (D-012).
- A directive applies only within its **scope**. Extending a directive to another
  scope is an inference: write it as a proposed decision in
  [`../decisions/`](../decisions/README.md), do not apply it.
- Agents never add, reword or remove a directive on their own. A change requires
  an explicit request from the owner and a commit carrying an `Owner-Approved:`
  trailer (see [decision 0005](../decisions/accepted/0005-owner-approved-trailer.md)).

Scopes in use: `global` (the whole repository), `stt` (speech-to-text).

| ID | Scope | Date | Directive |
|---|---|---|---|
| D-001 | global | 2026-11-07 | Public repository `audio-lab`, MIT license. Code and documentation are written by LLM agents; the owner writes no code. |
| D-002 | global | 2026-11-07 | Agents of varying capability maintain the repository (the most capable for difficult tasks, less capable ones for routine work); its structure accounts for this. |
| D-003 | global | 2026-11-07 | The repository covers several problem families (speech-to-text, text-to-speech, possibly full-duplex later). |
| D-004 | global | 2026-11-07 | Inspired by the owner's earlier private work; earlier projects are not migrated as-is. |
| D-005 | stt | 2026-11-07 | Redo the French Parakeet fine-tuning with a more rigorous method, including per-dataset evaluation and fast tests. |
| D-006 | global | 2026-11-07 | A model is published (on Hugging Face) only at the owner's explicit request, after the owner has tested it. |
| D-007 | global | 2026-11-07 | Everything in the repository is in English. |
| D-008 | global | 2026-11-07 | The GPU used is selectable; everything runs on a single GPU. |
| D-009 | stt | 2026-11-07, amended 2026-11-08 | French is the priority. English is kept, and its WER is reduced where this is simple. Other languages are out of scope. The target use is mainly dictation by a French developer to coding agents and language models: French with many English words and technical terms, switches between French and English within a sentence, passages entirely in English, and English spoken with a French accent. |
| D-010 | global | 2026-11-07 | Scripts stay very simple and build on a shared base; using the repository never requires copying many files. Documentation is very clear to human readers. |
| D-011 | global | 2026-11-07 | Owner directives are kept distinct from agent inferences. A directive applies only within its scope. |
| D-012 | global | 2026-11-07 | Directives are recorded faithful to the owner's meaning, in neutral wording validated by the owner. |
| D-013 | global | 2026-11-07 | The repository never contains voice recordings, dictations, or personal audio or transcripts. |
| D-014 | global | 2026-11-07 | Agents commit directly to `main`; no pull requests. |
| D-015 | stt | 2026-11-07 | The evaluation can run against any model served through an OpenAI-compatible API, without installing that model's dependencies. |
| D-016 | stt | 2026-11-07 | Local models are evaluated the same way: a small, temporary server exposes them through an OpenAI-compatible API, so that supporting a new model family (for example a fine-tuned Qwen ASR) requires few changes. |
| D-017 | global | 2026-11-08 | Expensive results are cached and reused as long as what produced them has not changed, so that adding data or changing code recomputes only what is affected. |
| D-018 | stt | 2026-11-08 | Every data source considered is recorded with its status (used, evaluation only, candidate, abandoned, rejected), its license and the reason. An entry can be rewritten, but the reason a source was abandoned or rejected stays readable in the current file. |
| D-019 | stt | 2026-11-08 | Training data must carry a license that allows publishing the trained model: public domain, CC0, CC-BY (any version), Apache-2.0, MIT or Licence Ouverte. Share-alike, non-commercial, no-derivatives and unknown licenses are not trained on. |
| D-020 | global | 2026-11-08 | Models served by a hot-swapping server are used in batches: all the requests for one model are sent before moving on to the next, so that the server unloads and reloads models as rarely as possible. |
| D-021 | global | 2026-11-09 | Every long task (training, evaluation, data preparation, labelling, downloads) can be stopped and resumed: on Ctrl-C or SIGTERM it finishes its current step (for training, the current gradient update), saves its state and releases its resources quickly, and running the same command again resumes from that state. A long task shows its progress (a progress bar such as tqdm, where applicable). |
| D-022 | global | 2026-11-09 | Every request to an OpenAI-compatible API can be sent N at a time, to go faster; N is configurable and defaults to 1, because what an engine supports varies. Before its first request, a run sends a warmup request and waits until the model answers, since an external model may be slow to start. |
| D-023 | global | 2026-11-09 | Any directive can be changed for a technical reason: the agent proposes the change with its justification, and the owner validates it. A directive that takes much work to respect, or forces a solution much more complex than it would otherwise be, signals a missing or wrong directive: the agent proposes a change instead of working around it. |
