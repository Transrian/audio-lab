# Projects to watch

- Status: watch list; no project below has been run or evaluated here.
- Purpose: external projects that may matter to a problem family of this
  repository later. An entry is a pointer, not an adoption, a plan or an
  evaluation result. Work to try one goes to the [backlog](../backlog.md).

Each entry records what the project is, why it is relevant here, its license,
the hardware it needs and the date its sources were checked. Figures quoted from
a project are its authors' claims unless stated otherwise; nothing here was
measured with the [common evaluation](../stt/evaluation.md). Recheck an entry
before relying on it: young projects change quickly.

## Full-duplex speech

Full-duplex speech may follow as a problem family
([D-003](../owner/directives.md)); nothing is planned for it yet.

### Sophon-Realtime (Adventists AI)

- Links: [repository](https://github.com/adventists-ai/sophon-realtime),
  [CHANGELOG](https://github.com/adventists-ai/sophon-realtime/blob/main/CHANGELOG.md),
  [Tag model card](https://huggingface.co/adventists-ai/Sophon-Realtime-32B-Tag).
- Checked: 2026-11-14 (version 1.3.0; repository created 2026-11-04, first
  open release 1.0 on 2026-11-08, new versions almost daily since).
- License: Apache-2.0 for code and weights, per the repository and model card;
  each build's `NOTICE` lists third-party components.
- What it is: an open full-duplex voice model family. One 32B "thinker"
  ([DuplexJev](#duplexjev-adventists-ai)-32B-Para with reply writing merged in)
  listens continuously, decides at each pause whether the user has finished,
  drafts the reply and annotates it with sound tags (laughter, sigh, stress).
  Three builds differ only in how the reply becomes speech:
  - Tag: a streaming TTS based on Fun-CosyVoice3-0.5B performs the tags;
  - Flash: a lighter, faster speech model;
  - E2E: a talker of about 350M parameters generating continuous voice vectors.
- Features: barge-in, backchannels, zero-shot voice cloning from a few seconds
  of audio, OpenAI Realtime and GPT-Live compatible APIs served by vLLM-Omni,
  LoRA fine-tuning of the thinker with ms-swift.
- Languages, per the README's internal tests: Chinese and English fully
  supported; French, Italian, German, Spanish and Portuguese "ready for
  conversation"; Japanese and Russian on Flash only.
- Hardware: one 80 GB GPU per build. A 4-bit Flash variant uses about 30 GB;
  per its CHANGELOG it was not yet run on an actual RTX 5090, and 24 GB GPUs are
  not enough.
- Why watch it: an open, permissively licensed reference for the full-duplex
  family, built on the Qwen3-ASR encoder family this repository already
  evaluates for STT. Its data disclosure already yielded a
  [candidate source](../stt/sources.md#found-in-the-duplexjev--sophon-realtime-data-disclosure-2026-11-13).
- Caveats: benchmark figures are self-run by the authors; French quality is not
  documented beyond the summary above.

### DuplexJev (Adventists AI)

- Links: [repository](https://github.com/adventists-ai/duplexjev),
  [paper](https://arxiv.org/abs/2610.02638),
  [32B-Para model](https://huggingface.co/adventists-ai/DuplexJev-32B-Para).
- Checked: 2026-11-14.
- License: Apache-2.0 (repository).
- What it is: a speech decision model, the "ears" of Sophon-Realtime. A frozen
  Qwen3-ASR-0.6B encoder feeds a frozen LLM (Qwen3-VL-32B language model, or
  Qwen3-4B) through a trained connector. Questions declared at runtime (turn
  ended, real interruption or backchannel, emotion, stressed word, speaker
  gender) are answered as multiple choice from one next-token distribution, in
  a single forward pass, without transcribing.
- Hardware: about 10 GB for the 4B models, one 80 GB GPU for the 32B, per the
  README.
- Why watch it: turn-taking and barge-in decisions are the part of a
  full-duplex system closest to speech recognition, and the 4B models fit
  smaller GPUs.
