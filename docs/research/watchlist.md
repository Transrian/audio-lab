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

### Haru-4B (higurashi)

- Links: [repository](https://github.com/higurashi-ai/Haru),
  [model card](https://huggingface.co/higurashi-ai/Haru-4B),
  [technical blog](https://blog.higurashi.ai/melte/haru4b/).
- Checked: 2026-11-15 (repository and weights published 2026-11-12; packages
  `haru-speech` and `haru-speech-vllm` 0.1.1 on 2026-11-13; model revision
  `8d6e83bc`).
- License: Apache-2.0 for code and weights, per the repository and model card;
  `THIRD_PARTY_NOTICES.md` lists the included third-party work.
- What it is: a native full-duplex speech-to-speech model, released as a
  pretrained base model for research, not an assistant. A Qwen3.5-4B text
  backbone (about 5B parameters in total, bfloat16) runs at 7.5 Hz: every
  133 ms, the user's audio latent, the model's previous audio latent and its
  previous text token are summed into one position, and the model predicts its
  next text token and next audio latent. Audio is a continuous 64-dimensional
  latent of the causal VibeVoice acoustic tokenizer (24 kHz), produced in one
  sampling step by a small MeanFlow MLP head; there are no discrete audio
  tokens.
- Training, per the blog: over 40 million hours of real speech in more than
  160 languages, explicitly without LLM-generated dialogues or TTS speech.
  Stage 1 mixes ASR, speech and text continuation and text-only rehearsal;
  stage 2 uses over 13 million hours of dialogues split into two speaker
  streams by the authors' separation model, in samples up to 60 minutes. The
  data sources themselves are not named.
- Features: backchannels and overlapping speech, a character card as text
  prompt, zero-shot voice from a 3 s to 3 min reference clip, PyTorch, MLX
  (macOS app) and vLLM-Omni implementations.
- Languages: English, Spanish, Portuguese, German, French, Chinese, Japanese,
  Russian and Korean as primary languages.
- Hardware: about 12 GB of GPU memory for the PyTorch demo, per the README; a
  fix in 0.1.1 targets GPUs with less shared memory such as the RTX 4090. The
  authors report about 65 ms end-to-end latency on an M1 Max (int8) plus up to
  133 ms of framing.
- Why watch it: an open, permissively licensed full-duplex model small enough
  for a 24 GB card, with French among its primary languages. Two companion
  tools may matter for data preparation:
  [Sidon-Triton](https://github.com/higurashi-ai/Sidon-Triton) (speech
  restoration, MIT) and
  [FastDialogueSidon](https://github.com/higurashi-ai/FastDialogueSidon)
  (two-speaker dialogue separation, Apache-2.0).
- Caveats: no quantitative benchmark is published yet, only demos; a technical
  report and training code are announced. The text stream carries the model's
  own words; no API for transcribing the user is documented. Base model:
  facts and instruction following are unreliable, per the authors.
