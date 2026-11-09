# Low-bit audio models: QAT and QAD feasibility

- Status: documentary feasibility study; no local quantization experiment run.
- Updated and sources accessed: 2026-11-09.
- Work item: [N-08 in the backlog](../backlog.md#not-started--deferred).
- Scope: STT and TTS; prioritize deployable 4-bit weights, with 8-bit as a fallback.

## 1. Executive conclusion

**Worth testing for compression; not a demonstrated way to improve the original
model's quality.** Speech research already supports low-bit recognition and
synthesis. The important distinction is between recovering a quantized model's
quality and beating its floating-point reference.

The recommended order is **deployment compatibility → PTQ → selective QAT →
QAD if needed**. Do not train a quantizer before proving that its exact format
can be exported and executed by the intended inference engine. If a currently
served artifact is already 4-bit PTQ, same-format QAT/QAD targets **better quality
at that size**, not another fourfold reduction. Establish the actual baseline
precision before estimating any additional saving.

| Target | Feasibility judgment, not a measured project result | Recommended first direction |
| --- | --- | --- |
| Parakeet TDT 0.6B v3 | Existing mostly-4-bit PTQ runtime; exact French quality and deployable QAT recovery remain unverified | Compare F16/Q8_0/Q4_K in a pinned compatible runtime before writing custom QAT; retain architecture-specific precision guards |
| Transformer-decoder ASR, including Qwen3-ASR | Existing mixed-precision weight-only routes; no established French QAT recovery result | Quantize the large decoder/body first, preserving audio/output precision guards; include long-audio checks |
| TTS | Credible for selected modules; higher risk of audible degradation than a model-size headline suggests | Quantize the large backbone first; keep codec/vocoder and waveform-facing layers at reference precision initially |
| End-to-end W4A4 | Higher-risk second-stage research | Not the first acceptance target |
| NVFP4 | Conditional on the actual deployment GPU, kernels and architecture support | Optional Blackwell branch, not a prerequisite for useful INT4 deployment |

These are engineering recommendations, not accepted decisions or authorization
to run training. They do not change the STT baseline in
[decision 0004](../decisions/accepted/0004-parakeet-fr-baseline-recipe.md), resume
[paused work](../backlog.md#paused), or select a TTS project.

## 2. What the terms mean

| Term | What changes | What it does not establish |
| --- | --- | --- |
| PTQ: post-training quantization | Quantize existing weights, sometimes with calibration/reconstruction, without a normal training campaign | That naive rounding is the best PTQ method, or that quality will be unchanged |
| QAT: quantization-aware training | Optimize weights while exposing the forward pass to the intended quantization error | That optimizer states, gradients and the whole training process use 4 bits |
| QAD: quantization-aware distillation | QAT with a teacher-matching objective; often the original floating-point model teaches a same-architecture quantized student | That the student has fewer parameters, or that a text-logit KL objective transfers unchanged to every speech architecture |
| Structural distillation | Train a smaller architecture from a teacher; may be combined with quantization | That this is the same experiment as compressing an unchanged architecture |
| QLoRA | Keep a quantized backbone and train adapters to reduce fine-tuning memory | That the merged/exported model is QAT-trained for the final deployment grid |
| Low-precision training arithmetic | Actually execute some training operations at reduced precision | A consequence of merely enabling fake quantization |

QAD is interpreted here as *quantization-aware distillation*. It is a form of
QAT, not a mutually exclusive alternative. NVIDIA's current guide explicitly
makes this distinction and says that QAD needs extra teacher memory and compute
[S4]. Standard fake-QAT keeps trainable floating-point weights; adding fake
quantization alone is not a training-memory saving [S5].

`W4A16` means 4-bit weights and 16-bit activations **for the affected operations**;
`W4A8` must additionally specify whether A8 is INT8 or FP8. A quantized attention
projection does not imply a 4-bit attention softmax or a quantized KV cache.
Always publish the module/precision map, not just “4-bit model”.

## 3. What primary evidence actually supports

All numbers below are **reported by the source authors**, not reproduced in this
repository. Quality units, reference precision and unquantized components matter.
“Lossless” in a paper is not proof of statistical equivalence on French dictation.

### 3.1 Conformer ASR: strong motivation, not an exact Parakeet result

Ding et al., *4-bit Conformer with Native Quantization Aware Training* [S1],
Table 1, report the following for a 118M-parameter Conformer on English
LibriSpeech:

| Configuration in the paper | test-clean WER (%) | test-other WER (%) | Reported size (MB) |
| --- | --- | --- | --- |
| FP32 weights / FP32 activations | 2.0 | 4.4 | 475 |
| INT4 weights / FP32 activations (`I4W`) | 2.0 | 4.4 | 82 |
| INT4 weights / INT8 activations (`I4WI8A`) | 2.1 | 4.4 | 82 |
| INT4 weights / INT4 activations (`I4WA`) | 3.1 | 8.2 | 82 |

The quantization targets the encoder; decoder and convolutional kernels are
excluded. The often-quoted 5.8× compression is relative to **FP32**, not BF16.
The weight-only result is **W4A32**, not an experimental demonstration of W4A16.

The same paper's production experiment is a useful warning. On its English
Voice Search second pass, Table 2 reports:

- FP32: **5.7% WER**, 460 MB;
- INT8 PTQ: **5.8%**, 115 MB;
- INT4 PTQ: **16.0%**, 61 MB;
- INT4 QAT: **6.2%**, 61 MB;
- selected mixed INT4/INT8 layers (`E10`): **5.9%**, 94 MB.

Here QAT is much better than INT4 PTQ, but uniform INT4 still does not recover
the reference. Keeping only the first and last layers at higher precision was
not sufficient; sensitivity must be measured, not guessed. Training was native
QAT from scratch, including an approximately 400,000-hour production dataset.
This is **not** evidence that a short post-fine-tuning run on Parakeet will give
the same outcome.

Rybakov et al., *2-bit Conformer quantization* [S2], provide corroborating
weight-only results and another scale-dependent warning: their large-data
Table 2 has **6.0% FP32 WER vs 6.3% INT4**, and reducing to INT2 degrades further.
Their encoder-only experiments exclude depthwise convolutions. They do not
validate NVIDIA Parakeet, French, TDT duration decoding, or this project's stack.

**Implication:** mostly 4-bit Parakeet weights are a reasonable experiment;
all-4-bit activations are a materially different and riskier experiment. No exact
Parakeet TDT v3 French QAT/QAD reproduction has been established by this review.

Speech distillation is not merely an analogy with LLMs. **DQ-Whisper** [S12]
combines decoder quantization, teacher-logit matching and intermediate-layer
matching. Its main joint-distillation result is **8-bit**, not 4-bit; the teacher
and student may also differ in size. Its compression factors therefore cannot be
attributed to bit width alone.

Its Common Voice Table 4 includes French. For the fine-tuned Whisper-small
reference, French WER is **23.1%**, versus **25.4% with 8-bit quantization** and
**30.6% with naive 4-bit quantization**; four-language average WER is
16.5%, 17.9% and 23.1%, respectively. These are a different setup from the
Conformer experiment, but a useful counterexample to assuming that INT8 is
always lossless or that naive 4-bit PTQ is an adequate baseline for all methods.
The paper does not validate 4-bit QAD for Parakeet.

### 3.2 TTS: compression is real, quality and runtime trade-offs are real too

BitTTS [S3] studies a small, convolutional JETS-derived acoustic model with a
HiFi-GAN vocoder, trained on English LibriTTS-R. Its Table 1 reports:

| Configuration | Reported size (MB) | Naturalness MOS | CPU RTF |
| --- | --- | --- | --- |
| FP32 reference | 25.66 | 3.75 ± 0.09 | 0.042 |
| 4-bit acoustic model; reference vocoder | 11.40 | 3.34 ± 0.10 | 0.047 |
| 4-bit acoustic model and vocoder | 8.87 | 2.96 ± 0.11 | 0.068 |

These are **not near-lossless quality numbers**. Lower RTF is better: these
quantized variants are slower in the reported Apple M1 Pro CPU experiment.
The study uses fake QAT, applies quantization to Conv1d layers, uses 8-bit inputs,
and stores those low-bit weights in **INT8**, not tightly packed INT4. Its
special weight-indexing compression is for the ternary variants. The headline
83% size reduction concerns a **1.58-bit indexed model vs FP32**, not generic
4-bit GPU deployment. Listening involved 30 utterances and 15 raters.

The final waveform-facing convolution was excluded even when quantizing the
vocoder. Leaving the vocoder unquantized substantially helped quality. This is
useful evidence for selective precision, not proof for Pocket TTS, Kyutai's
larger streaming TTS or Qwen3-TTS, and not French quality evidence.

### 3.3 Recent speech-DiT PTQ: encouraging, with important baseline caveats

TTS-Q [S10], an October 2026 preprint, supplies more relevant evidence for
flow/diffusion TTS than the convolutional BitTTS experiment. Its PDF/HTML body
and abstract metadata disagree in title/method summary; the following refers
to the cross-checked PDF body, not an independently reproduced implementation. It quantizes **DiT
linear layers** to logical W4A4, optimizing transforms, scales and clipping while
keeping pretrained generator weights fixed. It combines reconstruction with
teacher-generated speaker-identity supervision: distillation is not limited to
next-token KL, and PTQ can include optimized calibration rather than just rounding.

For LongCat-AudioDiT 1B on Seed English, Table 1 reports **2.403% → 2.475% WER**,
speaker similarity **0.7623 → 0.7475**, and predicted UTMOS **3.789 → 3.730**.
These are promising point estimates, not a French listening-equivalence result.

Section 4.5 reports **1.74–3.10× complete-pipeline file compression** against a
reference using FP16 storage for the affected weights. On 60 warm H100 conditions,
LongCat runs **1.66–2.34× faster than its native FP32 reference**, not FP16. Its
logical four-bit codes execute in **INT8 containers**, not native packed INT4
kernels. F5-TTS instead uses floating-point W4A4 fusion: its RTF worsens from
**0.067 to 0.145**, and allocated memory from **0.83 to 0.95 GiB**, against native
FP16. The main quality comparison uses simulated quantization; a smaller deployed
quality audit is reported separately.

**Implication:** there is credible low-bit flow/diffusion TTS research and a
speech-specific teacher signal. Neither the large-model speedup nor the
quantization recipe is automatically transferable to a small Pocket TTS model,
French speech, a different GPU, or an FP16 baseline. This is not an NVFP4 result.

### 3.4 NVIDIA QAD: a relevant method, not speech validation

NVIDIA's NVFP4 QAD report [S4] recovers quality for LLMs and a VLM by matching the
original teacher's output distribution. This motivates a speech experiment;
it does **not** establish preserved pronunciation, TDT alignment, voice identity
or flow-model stability. Its robustness to synthetic or random text tokens must
not be generalized to arbitrary audio calibration or distillation inputs.

At the same architecture and deployment precision, QAD does not itself create a
faster inference graph than PTQ/QAT. Its purpose is better quality at that graph's
cost. If improvement over the floating-point model is claimed, include a
compute/data-matched floating-point fine-tuning control: otherwise extra training
or teacher knowledge is confounded with quantization.

### 3.5 Gemma 4 31B: a useful QAT reference, not an audio-model result

Google's official QAT release [S11] includes
[Gemma 4 31B QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf),
as well as unquantized QAT weights and W4A16 compressed-tensors artifacts. The
release explicitly targets a deployment quantization scheme and describes
quality close to BF16. This is a concrete example of the desired product:
**same architecture, adapted to low-bit deployment**, not a smaller distilled
architecture and not NVFP4.

The general model-card benchmark table is not a paired QAT-vs-BF16 ablation and
must not be presented as one. The 31B variant supports text and image inputs,
not native audio; the card lists audio capability for other family sizes.
Consequently, this example motivates the method but does not validate ASR/TTS.
Short-context comparisons also cannot establish long-context or long-generation
quality. For audio, explicitly test duration and streaming state rather than
assuming that text-context behavior predicts an acoustic model's behavior.

## 4. Practical deployment constraints

### 4.1 Training support is not serving support

TorchAO documents a prepare → train → convert workflow for INT4 weight-only and
INT8-dynamic-activation/INT4-weight QAT [S5]. Its basic recipe replaces
`torch.nn.Linear`; this is not automatic coverage of Conformer convolutions,
LSTMs, TDT decoding or a complete TTS codec.

Model Optimizer exposes QAT/QAD and mixed-precision recipes [S4, S6]. Its checked
Hugging Face QAT example is a **text-model workflow**, not a verified drop-in
trainer/exporter for Parakeet TDT or Qwen3's speech variants. A library accepting
PyTorch modules is weaker evidence than an exported speech model running with
the correct kernels.

TensorRT distinguishes INT4 **weight-only** quantization from FP4, and supports
INT8 as a separate path [S7]. ONNX Runtime's block-wise 4-bit route targets
supported constant-weight MatMul/Gather operations, not arbitrary Conv layers;
that leaves unconverted encoder/vocoder operators at higher precision and
requires checking the execution provider [S19]. ONNX export of an encoder, or
an FP32 TensorRT encoder build, does not establish an INT4 end-to-end speech
runtime [S8].

The project has no low-bit implementation in its
[documented training interface](../stt/training.md). Its
[pinned model stack](../../requirements/nemo.txt) is not the same as current
upstream development documentation. Any prototype needs a separately pinned,
isolated environment; do not upgrade the established environment silently.
The existing [GGUF backlog item](../backlog.md#not-started--deferred) has not
selected a universal exporter or target runtime.

### Existing low-bit audio runtimes make PTQ the practical first step

CrispASR's current upstream documentation [S13] explicitly supports F16,
Q8_0, Q5_K and Q4_K for **Parakeet** and **Qwen3-ASR**, plus Q4_K for
**Qwen3-TTS**. Its GGUF quantizer supports importance-matrix calibration and
per-tensor precision overrides. This is direct evidence of an existing
**PTQ/export/inference route**, not a claim that custom QAT is already integrated
or that the installed runtime matches upstream `main`.

| Candidate | What is actually documented | Remaining gate |
| --- | --- | --- |
| Parakeet TDT | GGUF weight quantization and a Q4_K runtime; norms/biases and architecture-specific tensors retain higher precision | Convert the selected fine-tune, pin the runtime, and measure French/English quality; no exact French QAT result established |
| Qwen3-ASR | Q4_K decoder/body with precision guards; maintainer regression notes report loops/empty output when audio towers are quantized below Q8 on long clips | Preserve the audio frontend and guarded output tensors; test long durations as well as short WER panels |
| Qwen3-TTS | Q4_K backend listed as supported | Inspect talker/code-predictor/codec precision and evaluate pronunciation, listening quality and actual memory |
| Pocket TTS | Upstream optional CPU INT8 path; multilingual checkpoints and sizes differ by version [S16] | Pin the French checkpoint and protected layers; no demonstrated French 4-bit quality equivalence |

audio.cpp separately documents Qwen3 ASR/TTS GGUF conversion to Q4_K and other
formats, while preserving lookup tables and unsupported shapes at higher
precision [S15]. Its Qwen3-ASR session options distinguish audio-encoder storage
from decoder storage; do not infer the loaded compute path from the converter's
format list alone. The documentation warns that quantization can harm automatic
language detection even when transcription with a known-language hint remains
good. Keep hint/chunking policies identical between float and quantized arms;
automatic language detection is an additional diagnostic, not a reason to
silently improve only the quantized arm. GGUF layouts are runtime-specific:
audio.cpp files are not automatically interchangeable with llama.cpp files.

These are maintainer-reported compatibility/regression observations, not an
independent multilingual benchmark. Generic GGUF K-quants also have row-alignment
fallbacks, so a file named Q4_K can contain Q4_0, Q8 or floating-point tensors.
Inventory the exported artifact. Do not collect calibration through an endpoint
that records private dictation; use permitted non-evaluation inputs in a separate
approved calibration run.

A separate English streaming-ASR study [S14] also includes Parakeet v3 ONNX
INT8/INT4 comparisons. It is **PTQ evidence**, not QAT/QAD or French evidence.
Its exported ONNX and original PyTorch baseline scores differ substantially;
that is an additional reason to compare floating-point and quantized artifacts
in the same target backend rather than attributing every difference to bit width.

**Practical priority:** first compare F16 → Q8_0 → Q4_K in an already compatible
runtime. There is no reason to implement a custom NeMo quantization/export stack
before testing this cheaper path. A format-aware QAT experiment remains a
separate, higher-effort question if PTQ misses the approved margin.

For a TTS pilot, **Qwen3-TTS-12Hz's large autoregressive backbone** [S17] is the
more direct match for existing packed-weight 4-bit runtimes. Its 12Hz codec path is
not the same topology as a diffusion TTS benchmark; preserve its codec and
residual-code predictor initially. **Pocket TTS** [S16] is a smaller,
continuous-latent model with a flow-style sampling head, not a discrete-token
softmax throughout. Its English technical report predates the later multilingual
release. An INT8 CPU comparison is a sensible lower-cost control, but neither
that feature nor a generic TorchAO INT4 API establishes preserved French output.
For a 100M-class model the ideal 16-to-4-bit weight saving is only 150 MB; this
makes speed/quality and engineering cost especially important. These are pilot
recommendations, not a TTS selection or a license approval.

### 4.2 INT4, NF4 and NVFP4 are not interchangeable

- **INT4 weight-only:** useful without native FP4 arithmetic if a compatible
  packed-weight kernel dequantizes tiles for higher-precision computation.
  Lack of native NVFP4 is not an argument against all 4-bit deployment.
- **NF4:** a nonlinear codebook format associated with low-memory fine-tuning;
  not another spelling of integer INT4 or hardware-native NVFP4.
- **NVFP4:** FP4 E2M1 values with fine-grained scaling, intended for appropriate
  Blackwell kernels. It does not mean “any 4-bit checkpoint”.
- **MXFP4:** another microscaling scheme; different scaling and block semantics.
  Do not substitute it for NVFP4 without retraining/revalidation as appropriate.

The [Model Optimizer feature matrix][S6] lists W4A16 for Ampere and later,
versus FP4 for Blackwell and later. These are **format/backend capabilities**,
not guarantees for every audio architecture. The exact deployment GPU and
runtime must be identified separately from the GPU used for training.

| Generic deployment hardware | What the evidence permits saying | What remains model/runtime-specific |
| --- | --- | --- |
| Ampere, such as RTX 3090 | No native NVFP4 path in NVIDIA's matrix; weight-only INT4 remains useful with compatible kernels | Actual operator coverage, resident memory and latency of the audio runtime |
| GeForce Blackwell, such as RTX 5090 | Native FP4 is real; NVIDIA demonstrates TensorRT FP4 for FLUX, and TensorRT-LLM lists NVFP4 for SM120 [S18] | That does not make Parakeet, Qwen3-ASR or Qwen3-TTS supported NVFP4 models; text Qwen3 support is not speech-model support |
| Unspecified other GPU or CPU | Do not infer support from the training machine | Identify the device, instruction set, execution provider and loaded artifact |

Consumer and data-center Blackwell have different support rows. NVIDIA's
non-LLM FLUX example also disproves a blanket claim that FP4 is intrinsically
LLM-only; applying that custom export/kernel path to audio is still engineering
work, not an existing validated recipe [S18].

For a deployment target without native NVFP4, prioritize an available INT4
weight-only path; retain INT8 if it gives the better quality/latency trade-off.
Simulating a format during QAT is a different capability from accelerating that
format at inference: successful fake-NVFP4 training is not proof of native
serving support. Do not train specifically for NVFP4 and then assume an
INT4/GGUF conversion preserves the trained quantization behavior.

### 4.3 How much memory could actually be saved?

Ideal **weight payload only**, decimal GB, excluding scales, zero-points,
unquantized modules, buffers and all runtime memory:

| Parameter count | FP16/BF16 (2 bytes/weight) | INT8 (1 byte) | Packed 4-bit (0.5 byte) |
| --- | --- | --- | --- |
| 0.1B | 0.20 GB | 0.10 GB | 0.05 GB |
| 0.6B | 1.20 GB | 0.60 GB | 0.30 GB |
| 1.7B | 3.40 GB | 1.70 GB | 0.85 GB |

Thus 0.6B has an ideal saving of **0.90 GB vs 16-bit**, not several GB of
weights. Smaller TTS models offer less absolute saving, which can change the
engineering return. Actual parameter counts and quantized fractions must be
inventoried rather than inferred from rounded model names.

If 90% of parameters use packed 4-bit and 10% remain 16-bit, the average is
**5.2 bits/weight**, already only about **3.08× weight compression vs 16-bit**
before metadata. NVFP4's per-16-value FP8 scale alone adds 0.5 bit/value, giving
4.5 bits/value before the additional tensor scale and padding [S9].

Total inference memory also includes activations, padding, workspace, caches,
codec/vocoder state, allocator overhead and sometimes temporary dequantized
copies. Encoder compute may be dominated by matrix-matrix work, while
autoregressive decoding may benefit more from reduced weight bandwidth. A speedup
is a measurement question, not a consequence of dividing weight bytes by four.

## 5. Proposed bounded experiment

This is an **unselected experimental protocol**, not a new official scoring rule
or training recipe. Budget, quality margins and the deployment target require
owner review before execution. No dataset acquisition is needed for this review.

### Gate A — prove there is an executable product

For one selected checkpoint and the actual target device:

1. Identify the artifact currently served before calling it a floating-point
   baseline. The existing [Parakeet API config](../../configs/models/parakeet-tdt-0.6b-v3-crispasr.yaml)
   and [Qwen API config](../../configs/models/qwen3-asr-1.7b-audiocpp.yaml) name an
   endpoint/model ID, not the loaded weight precision. An upstream Q4_K default
   does not prove which artifact a particular server uses. Pin the source
   checkpoint, conversion code, runtime and dependencies.
2. Inventory parameters and operators by module. Record the exact weight,
   activation and cache precision, quantization grouping and excluded modules.
3. Export/load a floating-point version in the proposed target backend. Check
   output parity with the reference before introducing quantization.
4. Load/export the complete model using an existing supported path when one is
   available. Prototype a representative slice first only if custom integration
   is necessary. Inspect actual kernels, graph partitions, fallbacks and peak
   memory.
5. Stop before QAT if weights expand permanently to full precision at load, the
   relevant operations lack a usable backend, or export changes the decoding.
   Storage-only compression can be a separate goal, but must be named as such.

**Parakeet first hypothesis:** quantize supported encoder linear projections and
feed-forward matrices; leave convolution/subsampling, normalization, prediction
network, joiner and duration outputs at reference precision initially. This is a
conservative sensitivity experiment, not a measured optimal precision map.

**TTS first hypothesis:** select a concrete public model and permitted voice;
quantize its large backbone first, keeping codec/vocoder and output-facing
modules at reference precision. Do not transplant a convolutional BitTTS recipe
unchanged to a flow or autoregressive model.

### Gate B — establish whether training is necessary

Compare the same checkpoint and decoding settings in:

- original floating-point backend;
- target-backend floating point, to isolate conversion/runtime differences;
- target-backend INT8 PTQ, specifying weight-only or activation quantization;
- target-backend 4-bit weight-only PTQ, with the explicit mixed-precision map;
  use a supported calibration/reconstruction method rather than treating naive
  rounding as the strongest PTQ baseline.

Use representative **training/calibration** audio covering French, English,
code-switches, technical terms, short/long durations and relevant acoustic
conditions. Never calibrate on frozen evaluation or sealed panels. For each
candidate report stored bytes, resident weights, peak RAM/VRAM, warm median/p95
latency and RTF; measure on the same device, durations, batching and concurrency.
For TTS add time to first playable audio and streaming continuity.

If 4-bit PTQ already satisfies the approved quality and resource goals, **stop:
QAT/QAD is unnecessary cost**. If INT8 is good but INT4 regresses, perform a small
module-sensitivity comparison before considering a recovery run.

### Gate C — compare recovery methods, not unrelated models

Only after Gate A passes, compare selective QAT and, if affordable, same-model
teacher QAD against the selected PTQ baseline. Keep starting weights, data,
quantization grid and evaluation fixed. The teacher stays frozen. Include a
matched floating-point fine-tuning control if claiming a quality improvement.

For Parakeet, encoder feature matching at corresponding audio frames is a
possible first distillation signal. A text-LLM KL loop is not automatically
valid for TDT: blank/token probabilities, durations and transducer alignments
must be handled consistently. For autoregressive speech models, logit matching
needs identical conditioning/tokenization and teacher-forced alignment. For
flow-based TTS, an intermediate-feature or flow/velocity objective needs its own
validation; there is no universal next-token softmax to match. These are
implementation hypotheses, not validated losses for this project.

Export with exactly the quantizer used in recovery: format, group size, scaling,
clipping, rounding and module map. A second unrelated quantization in a GGUF or
other converter is a new experiment. Evaluate **the packed deployed artifact**,
not only its fake-quantized training model.

A future run must respect single-GPU operation (D-008), stop/resume (D-021) and
cache correctness (D-017) from the [directives](../owner/directives.md). Resume
state must include quantizer/observer state and teacher identity where needed;
using a teacher does not authorize sharding across multiple GPUs.

### Gate D — quality means non-inferiority, not an insignificant p-value

For STT reuse the [official API evaluation](../stt/evaluation.md), with the
[frozen campaign and sealed-set rule](../decisions/accepted/0002-frozen-evaluation-and-sealed-test-set.md).
Check French, English and code-switch panels separately; also inspect
hallucinations, empty outputs, truncation and failed requests. A single aggregate
WER can hide a regression on the intended use.

Before running, define an acceptable WER increase for each important panel.
For illustration only, a **0.3 percentage-point** margin at a 10.0% reference WER
would permit at most 10.3%; this is not an accepted project tolerance. Use the
upper bound of the paired difference interval against that margin. An interval
that crosses zero, or the evaluator's `not separable` verdict, is **not proof of
equivalence**. A panel too small or correlated to rule out the margin yields an
inconclusive result, not a pass. The evaluation document records these coverage
and bootstrap limitations; do not open the sealed set to resolve repeated
candidate selection.

Technical-term recall and real spontaneous FR/EN dictation remain coverage gaps.
Any added diagnostics must be labelled exploratory and must not silently change
existing panel contents or normalization.

For TTS, define a separate protocol before selection: fixed text/language and
voice prompts, pronunciation of English terms in French, numbers, long speech,
prosody, voice similarity, clipping/noise/instability, and **blinded paired human
listening**. Use repeated seeds when sampling matters. ASR-derived WER and
predicted MOS are useful diagnostics, not sufficient evidence of preserved
naturalness. Establish an owner-reviewed listening non-inferiority criterion;
there is no accepted TTS evaluation protocol in the repository yet.

Use distinct artifact identities/model IDs for each precision and runtime.
The current [transcript cache](../stt/evaluation.md#4-run-and-read-the-result)
cannot detect a changed remote engine behind an unchanged ID/endpoint. Never
reuse floating-point transcripts for quantized weights; follow the documented
fresh-run rules after a runtime replacement. Timing comparisons need actual
fresh requests, not cached historical latency.

### Decision after the pilot

- **Go:** the exported mostly-4-bit model meets pre-agreed quality margins on
  required panels/listening tests and a useful measured deployment saving.
- **Prefer INT8/mixed precision:** 4-bit gains are too small, too slow or too
  damaging; a fallback is a useful outcome, not a failed investigation.
- **No-go for the chosen stack:** no executable low-bit path, unacceptable
  recovery cost, or persistent quality regression. Preserve the negative result.
- **Inconclusive:** available evaluation cannot bound the allowed regression.
  Request an approved new campaign/protocol rather than claiming no loss.

## 6. Cost, risks and outstanding choices

| Work | Relative effort | Main uncertainty |
| --- | --- | --- |
| Existing runtime PTQ and quality comparison | Lowest if exact architecture already supported | Mixed-module coverage and target-device kernels |
| Custom selective QAT plus export | Moderate to high | Matching training numerics to the actual serving representation |
| QAD | Higher training memory/compute than comparable QAT | Speech-appropriate teacher objective and single-GPU fit |
| Whole-graph W4A4 or custom INT4 convolution/TDT kernels | Highest; potentially a runtime project | Benefit may not justify engineering and quality cost |

No credible GPU-hour or recovery-data budget follows from the reviewed papers
for this exact setup. Estimate those only after a small approved profiling run.

Before execution, settle:

1. Which checkpoint, deployment GPU and inference engine are the first target?
2. Is the goal disk size, idle residency, peak memory, latency, or coexistence
   with other models? What minimum measured gain makes the work worthwhile?
3. What per-panel quality margins and TTS listening criteria are acceptable?
4. Which data and voices are permitted for calibration, teacher targets and
   recovery, and what is the compute ceiling?
5. Is a custom exporter/kernel in scope if existing runtimes fail Gate A?

Data/model/voice rights remain distinct. STT training and teacher targets must
respect the [source register](../stt/sources.md), D-018/D-019 and the
[model-license decision](../decisions/accepted/0001-parakeet-fr-model-license.md).
Unlabelled audio or synthetic teacher targets are not a license exemption.
The register's TTS candidates are not an accepted TTS project or blanket
permission to use a voice. Personal audio/transcripts stay out of the repository
(D-013); datasets, weights and run outputs remain external. Publication still
requires the owner's request (D-006).

## 7. Sources and evidence limits

All sources accessed 2026-11-09. Paper revisions are fixed below; upstream
`main`/`latest` documentation is mutable and is not a tested environment lock.
Library documentation was also checked through Context7. Search snippets and
third-party summaries are not treated as experimental proof.

- **[S1]** Ding et al., Interspeech 2022, revised 2023.
  [4-bit Conformer with Native Quantization Aware Training, v4](https://arxiv.org/html/2203.15952v4).
  Full text, Tables 1–2 and quantization scope read. Conformer transducer,
  English; not Parakeet/TDT or French evidence.
- **[S2]** Rybakov et al., Interspeech 2023.
  [2-bit Conformer quantization for automatic speech recognition](https://www.isca-archive.org/interspeech_2023/rybakov23b_interspeech.pdf).
  Full text, Tables 1–2 read. Useful INT4 reference and limits of more aggressive
  compression; no direct project reproduction.
- **[S3]** Kawamura et al., 2025.
  [BitTTS, v1](https://arxiv.org/html/2506.03515v1).
  Full text, Table 1 and Sections 3.1–3.3 read. English convolutional TTS,
  INT8 storage of 4-bit levels, vocoder sensitivity; not packed INT4 GPU proof.
- **[S4]** NVIDIA, 2026.
  [NVFP4 QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf),
  [research overview](https://research.nvidia.com/labs/nemotron/nemotron-qad),
  [QAT/QAD guide](https://nvidia.github.io/Model-Optimizer/guides/quantization_aware_training_and_distillation.html)
  and [executable example/support scope](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat).
  Report passages and primary guides read. LLM/VLM accuracy recovery, not a
  speech benchmark or proof that audio uses the example unchanged.
- **[S5]** PyTorch.
  [TorchAO QAT workflow](https://github.com/pytorch/ao/blob/main/docs/source/workflows/qat.md).
  Prepare/convert and fake-quantization documentation read; Linear coverage is
  not whole-speech-model coverage.
- **[S6]** NVIDIA.
  [Model Optimizer feature support matrix](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html).
  Format/device/backend prerequisites read; not model-specific validation.
- **[S7]** NVIDIA.
  [TensorRT quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html).
  Distinguishes INT4 weight-only, INT8 and FP4. Kernel/model availability must
  still be checked for the selected version and graph.
- **[S8]** NVIDIA NeMo Speech.
  [ASR inference/export documentation](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/asr/inference.rst).
  Encoder export support checked through Context7; not a promise of quantized
  end-to-end TDT serving on the repository's pinned NeMo version.
- **[S9]** NVIDIA.
  [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
  Fine-grained scaling description; scale overhead is separate from raw FP4
  payload and does not specify total audio-model memory.

- **[S10]** Feng et al., October 2026 preprint.
  [TTS-Q v2 PDF](https://arxiv.org/pdf/2610.12474v2) and
  [HTML](https://arxiv.org/html/2610.12474v2).
  Method, Table 1, Section 4.5 and deployment limitations read and cross-checked
  against the PDF. The abstract page/metadata has a different title and method
  summary from the PDF/HTML body; the claims here refer specifically to the
  latter's speaker-aware experiment. Logical W4A4 with speaker distillation is
  not native packed-INT4 or NVFP4 execution; no French or human-listening
  equivalence result established.
- **[S11]** Google, June 2026 release.
  [Gemma 4 QAT announcement](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
  and [official 31B Q4_0 model card](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf).
  Announcement and card read; confirms QAT deployment artifacts, not a paired
  ASR/TTS quality benchmark or long-context equivalence claim.

- **[S12]** Shao et al., SLT 2024 revision.
  [DQ-Whisper, v2](https://arxiv.org/html/2305.10788v2).
  Joint-distillation setup and Common Voice Table 4 read. The joint method is
  8-bit; the naive 4-bit comparison is not evidence against all 4-bit PTQ.

- **[S13]** CrispASR maintainers.
  [GGUF quantization and backend support](https://github.com/CrispStrobe/CrispASR/blob/main/docs/quantize.md)
  and [performance/regression notes](https://github.com/CrispStrobe/CrispASR/blob/main/PERFORMANCE.md).
  Quantizer documentation and relevant FastConformer/long-audio regression
  passages read. Existing PTQ routes, not independent French quality validation
  or proof of matching the runtime currently installed on a target machine.
- **[S14]** April 2026 study.
  [Pushing the Limits of On-Device Streaming ASR, v2](https://arxiv.org/html/2604.14493v2).
  Methods and batch comparison read; a delegated review examined Appendix A.4's
  exact Parakeet quantization table. This review does not quote its quantization
  deltas as an isolated effect because the floating-point export baseline
  differs from the original PyTorch baseline. English PTQ, not French QAT.

- **[S15]** audio.cpp maintainers.
  [Qwen3 conversion documentation](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/qwen3.md)
  and [Qwen3-ASR runtime options](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/qwen3_asr.md).
  Both read. Mixed GGUF precision and language-detection warning; no French
  low-bit quality guarantee or inference-performance claim follows.

- **[S16]** Kyutai.
  [Pocket TTS upstream documentation](https://github.com/kyutai-labs/pocket-tts),
  [original English report](https://kyutai.org/blog/2026-01-13-pocket-tts)
  and [quantization discussion](https://github.com/kyutai-labs/pocket-tts/issues/7).
  Current README read; delegated research checked the report and CPU INT8 path.
  The original English report must not be used as measured quality for later
  French checkpoints. Community INT8 reports are not a published French
  quality-equivalence test.

- **[S17]** Qwen Team, January 2026.
  [Qwen3-TTS technical report](https://arxiv.org/abs/2601.15621).
  Architecture and quantization scope checked by delegated research against the
  full report: 12Hz autoregressive codec architecture, not a published low-bit
  quality study. Existing Q4 runtime support is documented separately in [S13].

- **[S18]** NVIDIA.
  [GeForce RTX 50 FP4 deployment example](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)
  and [TensorRT-LLM quantization/model/hardware matrices](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html).
  Both read, including the SM120 NVFP4 hardware row and separate supported-model
  table. FP4 silicon and a demonstrated FLUX path are not audio validation.
- **[S19]** Microsoft.
  [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
  Delegated primary-documentation and Context7 review: weight-only block
  quantization covers selected MatMul/Gather, not general convolution; exact
  provider/version and exported graph remain deployment gates.

[S1]: https://arxiv.org/html/2203.15952v4
[S2]: https://www.isca-archive.org/interspeech_2023/rybakov23b_interspeech.pdf
[S3]: https://arxiv.org/html/2506.03515v1
[S4]: https://nvidia.github.io/Model-Optimizer/guides/quantization_aware_training_and_distillation.html
[S5]: https://github.com/pytorch/ao/blob/main/docs/source/workflows/qat.md
[S6]: https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html
[S7]: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html
[S8]: https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/asr/inference.rst
[S9]: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
[S10]: https://arxiv.org/pdf/2610.12474v2
[S11]: https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf
[S12]: https://arxiv.org/html/2305.10788v2
[S13]: https://github.com/CrispStrobe/CrispASR/blob/main/docs/quantize.md
[S14]: https://arxiv.org/html/2604.14493v2
[S15]: https://github.com/0xShug0/audio.cpp/blob/main/docs/models/qwen3.md
[S16]: https://github.com/kyutai-labs/pocket-tts
[S17]: https://arxiv.org/abs/2601.15621
[S18]: https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
[S19]: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

### What remains unproven

No local low-bit artifact, WER/MOS measurement, memory measurement or speedup is
reported here. Exact Parakeet TDT v3 French recovery, current TTS candidate
quality, full audio-encoder/codec coverage, and a QAT-to-serving exporter that
preserves the trained grid are still experimental questions. A negative literature search is not proof
that no implementation exists; recheck exact versions at Gate A.
