# Low-bit training for audio models

- Status: idea; not investigated.
- Updated: 2026-11-09.
- Work item: [N-08 in the backlog](../backlog.md#not-started--deferred).

## Question

Could a low-bit or quantization-aware training approach for an audio model keep
quality comparable to a BF16 or FP16 reference? Parakeet is one possible first
case, not a selected implementation.

This records a research question, not a claim of feasibility, a training recipe
or permission to start a run. No literature review, code or measurements have
been produced for this topic.

## Open points before an experiment

- Define which method is being tested and what its precision settings mean;
  select the training and inference conditions rather than treating all
  low-bit approaches as interchangeable.
- Establish library and model support from primary documentation and research.
- Choose the reference, fixed data, budget and comparison protocol. Define what
  counts as a meaningful quality difference before running the comparison.
- Decide which resource measurements matter. Training memory and throughput,
  exported model size and inference cost are separate questions, not implied
  benefits of the hypothesis.
- For an STT case, reuse the [official evaluation](../stt/evaluation.md) and its
  frozen campaign rules. A diagnostic inside training is not an official score.

## Evidence and outcome

None yet. Add sources, a proposed protocol and, if useful, a linked standalone
experiment only when this work is taken up. There is no scheduled run or
conclusion to reproduce at this stage.
