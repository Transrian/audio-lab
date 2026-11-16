# 0014 — VoxPopuli: train clips from the evaluation set's plenary sessions

- Scope: stt
- Derived from: the leakage table of [datasets.md](../../stt/datasets.md#training-sources)
- Decided by: owner, 2026-11-15 (option 2)

## Finding

`voxpopuli-en-nonnative` (737 clips, 2.36 h, 27 speakers, 307 plenary
sessions) and `voxpopuli-fr-train` (12 000 clips, 29.0 h, 229 speakers, 2 265
plenary sessions) are built from the same European Parliament corpus with
different language filters. An audit on 2026-11-13 of the prepared manifests
found: a *session* is the clip id's prefix before the language tag (for
example `20090310-0900-PLENARY-7`), and the population is the prepared data —
the two manifests, not the corpora — reproducible with
[`experiments/leakage-audit/sessions.py`](../../experiments/leakage-audit/README.md):

- **no shared speaker**: none of the 27 evaluation speakers is among the 229
  prepared train speakers (the train fetcher already excludes the 896 clips of
  the evaluation set's speakers);
- **206 of the evaluation set's 307 sessions (67%) are also in the train set**:
  the same plenary sittings, recorded at the same time in the same rooms and
  with the same microphones. 480 of the evaluation set's 737 clips (1.55 h of
  its 2.36) and 1 956 of the train set's 12 000 clips (4.67 h of its 29.0) come
  from those 206 sessions.
- in the larger preparation `voxpopuli-fr-train-40k` (26 973 clips, 2 462
  sessions), 219 of the 307 sessions are shared: 508 of the evaluation set's
  737 clips (1.63 h of its 2.36) and 4 815 of the preparation's 26 973 clips
  (11.61 h) come from those 219 sessions.

A clip and a train clip from the same sitting are different languages and
different utterances, so this is not a text leak: the train text never contains
the evaluation sentences. What is shared is the acoustics — room, microphone
and background of the sitting — and, for the evaluation set's 27 speakers,
nothing at all.

## Question

Does the shared acoustics disqualify the 1 956 train clips, or the evaluation
set's 480 clips, or neither? The leakage rules say a clip trained on cannot be
scored; these are not that case. A model trained on the rooms in which the
evaluation was recorded can still be testing the speakers, the accents and the
sentences it has never heard.

## Options

1. **Keep both sets as they are.** The overlap is acoustic only, the evaluation
   set measures accented speech in plenary rooms, and a model that has heard
   other languages in the same rooms is arguably a more realistic opponent.
   No re-preparation; the leakage table states the numbers.
2. **Drop the train clips of the shared sessions** and re-prepare
   `voxpopuli-fr-train`: the 12 000-clip set loses its 1 956 shared-session
   clips and the rebuild refills it to 12 000 from the sessions it does not
   share; the `-40k` preparation drops its 4 815 shared-session clips, and the
   speakers it holds at the 200-clip share refill part of them from their
   other sessions, so its size is known after the rebuild. The train set then
   shares neither speaker nor room with the evaluation set.
3. **Rebuild the evaluation set from the 101 sessions it does not share.** The
   257 remaining clips are about 0.8 h — below the set's purpose, and the
   panel is frozen, so this option is only listed for completeness.

Option 2 is the conservative reading of the leakage rules; option 1 is the
narrow one. The screening run `screen-voxpopuli-fr` has trained on 10 000
clips of the set — its committed [config](../../configs/train/screen-voxpopuli-fr.yaml)
names `voxpopuli-fr-train` with `clips: 10000` — so the set's figures describe
data a run has already seen; this decision changes the set going forward, not
any number that has been reported.

## Consequence

The `voxpopuli-fr-train` preparations (12 000 and `-Nk`) drop those sessions:
every preparation leaves out each clip of the evaluation set's 307 sessions,
read from the committed pin
[`src/audiolab/data/voxpopuli_eval_sessions.json`](../../src/audiolab/data/voxpopuli_eval_sessions.json)
by `src/audiolab/data/voxpopuli_train.py`.
