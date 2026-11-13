# 0014 — VoxPopuli: train clips from the evaluation set's plenary sessions

- Scope: stt
- Derived from: plan 01 (T8), the leakage rules of [datasets.md](../../stt/datasets.md)
- Decided by: proposed by an agent, 2026-11-13; awaiting owner review

## Finding

`voxpopuli-en-nonnative` (737 clips, 2.36 h, 27 speakers, 307 plenary
sessions) and `voxpopuli-fr-train` (12 000 clips, 29.0 h, 229 speakers, 2 263
plenary sessions) are built from the same European Parliament corpus with
different language filters. An audit on 2026-11-13 of the prepared manifests
found:

- **no shared speaker**: none of the 27 evaluation speakers is among the 229
  prepared train speakers (the train fetcher already excludes the 896 clips of
  the evaluation set's speakers);
- **206 of the evaluation set's 307 sessions (67%) are also in the train set**:
  the same plenary sittings, recorded at the same time in the same rooms and
  with the same microphones. 480 of the evaluation set's 737 clips (1.55 h of
  its 2.36) and 1 967 of the train set's 12 000 clips (4.70 h of its 29.0) come
  from those 206 sessions.

A clip and a train clip from the same sitting are different languages and
different utterances, so this is not a text leak: the train text never contains
the evaluation sentences. What is shared is the acoustics — room, microphone
and background of the sitting — and, for the evaluation set's 27 speakers,
nothing at all.

## Question

Does the shared acoustics disqualify the 1 967 train clips, or the evaluation
set's 480 clips, or neither? The leakage rules say a clip trained on cannot be
scored; these are not that case. A model trained on the rooms in which the
evaluation was recorded can still be testing the speakers, the accents and the
sentences it has never heard.

## Options

1. **Keep both sets as they are.** The overlap is acoustic only, the evaluation
   set measures accented speech in plenary rooms, and a model that has heard
   other languages in the same rooms is arguably a more realistic opponent.
   No re-preparation; the leakage table states the numbers.
2. **Drop the 1 967 train clips of the 206 shared sessions** and re-prepare
   `voxpopuli-fr-train` (about 24 h remain). The train set then shares neither
   speaker nor room with the evaluation set.
3. **Rebuild the evaluation set from the 101 sessions it does not share.** The
   257 remaining clips are about 0.8 h — below the set's purpose, and the
   panel is frozen, so this option is only listed for completeness.

Option 2 is the conservative reading of the leakage rules; option 1 is the
narrow one. This decision does not change any number that has been reported:
no campaign has trained on `voxpopuli-fr-train` yet.
