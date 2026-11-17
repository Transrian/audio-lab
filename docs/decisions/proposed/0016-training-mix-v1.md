# 0016 — Training mix v1

- Scope: stt
- Derived from: decision 0004 (the run the mix feeds), the per-source screening
  ([screening.md](../../stt/screening.md)) and the larger training preparations
  ([cards](../../stt/datasets.md), [register rows](../../stt/sources.md))
- Decided by: proposed by an agent, 2026-11-17; awaiting owner review

## Finding

The screening trained the base model once per source on 10 000 clips and found
that all four talk sources — the two French PeerTube listings and the two
English conference sources, every one of them Qwen/Whisper-agreement labels —
gain separably on the code-switched panels, the panels that measure
[D-009's](../../owner/directives.md) target use, with no point estimate worse
than the base, and that the Granary anchor harms nothing anywhere. The nine
larger preparations made the material for a mix
exist, and the mix is now buildable: a recipe may list several datasets with
`exclusive: true`, so a row's pool drops what the whole datasets of the earlier
rows already hold (the duplicate class of its talk for a talk source, its audio
`sha256` otherwise), and the run's [leakage
guard](../../stt/training.md#the-leakage-guard) rechecks the drawn clips against
the evaluation sets before the model is loaded
([training interface](../../stt/training.md#mixing-datasets)).

The mix's rows and shares, measured read-only on 2026-11-17 over the current
prepared data with the code path a run uses — the recipe read by the config
reader, the draw by `select_mix` with `exclusive: true`, and the guard's
`check_leakage` on the drawn clips ([config](../../../configs/train/proposed-mix-v1.yaml),
which is a proposal and must not be started before this decision). Rows are in
the proposal's order, the tech talks before the two PeerTube listings they are a
filtered union of, so the subtraction takes them out of those rows:

| mix row | available (after subtraction) | hours | share | drawn at 151 882 |
|---|---|---|---|---|
| `granary-fr-train-40k` | 37 971 — the whole set | 26.3 | 25 % | 37 971 |
| `peertube-tech-fr-train-40k` | 40 000 | 110.9 | 15 % | 22 783 |
| `peertube-fr-train-40k` | 27 697 (12 303 lost to the tech row) | 73.6 | 10 % | 15 188 |
| `peertube-edu-fr-train-40k` | 36 644 (3 356 lost to the tech row) | 97.4 | 10 % | 15 188 |
| `voxpopuli-fr-train-40k` | 24 259 | 59.4 | 5 % | 7 594 |
| `african-accented-fr-train` | 11 424 | 13.5 | 5 % | 7 594 |
| `librispeech-pc-en-train-40k` | 25 449 | 89.5 | 10 % | 15 188 |
| `debconf-en-train-30k` | 30 000 | 84.9 | 7½ % | 11 391 |
| `fosdem-en-train-30k` | 30 000 | 89.4 | 7½ % | 11 391 |
| `nasa-en-train-30k` | 30 000 | 86.8 | 5 % | 7 594 |
| **total** | **293 444** | **731.7** | 100 % | **151 882** (339.3 h, mean clip 8.0 s) |

"Available" is the usable pool a row draws from (text, 0.3–20 s) after the
subtraction; the hours are that pool's hours. The French rows hold 70 % of the
clips and the English rows 30 %: the proposal's reading of D-009's "French is
the priority, English is kept". At the [decision 0004](../../decisions/accepted/0004-parakeet-fr-baseline-recipe.md)
run — its about 30 000 updates and constant rates, at the screening configs'
16 clips per update, weight decay, 20 s window and seed — about 480 000
presentations, so each drawn clip is seen about 3 times.

Facts and flags for the decision:

1. **Granary caps the scale.** At 25 %, the mix reaches 151 884 clips exactly
   when all 37 971 granary clips are in — every other row has material left
   over. A larger mix needs the shares rebalanced and granary below 25 %: the
   pools hold 293 444 clips (731.7 h) after the subtraction, so a mix of that
   size is buildable (about 1.6 presentations per clip at the decision-0004
   run).
2. **`peertube-fr-train-40k` rejected 34.1 % of its labelled clips** (20 730 of
   60 855; the 2026-11-16 re-preparation the row uses measures 34.2 %, 20 838 of
   60 966) — above the other talk rows of the mix (tech 29.0 %, 16 340 of
   56 404 in its re-preparation; edu 29.0 %, debconf 31.8 %, fosdem 21.7 %).
   That is a label-quality warning on this one row; the mix keeps it at the
   proposed 10 %.
3. **VoxPopuli, after [decision 0014](../../decisions/accepted/0014-voxpopuli-shared-sessions.md):**
   the 40k preparation holds 24 259 clips (59.4 h) — the 4 815 clips (11.6 h)
   of the 219 plenary sessions shared with `voxpopuli-en-nonnative` are out of
   every preparation. The row's pool still holds 3.2× its 7 594 draw.
4. **The babble pools cannot be mix rows.** `mls-fr-train` and
   `librispeech-en-train` hold the babble utterances the noisy evaluation sets
   mix under their test clips (524 and 488 clips of the committed
   `babble_utterances.json`), and a new training run that holds one is refused
   by the guard. The mix's `librispeech-pc-en-train-40k` row is a different set:
   the 447 LibriSpeech utterances the noisy sets mix as babble left it in its
   2026-11-16 re-preparation (25 896 → 25 449 clips).
5. **The training↔training re-uploads, and how `exclusive` handles them.** The
   [content audit](../../../experiments/leakage-audit/README.md#content-level-leakage-6-word-text-shingles)
   re-run on 2026-11-17 over the rebuilt data reports **no same-recording pair
   between any evaluation group and any training set** (the eight leak pairs of
   2026-11-15 left the sets with the re-preparations), and no evaluation-vs-
   training pair shares an audio `sha256`. Its train-vs-train re-upload list
   (ratio ≥ 75 % of a talk's 6-word shingles, different group ids) holds:
   - the cross-row pairs, which the duplicate-class rule unifies and the
     subtraction removes: a ministry talk re-uploaded under a second id, in the
     edu row and in the tech row (`b01320a7…` and `c18d134b…`, 200 clips each) —
     one duplicate class, so the tech row, listed first, absorbs the class and
     the edu row loses its 200 clips as part of its 3 356; and the Capitole
     re-upload pair (`44325ad5…` and `66095949…`, one class) — the 40k
     re-preparation left the re-upload out of the fr row (its 154 clips sit only
     in the frozen 12 000-clip `peertube-fr-train` set, which is not in the
     mix) and the subtraction removes the fr row's 162 clips of the kept id —
     the tech row holds the same talk with 160 clips — part of its 12 303;
   - the in-set repetitions, which a row's subtraction does not touch (it only
     drops what other rows hold) and the mix's cross-dataset refusal tolerates
     by design (a repetition inside one dataset is that dataset's business):
     16 granary id-pairs — the same videos under a second id, 182 of the 37 971
     clips of `granary-fr-train-40k` — and, below the 75 % shortlist floor but
     in the audit's ≥ 30 % same-recording band, two pairs of the edu row (418
     clips) and one pair of the nasa row (71 clips).
6. **Known costs of the proposed rows, from the per-source screening**
   ([what each source does alone](../../stt/screening.md#what-each-source-does-alone)):
   `african-accented-fr-train` still harms the code-switched panels (servicenow
   en-fr +10.51, en-fr-ca +5.33) and non-native English (+5.14);
   `nasa-en-train-30k`'s source is mixed in the screening (on the 12 000-clip
   `nasa-en-train`): its first screen harmed librispeech-en
   (+1.16) and fleurs-en (+0.61), a text defect its re-preparation fixes (its
   re-screen is not separable on librispeech-en and separably better on
   fleurs-en), but it harms notsofar-en-eval
   (+3.69 [1.97, 5.65]) and keeps the non-speech hallucination rate above the
   base; `voxpopuli-fr-train-40k`'s source, screened on the 12 000-clip set,
   costs 4.7 points on non-native English, the panel closest to it. The four
   screened talk sources (the tech row was not screened on its own) show no
   point estimate worse than the base on any panel, and Granary shows no
   separable harm on any panel.

   The guard's check on this mix passes: `check_leakage` on the 151 882 drawn
   clips checked them against 42 489 evaluation clips of all 18 prepared
   evaluation datasets (none missing) and refused nothing.

## Question

Which mix does the decision-0004 run train on?

## Options

1. **Mix v1 as measured above**: 151 882 clips (339.3 h) at the table's shares,
   about 3 presentations per clip at the decision-0004 run. It is buildable
   today: [configs/train/proposed-mix-v1.yaml](../../../configs/train/proposed-mix-v1.yaml)
   implements it with `exclusive: true` and its drawn clips pass the guard.
2. **A larger mix, with granary below 25 %**: the pools hold 293 444 clips
   (731.7 h) after the subtraction, so any size up to that is buildable with
   the shares rebalanced (about 1.6 presentations per clip at the full pool);
   above 151 884 clips, granary cannot hold 25 %. The owner would state the
   target size and how the shares may move.
3. **Reduce or re-screen the 34.1 % row**: lower
   `peertube-fr-train-40k`'s share in option 1 (the rejection rate is a
   label-quality warning on that row alone), or re-screen the source before
   training, with the mix at whatever the re-screening leaves.

## Label quality (O-11, decided with the mix)

75 % of the mix's clips are model labels: the granary row's pseudo-labels and
the five Qwen/Whisper-agreement rows (tech, fr, edu, debconf, fosdem). The
human-labelled rows —
`voxpopuli-fr-train-40k`, `african-accented-fr-train`,
`librispeech-pc-en-train-40k` and `nasa-en-train-30k` — hold the other 25 %.
The [label-style test](../../stt/screening.md#label-style-same-audio) trained
on the same 10 000 NASA clips with human labels and with Qwen's labels alone
and compared: model labels cost **+1.46 points [1.07, 1.88] on
`nasa-en-test`** and **+2.17 points [1.08, 3.44] on
`peertube-edu-captions-fr`** against the human labels, while none of the
four code-switched panels is separable — where the talk gain is real, it comes
from the audio, not the label style. The other side of the ledger, outside the
pre-registered reading: the model labels are **better on `notsofar-en-eval`
(−3.30 points [−5.10, −1.68]) and on `voxpopuli-en-nonnative` (−2.87 points
[−3.97, −1.98])**. The test's pre-registered option to
tighten the agreement threshold names no panel in its registered text — only
reading (a) names the code-switched ones; read on that scope it does not fire,
while on the human-reference panels the human labels are better (+1.46, +2.17),
which its text matches. Its scope was not fixed at registration, so both sides
of the ledger are reported to the owner, to be decided with the mix
([O-11](../../backlog.md)). The options it opens:

- **Tighten the agreement threshold** `MAX_DISAGREEMENT` (0.15,
  `src/audiolab/data/talks.py`) and re-label the model-labelled talk rows
  before training;
- **Raise the share of the human-labelled rows** in the mix (25 % as measured
  above);
- **Accept**: train on the mix as measured, on the reading that the code-
  switched panels — the target use — do not separate.

Whichever way this is decided, the data stays as measured: no set is touched
by the decision, and the table above is the state of the current prepared
data, recomputed whenever it changes.
