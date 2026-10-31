# Pre-submission audit notes (checklist items 1-3)

## Item 1 - replay-budget random gate (from the saved 10,000-draw file)
Source: `random_budget_gate_qwen_draws.csv` (10,000 unique draws; every draw has exactly 635 replayed samples out of 1,145; seed 42)
and `random_budget_gate_qwen_summary.json`, both written by `agir_corrections_notebook.py` (Task C).
Percentiles use NumPy's default linear interpolation.

| statistic | value |
|---|---|
| mean | 64.14% |
| 5th percentile | 63.32% |
| median | 64.10% |
| 95th percentile | **64.89%** (the manuscript previously printed 64.49%, a transcription error) |
| frozen AGIR | 64.63% |
| draws strictly below AGIR | 82.21% |
| draws tied with AGIR | 4.07% |
| draws strictly above AGIR | 13.72% |

The saved summary JSON already contained p95 = 0.64891; the manuscript number was mistyped. No re-simulation was needed.

## Item 2 - the 45 out-of-bounds evidence boxes
Per-ID inventory of all 174 exclusions: `area_matched_faithfulness_eligibility.csv` (106 fewer than five controls, 45 box out of bounds, 23 no cached box).
Cached boxes for the 45: `oob45_cached_boxes.csv` (first Grounding DINO box, from `dino_replay_seed42.jsonl`).
Geometry of those 45 boxes:
- 28 have a negative x1 or y1, the most negative being -5.9 px; 12 have x2 or y2 above 640 px, the maximum being 642.6 px.
- No coordinate is far outside any plausible image (no value above 643 px), which is what small detector overshoot at the image edge looks like. A coordinate-frame or resize bug would give large or scaled-wrong values.
- Median box area is about 100,000 px for the 45 versus about 18,300 px for the eligible 326: exclusion removes large boxes that touch the image edge, so the matched-control subset favours smaller boxes.
- Not verified here: the true width/height of each of the 45 images, EXIF orientation, and the exact coordinate convention used by the original masking run. This needs access to the A-OKVQA validation images, which were not reachable from the analysis environment. Boxes were not clamped.
- PIL's `ImageDraw.rectangle` clips out-of-range coordinates, so a few pixels of overshoot do not change which pixels the original N=500 deletion masks cover; this is an inference, not a re-run.

## Item 3 - precision
The main pipelines (`qwen/00`, `qwen/01`, `qwen/02`, `llava/01`) load models in `torch.bfloat16`.
The upgraded fixed-answer scoring in `agir_corrections_notebook.py` (Task B) loads Qwen in `torch.float16`.
The manuscript's precision table said bfloat16 for all analyses; it now states the exception. A BF16 sensitivity run
can be done by setting `AGIR_SCORING_DTYPE=bfloat16` before running Task B (see notebook); it has not been run.
