# AGIR — Reproducibility Code

Code accompanying "AGIR: Deciding When Visual Grounding Should Intervene in
Vision-Language Models" (Qwen2.5-VL-7B-Instruct and
llava-hf/llava-v1.6-mistral-7b-hf on A-OKVQA, with RefCOCOg localization and
deletion-based faithfulness evaluation).

**This repository contains the final code and frozen artifacts behind the numbers
reported in the paper.** No result tables, figures or model weights are included;
the frozen per-sample artifacts the numbers derive from are in `frozen_artifacts/`.

## Contents

```
qwen/
  00_build_pool_caches_and_validation_runs.ipynb   (builds the 10,000-example pool caches + full-validation runs)
  01_pipeline_predictions_faithfulness_refcoco.ipynb
  01b_fit_corrected_gate.py                         (reconstruction, verified against frozen gate)
  02_gate_correction_threshold_efficiency.ipynb
  frozen_artifacts/
    pool_caches/            intervention_cache_deduped, extended_features_cache, full_validation_all_configs, qwen_corrected_final_results
    validation_predictions/ direct_predictions_seed42, dino_replay_seed42
    faithfulness_refcoco/   faithfulness_v3_n500, faithfulness_v3_summary, refcoco_v3_full_metrics
    final_gate_v9/          frozen gate pickle, split IDs, freeze/spec files, validation decisions, primary stats, efficiency run
    threshold_sweeps/       dev-sweep CSVs for the threshold figure
llava/
  01_full_pipeline_gate_faithfulness_refcoco_efficiency.ipynb
  frozen_artifacts/         final v5 gate + scaler pickles, split IDs, threshold sweep, validation decisions,
                            faithfulness / RefCOCOg / efficiency per-sample outputs, statistics
ivc_revision_corrections/
  agir_corrections_notebook.py
  frozen_outputs/           outputs of the three revision analyses (A audit, B matched controls, C random gate)
figures/
  make_figures.py
```

## Frozen artifacts and what "reproduce" means here

Model outputs from large VLMs will not be bit-identical on a rerun (hardware,
library versions, non-deterministic kernels). `frozen_artifacts/` therefore
holds the exact per-sample outputs and fitted gates the paper's numbers were
computed from, so every downstream step (gate fitting, policy evaluation,
statistics, figures, corrections) can be re-run and checked from them without
a GPU. Re-running the model-generation notebooks from scratch should give
close but not identical values. No paper tables, figures or PDFs are included.

### `qwen/00_build_pool_caches_and_validation_runs.ipynb`
Builds the 10,000-example intervention cache (three-class labels, native-box
fusion), the extended-feature cache (answer confidence etc.), the full
A-OKVQA-validation runs, RefCOCOg and fusion comparisons, from the models and
datasets directly. Its outputs are the files in `qwen/frozen_artifacts/pool_caches/`.

### `qwen/01b_fit_corrected_gate.py`
Fits the corrected confidence-only gate (`StandardScaler` + balanced
`LogisticRegression`, target beneficial=1 / harmful=0 / neutral=0, frozen
train/dev IDs from `CORRECTED_GATE_SPLIT.json`) and applies it at the frozen
$\tau^\star=0.35$ to the 1,145 validation rows.
**Provenance:** this script re-implements the gate-fitting step from the frozen
specification files (`CORRECTED_GATE_SPLIT.json`, `CORRECTED_GATE_TARGET_SPEC.md`)
and cached pool features. It is verified against the frozen artifacts:
coefficient, intercept and scaler statistics match `confidence_benefit_gate.pkl`
to numerical precision, and the 635/1,145 validation replay decisions are
identical to the frozen predictions file.
**Note:** $\tau^\star=0.35$ is read from `CORRECTED_GATE_FREEZE.json` as a frozen
constant. The script checks the gate against the frozen development sweep in
`frozen_artifacts/threshold_sweeps/`: the development replay counts reproduce exactly at
all 19 thresholds, but the sweep's outcome columns (policy accuracy, beneficial recall)
do not reproduce from the retained pool caches, so the threshold search itself is not
re-derived from them.
**Inputs:** the cache files in `qwen/frozen_artifacts/pool_caches/` (built by
`00_build_pool_caches_and_validation_runs.ipynb`); run with
`python qwen/01b_fit_corrected_gate.py`.

### `qwen/02_gate_correction_threshold_efficiency.ipynb`
The audited gate: the corrected beneficial-replay target, removal of the
leaked post-replay feature, the disjoint train/development split, the
pre-declared threshold grid and lexicographic tie-break, the frozen
confidence-only gate and scaler, the corrected validation predictions, and
the efficiency timing run with the gate strictly executed before any
evidence-phrase generation (the order-correct timing protocol). Depends on
the frozen predictions produced by `01_...ipynb`.

### `llava/01_full_pipeline_gate_faithfulness_refcoco_efficiency.ipynb`
The complete, self-contained final LLaVA-v1.6-Mistral-7B pipeline: base
answer/confidence/grounding/replay code, the corrected gate (disjoint
train/development split, same threshold protocol as Qwen), the frozen
zero-replay primary policy ($\tau^\star=0.95$) and the superseded $\tau=0.50$
historical ablation, the faithfulness run, the RefCOCOg direct-expression
localization run, and the efficiency timing run. Unlike Qwen, no earlier
part of this run is superseded by a later one — this single notebook is
the final version.

### `ivc_revision_corrections/agir_corrections_notebook.py`
A marimo notebook written for the journal's major-revision checklist.
Read-only with respect to every frozen artifact above (no model is
retrained, no gate is refit, no threshold is reselected). It:
- audits the faithfulness score definition against the implementation
  (`log_likelihood_score`'s loss reduction vs. the manuscript's Eq. 22);
- builds an upgraded, deterministic, five-placement, non-overlapping,
  area-/shape-matched faithfulness control for Qwen, with eligibility
  reported for every one of the 500 original sample IDs; the equivalent
  LLaVA run is intentionally left stubbed and documented as blocked in the
  notebook, because no archived LLaVA cache retains the per-sample evidence
  box or phrase needed to rerun it without regenerating an answer;
- runs a replay-budget-matched random-gate diagnostic for Qwen (10,000
  seeded draws at the frozen 635/1,145 budget).

Run in [molab](https://molab.marimo.io/) or any marimo runtime
(`pip install marimo`, then `marimo edit agir_corrections_notebook.py`).
Task B's Qwen cells additionally need a CUDA GPU and download
Qwen2.5-VL-7B-Instruct from the Hugging Face Hub; they are gated behind an
explicit `RUN_QWEN_TASK_B` flag so the rest of the notebook runs without a
GPU.

### `figures/make_figures.py`
Regenerates every Qwen results figure in the paper directly from the frozen
output files listed above (dev-sweep, accuracy outcomes, latency
decomposition, faithfulness distribution, RefCOCOg IoU distribution). Code
only — the rendered figures themselves are not included in this repository.

## Scope

This repository contains the final versions of the code and frozen artifacts behind the
reported results. Earlier exploratory iterations of the gate and threshold
protocol (documented in the paper's evaluation-history paragraph) are not
included. No result tables, figures or PDFs are included.

## Integrity

`MANIFEST.sha256` lists SHA-256 hashes of every frozen artifact. The hashes recorded in the freeze files for the Qwen
Direct/DINO predictions, the Qwen gate and the LLaVA gate/scaler match the released files.

## License

The code in this repository is released under the MIT License (see `LICENSE`).
Third-party models and datasets listed below remain under their own licenses.

## Models and data used (not redistributed here)

- Qwen2.5-VL-7B-Instruct, llava-hf/llava-v1.6-mistral-7b-hf — loaded from
  the Hugging Face Hub under their respective model licenses.
- Grounding DINO (base) for open-vocabulary detection.
- A-OKVQA and RefCOCOg — third-party benchmarks used under their respective
  licenses; no benchmark images are redistributed here.
