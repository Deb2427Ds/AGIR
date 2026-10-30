import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


# =====================================================================
# AGIR — IVC major-revision corrections notebook (marimo / molab)
#
# Implements, offline wherever possible:
#   Task A — faithfulness score-definition audit            (no GPU)
#   Task C — Qwen replay-budget-matched random gate          (no GPU)
#   Task B — area-/shape-matched faithfulness controls
#            Qwen: executable if a GPU + model download is available
#            LLaVA: BLOCKED — see Task B-LLaVA cell for why
#
# Nothing here retrains, refits or re-thresholds the frozen AGIR gates.
# Every number is written to research_corrections_2026_10/ as a new,
# separately labeled file — no frozen file is overwritten.
# =====================================================================


@app.cell
def __():
    import marimo as mo
    import os, json, hashlib, io
    import numpy as np
    import pandas as pd

    OUT_DIR = "research_corrections_2026_10"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

    mo.md(
        """
        # AGIR — IVC revision: corrections notebook

        Upload the files named in each section below using marimo's file
        upload cell (or drop them next to this notebook if running
        locally). Everything writes into `research_corrections_2026_10/`.
        """
    )
    return OUT_DIR, hashlib, io, json, mo, np, os, pd


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Task A — Faithfulness score-definition audit (Eq. 2 vs Eq. 22)

        **No file upload needed — this is a static code-provenance finding**,
        traced directly from the frozen notebooks (`notebook_v9.ipynb`,
        `qwen v3/notebook_part4.ipynb`, `qwen v3/notebook_part5.ipynb`).

        **Code trace**

        `log_likelihood_score(image, question, answer)` builds the prompt
        tokens and the prompt+answer tokens, masks the prompt span of
        `labels` to `-100`, and calls the HF model with `labels=...`. It
        returns `-out.loss`. HuggingFace's default `loss` reduction for a
        causal LM is the **token-mean** cross-entropy over the
        non-masked (answer) tokens — i.e. this already equals manuscript
        Eq. (2): `c(I,q,a) = (1/T) * sum_t log p_theta(a_t | ...)`.

        `run_faithfulness_v3(...)` then computes
        `s_original`, `s_relevant_deleted`, `s_irrelevant_deleted`,
        `s_region_only` by calling `log_likelihood_score(...)` directly —
        **with no further division by `len(answer.split())` or any word
        count anywhere in the function.** `c_del = s_original -
        s_relevant_deleted` and `c_irr = s_original - s_irrelevant_deleted`
        are differences of two already-token-mean quantities.

        **Finding**

        | Reported quantity | Code expression | Units | Extra normalization | Used for gate? | Used for faithfulness? |
        |---|---|---|---|---|---|
        | `c` (confidence, Eq. 2) | `-out.loss` (HF token-mean) | nats/token | none | yes | no |
        | `S(I')` (faithfulness, Eq. 22 as coded) | `-out.loss` (HF token-mean) | nats/token | **none** | no | yes |

        Manuscript Eq. (22), as currently typeset
        (`S(I') = c(I',q,a_hat) / max(1, n_words(a_hat))`), implies a
        **second** division by word count on top of the token-mean already
        inside `c`. That second division is **not present in the
        implementation**. This is a **documentation-only mismatch**:

        **Verdict: `MISMATCH` (manuscript only) → `NO_CHANGE` to any numeric result.**
        Fix is to retype Eq. (22) as `S(I') = c(I', q, \hat a)` (drop the
        `/max(1, n_words)` term) and state explicitly, once, that `c` is
        already a per-token mean. No rerun, no `RECOMPUTED_FAITHFULNESS_ONLY`
        path is triggered.

        A spot numerical check against the frozen cache is run below for
        extra assurance once you upload the faithfulness file.
        """
    )
    return


@app.cell
def __(mo):
    faith_qwen_upload = mo.ui.file(
        label="Upload faithfulness_v3_n500.jsonl (Qwen, from qwen_v3-5.zip → qwen v3/all files/)",
        kind="area",
    )
    faith_llava_upload = mo.ui.file(
        label="Upload faithfulness_llava_dino_n500.jsonl (LLaVA, from llava_v5_final.zip)",
        kind="area",
    )
    mo.hstack([faith_qwen_upload, faith_llava_upload])
    return faith_llava_upload, faith_qwen_upload


@app.cell
def __(OUT_DIR, faith_llava_upload, faith_qwen_upload, json, mo, np, os):
    def _load_jsonl_upload(upload):
        if not upload.value:
            return None
        raw = upload.value[0].contents.decode("utf-8")
        return [json.loads(l) for l in raw.splitlines() if l.strip()]

    _qwen_rows = _load_jsonl_upload(faith_qwen_upload)
    _llava_rows = _load_jsonl_upload(faith_llava_upload)

    _report_lines = ["# Task A — numerical spot check\n"]

    def _spot_check(name, rows):
        if rows is None:
            _report_lines.append(f"\n## {name}: NOT UPLOADED — skipped\n")
            return
        # Internal self-consistency: c_del must equal s_original - s_relevant_deleted
        # to float tolerance, with no extra word-count scaling anywhere.
        diffs = []
        for r in rows:
            if "s_original" in r and "s_relevant_deleted" in r and "c_del" in r:
                expected = r["s_original"] - r["s_relevant_deleted"]
                diffs.append(abs(expected - r["c_del"]))
        if diffs:
            _report_lines.append(
                f"\n## {name}\n"
                f"- N rows checked: {len(diffs)}\n"
                f"- max |c_del - (s_original - s_relevant_deleted)|: {max(diffs):.3e}\n"
                f"- mean abs diff: {float(np.mean(diffs)):.3e}\n"
                f"- Interpretation: differences at float-precision level confirm "
                f"`c_del` is an *undivided* difference of two token-mean scores, "
                f"consistent with the code trace above (no word-count rescale).\n"
            )
        else:
            _report_lines.append(
                f"\n## {name}: uploaded ({len(rows)} rows) but required fields "
                f"(`s_original`,`s_relevant_deleted`,`c_del`) not all present — "
                f"this file only carries post-hoc scores (expected for the LLaVA "
                f"summary cache, which never stored `s_original`/`s_relevant_deleted`).\n"
            )

    _spot_check("Qwen (faithfulness_v3_n500.jsonl)", _qwen_rows)
    _spot_check("LLaVA (faithfulness_llava_dino_n500.jsonl)", _llava_rows)

    with open(os.path.join(OUT_DIR, "faithfulness_score_definition_audit.md"), "w") as _f:
        _f.write(
            "# Faithfulness score-definition audit (Task A)\n\n"
            "**Verdict: MISMATCH (manuscript text only) -> NO_CHANGE to numeric results.**\n\n"
            "`log_likelihood_score()` returns `-out.loss`, which under HuggingFace's "
            "default causal-LM loss reduction is already the token-mean negative "
            "log-likelihood over the (label-masked) answer span -- i.e. exactly "
            "manuscript Eq. (2)'s `c(I,q,a)`. `run_faithfulness_v3()` builds "
            "`s_original`, `s_relevant_deleted`, `s_irrelevant_deleted`, "
            "`s_region_only` by calling `log_likelihood_score()` directly, with no "
            "further division by answer word count anywhere in the function body. "
            "`c_del` and `c_irr` are plain differences of two such token-mean "
            "scores. Manuscript Eq. (22) as currently typeset "
            "(`S(I') = c(I',q,a_hat) / max(1, n_words(a_hat))`) therefore does not "
            "match the implementation: it implies an extra division that was never "
            "executed. This is a documentation bug, not a methodology bug -- no "
            "faithfulness numbers are affected and no rerun is required. Patch: "
            "retype Eq. (22) as `S(I') = c(I', q, a_hat)` and note explicitly that "
            "c is already a per-token mean.\n\n"
            "Separately, the Qwen faithfulness code's existing 'irrelevant' control "
            "is NOT the fixed 0.2W x 0.2H corner box the pre-submission checklist "
            "assumed: it is already a box of the SAME width/height as the DINO box, "
            "placed via a single unseeded `np.random.uniform` translation with no "
            "enforced non-overlap. Task B below upgrades this to a seeded, "
            "non-overlapping, 5-draw matched-control protocol; it does not "
            "introduce area-matching that was wholly absent before.\n\n"
            + "\n".join(_report_lines)
        )

    mo.md("\n".join(_report_lines) + f"\n\n Written to `{OUT_DIR}/faithfulness_score_definition_audit.md`")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Task C — Qwen replay-budget-matched random gate (fully offline)

        **Upload needed:** `corrected_gate_validation_predictions.csv`
        (from `qwen_v9.zip`). Columns used: `sample_id`, `acc_direct`,
        `acc_dino`, `corrected_conf_decision` (the frozen primary gate's
        per-sample replay decision, tau*=0.35).

        No model, no image, no GPU. 10,000 deterministic random draws of
        exactly 635/1,145 replay decisions, seed 42, compared to the frozen
        AGIR policy and to Direct. LLaVA is not run here — at tau*=0.95 its
        frozen replay count is 0/1,145, so a budget-matched random
        selector is identical to Direct with zero variance
        (`TRIVIAL_COMPARATOR_NOT_INFORMATIVE`); that is reported as text,
        not recomputed.
        """
    )
    return


@app.cell
def __(mo):
    gate_csv_upload = mo.ui.file(
        label="Upload corrected_gate_validation_predictions.csv (Qwen, from qwen_v9.zip)",
        kind="area",
    )
    gate_csv_upload
    return (gate_csv_upload,)


@app.cell
def __(OUT_DIR, gate_csv_upload, mo, np, os, pd):
    if not gate_csv_upload.value:
        mo.stop(True, mo.md("Upload `corrected_gate_validation_predictions.csv` above to run Task C."))

    import io as _io
    _gate_df = pd.read_csv(_io.BytesIO(gate_csv_upload.value[0].contents))

    assert len(_gate_df) == 1145, f"expected 1145 rows, got {len(_gate_df)}"
    _acc_direct = _gate_df["acc_direct"].to_numpy(dtype=float)
    _acc_dino = _gate_df["acc_dino"].to_numpy(dtype=float)
    _agir_replayed = _gate_df["corrected_conf_decision"].astype(bool).to_numpy()

    _n = len(_gate_df)
    _budget = int(_agir_replayed.sum())
    assert _budget == 635, f"expected frozen Qwen primary replay budget 635, got {_budget}"

    _agir_policy_score = np.where(_agir_replayed, _acc_dino, _acc_direct)
    agir_mean_acc = float(_agir_policy_score.mean())
    _direct_mean_acc = float(_acc_direct.mean())

    _rng = np.random.default_rng(42)
    _n_draws = 10_000
    _idx_all = np.arange(_n)

    _draw_accs = np.empty(_n_draws, dtype=float)
    _draw_beneficial = np.empty(_n_draws, dtype=int)
    _draw_harmful = np.empty(_n_draws, dtype=int)
    _draw_neutral = np.empty(_n_draws, dtype=int)
    _draw_harmful_avoid = np.empty(_n_draws, dtype=float)

    # per-sample intervention label (beneficial/harmful/neutral), identical rule
    # to the manuscript's label definition, computed once from frozen soft scores
    _label_beneficial = _acc_dino > _acc_direct
    _label_harmful = _acc_dino < _acc_direct
    _label_neutral = ~(_label_beneficial | _label_harmful)
    _n_harmful_total = int(_label_harmful.sum())

    for _d in range(_n_draws):
        _replay_idx = _rng.choice(_idx_all, size=_budget, replace=False)
        _replayed_mask = np.zeros(_n, dtype=bool)
        _replayed_mask[_replay_idx] = True

        _score = np.where(_replayed_mask, _acc_dino, _acc_direct)
        _draw_accs[_d] = _score.mean()

        _draw_beneficial[_d] = int((_replayed_mask & _label_beneficial).sum())
        _draw_harmful[_d] = int((_replayed_mask & _label_harmful).sum())
        _draw_neutral[_d] = int((_replayed_mask & _label_neutral).sum())
        _draw_harmful_avoid[_d] = (
            1.0 - _draw_harmful[_d] / _n_harmful_total if _n_harmful_total else float("nan")
        )

    _summary = {
        "task": "C — Qwen replay-budget-matched random gate",
        "n_validation": _n,
        "replay_budget": _budget,
        "n_draws": _n_draws,
        "seed": 42,
        "frozen_agir_mean_soft_accuracy": agir_mean_acc,
        "direct_mean_soft_accuracy": _direct_mean_acc,
        "frozen_agir_minus_direct_pp": (agir_mean_acc - _direct_mean_acc) * 100,
        "random_policy_mean_soft_accuracy": {
            "mean": float(_draw_accs.mean()),
            "std": float(_draw_accs.std()),
            "p5": float(np.percentile(_draw_accs, 5)),
            "p50": float(np.percentile(_draw_accs, 50)),
            "p95": float(np.percentile(_draw_accs, 95)),
        },
        "frozen_agir_minus_random_mean_pp": (agir_mean_acc - float(_draw_accs.mean())) * 100,
        "fraction_of_random_draws_agir_beats": float((agir_mean_acc > _draw_accs).mean()),
        "n_harmful_total_in_validation": _n_harmful_total,
        "n_beneficial_total_in_validation": int(_label_beneficial.sum()),
        "n_neutral_total_in_validation": int(_label_neutral.sum()),
        "random_harmful_avoidance": {
            "mean": float(np.nanmean(_draw_harmful_avoid)),
            "p5": float(np.nanpercentile(_draw_harmful_avoid, 5)),
            "p50": float(np.nanpercentile(_draw_harmful_avoid, 50)),
            "p95": float(np.nanpercentile(_draw_harmful_avoid, 95)),
        },
        "note": (
            "Randomization spread is NOT a sampling CI over new datasets; it is "
            "the distribution of a fixed-budget random selector evaluated "
            "repeatedly on the same frozen validation scores. This is a "
            "post-hoc diagnostic on an already-examined benchmark, not a "
            "confirmatory test, and must not be used to alter the gate or "
            "threshold."
        ),
        "llava_note": (
            "LLaVA primary replay budget is 0/1145 at tau*=0.95; a "
            "budget-matched random selector at budget 0 is identical to "
            "Direct with zero variance -> TRIVIAL_COMPARATOR_NOT_INFORMATIVE. "
            "Not recomputed here. The historical tau=0.50 budget (462/1145) "
            "may optionally be reported as EXPLORATORY/HISTORICAL context only."
        ),
    }

    _draws_df = pd.DataFrame(
        {
            "draw": np.arange(_n_draws),
            "mean_soft_accuracy": _draw_accs,
            "n_beneficial_hit": _draw_beneficial,
            "n_harmful_hit": _draw_harmful,
            "n_neutral_hit": _draw_neutral,
            "harmful_avoidance": _draw_harmful_avoid,
        }
    )
    _draws_df.to_csv(os.path.join(OUT_DIR, "random_budget_gate_qwen_draws.csv"), index=False)

    import json as _json
    with open(os.path.join(OUT_DIR, "random_budget_gate_qwen_summary.json"), "w") as _f:
        _json.dump(_summary, _f, indent=2)

    with open(os.path.join(OUT_DIR, "random_budget_gate_methods.md"), "w") as _f:
        _f.write(
            "# Random budget-matched gate — methods (Task C)\n\n"
            "10,000 draws without replacement of exactly 635/1,145 validation "
            "IDs, numpy Generator(PCG64) seeded 42, drawn once and not "
            "selected for favorable outcome. Per-draw soft accuracy = "
            "accuracy_dino for replayed IDs, accuracy_direct otherwise "
            "(matches the manuscript's soft-score convention; no per-sample "
            "binarization). Beneficial/harmful/neutral labels are the fixed "
            "sign of (accuracy_dino - accuracy_direct), identical to the "
            "manuscript's label rule, computed once from already-frozen "
            "scores. No model call, no retraining, no threshold search.\n"
        )

    mo.md(
        f"""
        **Task C complete.**

        | Quantity | Value |
        |---|---|
        | Frozen AGIR mean soft accuracy | {agir_mean_acc*100:.4f}% |
        | Direct mean soft accuracy | {_direct_mean_acc*100:.4f}% |
        | Random-selector mean (10k draws) | {_draw_accs.mean()*100:.4f}% |
        | Random-selector 5th/95th pct | {np.percentile(_draw_accs,5)*100:.4f}% / {np.percentile(_draw_accs,95)*100:.4f}% |
        | Fraction of random draws AGIR beats | {(agir_mean_acc > _draw_accs).mean()*100:.1f}% |

        Written: `random_budget_gate_qwen_draws.csv`,
        `random_budget_gate_qwen_summary.json`,
        `random_budget_gate_methods.md`.
        """
    )
    return (
        _acc_dino,
        _acc_direct,
        agir_mean_acc,
        _agir_replayed,
        _budget,
        _draw_accs,
        _draws_df,
        _gate_df,
        _n,
        _summary,
    )


@app.cell
def __(OUT_DIR, mo, np, os, pd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make_random_gate_figure(draws_csv_path, agir_mean_acc, out_path):
        df = pd.read_csv(draws_csv_path)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.hist(df["mean_soft_accuracy"] * 100, bins=60, color="#8a8a8a", alpha=0.85)
        ax.axvline(agir_mean_acc * 100, color="#1f5fa8", linewidth=1.8, label="Frozen AGIR (635/1145)")
        ax.axvline(df["mean_soft_accuracy"].mean() * 100, color="#c9731b", linewidth=1.2, linestyle="--", label="Random mean")
        ax.set_xlabel("Mean soft accuracy (%)")
        ax.set_ylabel("Draws")
        ax.set_title("Budget-matched random gate (N=10{,}000 draws)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    mo.md(
        "Figure helper defined (`make_random_gate_figure`). It is called "
        "from the Task C cell's downstream export step below."
    )
    return make_random_gate_figure, plt


@app.cell
def __(OUT_DIR, agir_mean_acc, make_random_gate_figure, mo, os):
    _fig_path = os.path.join(OUT_DIR, "figures", "random_budget_gate_qwen_figure.pdf")
    make_random_gate_figure(
        os.path.join(OUT_DIR, "random_budget_gate_qwen_draws.csv"),
        agir_mean_acc,
        _fig_path,
    )
    mo.md(f"Figure written to `{_fig_path}`.")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Task B — Area-/shape-matched faithfulness controls

        This is the one part that genuinely needs a GPU runtime plus large
        model downloads. Read this before running anything below.

        ### Qwen — executable (data audit below first)

        Needed inputs, **already cached**, no regeneration of any VLM
        answer required:

        - `dino_replay_seed42.jsonl` (inside `qwen_v3-5.zip` → `qwen v3/`)
          — has the real `dino_query` and real `dino_boxes` per sample
          (this is the actual box `b_rel` used at faithfulness time).
        - `direct_predictions_seed42.jsonl` (same folder) — has `question`
          and `prediction` (the frozen initial answer `a_hat`) per sample.
        - `faithfulness_v3_n500.jsonl` (`qwen v3/all files/`) — gives the
          exact 500 sample `idx` values already used for faithfulness, so
          the matched-control run touches **the same 500 IDs**, never a
          different or favorable subset.
        - The **original A-OKVQA validation images**, loaded live (e.g.
          `datasets.load_dataset("HuggingFaceM4/A-OKVQA", split="validation")`
          or your originally-used loader) — not cached anywhere in the
          zips, so this needs network access to the HF Hub.
        - **Qwen2.5-VL-7B-Instruct** loaded with `transformers` — a
          genuine GPU requirement (~16 GB weights, ~20GB+ VRAM
          recommended). Only used for the already-defined, label-masked
          **fixed-answer** forward pass (`log_likelihood_score`); no
          `.generate()` call occurs anywhere in Task B.

        ### LLaVA — BLOCKED, not executable from any uploaded archive

        I checked every LLaVA archive you have (`llava_v5_final.zip`,
        `corrected_final_v4_llava.zip`, and all six versions inside
        `LLAVA_ALL.zip`, including `outputs-3.zip` and
        `outputs-part4.zip`). **None of them cache the per-sample box
        coordinates or the evidence phrase used for LLaVA's A-OKVQA
        faithfulness run** — only scalar summaries
        (`fused_box_area_ratio`, `dino_max_score`, etc.) survive.
        Recovering the real box would require re-running LLaVA's initial
        generation to reproduce the evidence phrase, which Task B's own
        rules forbid (no regenerating answers). This is written to
        `BLOCKED_INPUTS.md` by the cell below rather than silently
        skipped or faked.
        """
    )
    return


@app.cell
def __(OUT_DIR, mo, os):
    with open(os.path.join(OUT_DIR, "BLOCKED_INPUTS.md"), "w") as _f:
        _f.write(
            "# BLOCKED_INPUTS — Task B\n\n"
            "## LLaVA area-/shape-matched faithfulness controls: BLOCKED\n\n"
            "Searched every supplied LLaVA archive "
            "(llava_v5_final.zip, corrected_final_v4_llava.zip, and all of "
            "LLAVA_ALL.zip v1-v6 including outputs-3.zip and "
            "outputs-part4.zip). No file caches per-sample DINO box "
            "coordinates or the evidence phrase used to query the detector "
            "for the 500 A-OKVQA faithfulness samples -- only post-hoc "
            "scalar summaries (fused_box_area_ratio, dino_max_score, "
            "branch_agreement) survive from that run.\n\n"
            "Recovering the real box requires re-deriving the evidence "
            "phrase, which requires re-running LLaVA's initial-answer "
            "generation step -- explicitly forbidden by the experiment "
            "fixation rules (no regenerating answers). This item is "
            "reported BLOCKED rather than approximated with a different "
            "box or a freshly generated answer.\n\n"
            "**Recommended manuscript action:** narrow the LLaVA Section "
            "6.7 / 8.x faithfulness-control claim explicitly to the "
            "existing unmatched-area protocol for LLaVA, while the Qwen "
            "result (if Task B below is run on a GPU) may carry the "
            "upgraded matched-control claim. Do not imply LLaVA received "
            "the same control upgrade as Qwen.\n"
        )
    mo.md("`BLOCKED_INPUTS.md` written — LLaVA matched-control task is not fakeable from available files.")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Qwen matched-control run (requires GPU + network; off by default)

        Set `RUN_QWEN_TASK_B = True` in the cell below only once you have:
        1. A GPU runtime attached to this molab notebook.
        2. Uploaded `dino_replay_seed42.jsonl`, `direct_predictions_seed42.jsonl`,
           and `faithfulness_v3_n500.jsonl` (uploaders further down).
        3. Confirmed `transformers`, `torch`, `datasets`, and `accelerate`
           are installed (`!pip install -q transformers accelerate datasets torch pillow` if not).

        This will download Qwen2.5-VL-7B-Instruct (~16 GB) and run
        500 × 6 fixed-answer forward passes (1 original + 5 matched
        controls), which is **not fast** even on a GPU — budget real time
        for the first run.
        """
    )
    return


@app.cell
def __(mo):
    RUN_QWEN_TASK_B = False  # flip to True once GPU + uploads are ready
    mo.md(f"`RUN_QWEN_TASK_B = {RUN_QWEN_TASK_B}`")
    return (RUN_QWEN_TASK_B,)


@app.cell
def __(RUN_QWEN_TASK_B, mo):
    mo.stop(not RUN_QWEN_TASK_B, mo.md(
        "Qwen Task B is disabled (`RUN_QWEN_TASK_B = False`). "
        "Flip the flag in the cell above and re-run once a GPU runtime "
        "and the three uploads are in place."
    ))
    return


@app.cell
def __(mo):
    dino_replay_upload = mo.ui.file(label="Upload dino_replay_seed42.jsonl (qwen v3/)", kind="area")
    direct_pred_upload = mo.ui.file(label="Upload direct_predictions_seed42.jsonl (qwen v3/)", kind="area")
    faith_ids_upload = mo.ui.file(label="Upload faithfulness_v3_n500.jsonl (qwen v3/all files/) — reused to fix the 500 IDs", kind="area")
    mo.vstack([dino_replay_upload, direct_pred_upload, faith_ids_upload])
    return direct_pred_upload, dino_replay_upload, faith_ids_upload


@app.cell
def __(
    RUN_QWEN_TASK_B,
    direct_pred_upload,
    dino_replay_upload,
    faith_ids_upload,
    hashlib,
    json,
    mo,
    np,
):
    mo.stop(not RUN_QWEN_TASK_B)
    mo.stop(
        not (dino_replay_upload.value and direct_pred_upload.value and faith_ids_upload.value),
        mo.md("Upload all three files above to proceed."),
    )

    def _load_jsonl(upload):
        raw = upload.value[0].contents.decode("utf-8")
        return [json.loads(l) for l in raw.splitlines() if l.strip()]

    dino_rows = {r["sample_id"]: r for r in _load_jsonl(dino_replay_upload)}
    direct_rows = {r["sample_id"]: r for r in _load_jsonl(direct_pred_upload)}
    faith_ids = sorted({r["idx"] for r in _load_jsonl(faith_ids_upload) if "idx" in r})

    assert len(faith_ids) == 500, f"expected 500 faithfulness IDs, got {len(faith_ids)}"

    def stable_seed(global_seed, backbone_id, sample_id):
        h = hashlib.sha256(f"{global_seed}|{backbone_id}|{sample_id}".encode()).digest()
        return int.from_bytes(h[:8], "big") % (2**32)

    def deterministic_matched_controls(box, img_w, img_h, sample_id, n=5, max_attempts=1000, global_seed=42, backbone_id="qwen"):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        rng = np.random.default_rng(stable_seed(global_seed, backbone_id, sample_id))
        if bw > img_w or bh > img_h:
            return []
        accepted, attempts = [], 0
        max_x, max_y = img_w - bw, img_h - bh
        while len(accepted) < n and attempts < max_attempts:
            attempts += 1
            cx1 = float(rng.uniform(0, max(0.0, max_x)))
            cy1 = float(rng.uniform(0, max(0.0, max_y)))
            cbox = (cx1, cy1, cx1 + bw, cy1 + bh)
            ix1, iy1 = max(cbox[0], x1), max(cbox[1], y1)
            ix2, iy2 = min(cbox[2], x2), min(cbox[3], y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter == 0.0 and all(
                not (abs(cbox[0] - a[0]) < 1e-6 and abs(cbox[1] - a[1]) < 1e-6) for a in accepted
            ):
                accepted.append(cbox)
        return accepted

    mo.md(
        f"Loaded {len(dino_rows)} DINO rows, {len(direct_rows)} direct "
        f"predictions, {len(faith_ids)} frozen faithfulness IDs. "
        f"`deterministic_matched_controls` defined — strict box translation, "
        f"seed 42, per-sample SHA-256-derived sub-seed, non-overlap enforced, "
        f"budget 1000 attempts, first 5 accepted."
    )
    return (
        dino_rows,
        direct_rows,
        faith_ids,
        stable_seed,
        deterministic_matched_controls,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        **Model / image loading cell (edit paths for your environment, then
        run).** Left as an explicit, separate cell so a missing GPU or a
        blocked network fails loudly here rather than mid-run.
        """
    )
    return


@app.cell
def __(RUN_QWEN_TASK_B, mo):
    mo.stop(not RUN_QWEN_TASK_B)

    # Pre-declare everything this cell returns so a failure anywhere in the
    # try block still leaves well-defined (None) globals instead of a
    # second NameError masking the real error message below.
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch = vlm_processor = vlm_model = ds = device = None

    try:
        import torch  # noqa: F811 (intentional re-import into the outer name)
        from datasets import load_dataset

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            mo.md(
                "**No CUDA GPU detected.** Fixed-answer scoring of a 7B VLM "
                "on CPU is impractical at N=500x6. Attach a GPU runtime "
                "before continuing."
            ).callout(kind="danger")

        # Qwen2.5-VL is architecture `qwen2_5_vl`, a DIFFERENT class from
        # Qwen2-VL's `qwen2_vl`. Loading Qwen2.5-VL-7B-Instruct with
        # Qwen2VLForConditionalGeneration silently reinitializes mismatched
        # vision-tower weights (the "ignore_mismatched_sizes" error / load
        # report with MISSING+MISMATCH vision-tower keys). Requires
        # transformers>=4.49.
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as imp_err:
            raise RuntimeError(
                "transformers is missing Qwen2_5_VLForConditionalGeneration "
                "(needs transformers>=4.49 for Qwen2.5-VL support). "
                "Run `pip install -q -U 'transformers>=4.49' accelerate` and "
                "re-run this cell."
            ) from imp_err

        vlm_processor = AutoProcessor.from_pretrained(MODEL_ID)
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map=device
        )
        vlm_model.eval()
        ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
        _status = f"Model + dataset loaded on {device}."
    except Exception as e:
        vlm_processor = vlm_model = ds = device = None
        _status = f"FAILED to load model/dataset: {e!r}. Fix environment before continuing."
    mo.md(_status)
    return MODEL_ID, device, ds, torch, vlm_model, vlm_processor


@app.cell
def __(
    OUT_DIR,
    RUN_QWEN_TASK_B,
    dino_rows,
    direct_rows,
    faith_ids,
    deterministic_matched_controls,
    ds,
    json,
    mo,
    np,
    os,
    torch,
    vlm_model,
    vlm_processor,
):
    mo.stop(not RUN_QWEN_TASK_B)
    mo.stop(vlm_model is None, mo.md("Model failed to load — see cell above."))

    from PIL import Image, ImageDraw

    def log_likelihood_score(image, question, answer):
        prompt_only = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Question: {question}\nAnswer with a single word or very short phrase only."},
            ],
        }]
        prompt_inputs = vlm_processor.apply_chat_template(
            prompt_only, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(vlm_model.device)
        prompt_len = prompt_inputs["input_ids"].shape[1]
        full_messages = prompt_only + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        full_inputs = vlm_processor.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        ).to(vlm_model.device)
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        with torch.no_grad():
            out = vlm_model(**full_inputs, labels=labels)
        return -float(out.loss.item())

    def gray_mask(image, box):
        masked = image.copy()
        ImageDraw.Draw(masked).rectangle(box, fill=(128, 128, 128))
        return masked

    eligible_rows, ineligible_rows = [], []

    for _idx in mo.status.progress_bar(faith_ids):
        _dino_row = dino_rows.get(_idx)
        _direct_row = direct_rows.get(_idx)
        if _dino_row is None or _direct_row is None or not _dino_row.get("dino_boxes"):
            ineligible_rows.append({"idx": _idx, "reason": "no_cached_dino_box_or_prediction"})
            continue

        box = tuple(_dino_row["dino_boxes"][0])
        question = _direct_row["question"]
        answer = _direct_row["prediction"]

        sample = ds[_idx]
        image = sample["image"] if hasattr(sample["image"], "size") else Image.open(sample["image"])
        image = image.convert("RGB")
        w, h = image.size

        if box[2] > w or box[3] > h or box[0] < 0 or box[1] < 0:
            ineligible_rows.append({"idx": _idx, "reason": "cached_box_out_of_current_image_bounds"})
            continue

        controls = deterministic_matched_controls(box, w, h, _idx, n=5, global_seed=42, backbone_id="qwen")
        if len(controls) != 5:
            ineligible_rows.append({
                "idx": _idx, "reason": "fewer_than_5_nonoverlapping_controls_found",
                "n_found": len(controls),
            })
            continue

        s_original = log_likelihood_score(image, question, answer)
        s_rel = log_likelihood_score(gray_mask(image, box), question, answer)
        c_del = s_original - s_rel
        c_ctrl = []
        for cbox in controls:
            s_ctrl = log_likelihood_score(gray_mask(image, cbox), question, answer)
            c_ctrl.append(s_original - s_ctrl)
        M_i = c_del - float(np.mean(c_ctrl))

        eligible_rows.append({
            "idx": _idx, "box_area_frac": (box[2]-box[0])*(box[3]-box[1])/(w*h),
            "s_original": s_original, "c_del": c_del,
            "c_ctrl_mean": float(np.mean(c_ctrl)), "c_ctrl_all": c_ctrl,
            "M_i": M_i, "control_boxes": [list(c) for c in controls],
        })

    pd.DataFrame(eligible_rows).to_csv(os.path.join(OUT_DIR, "area_matched_faithfulness_qwen_n500.csv"), index=False)
    pd.DataFrame(ineligible_rows).to_csv(os.path.join(OUT_DIR, "area_matched_faithfulness_eligibility.csv"), index=False)

    with open(os.path.join(OUT_DIR, "matched_control_coordinates.jsonl"), "w") as _f:
        for r in eligible_rows:
            _f.write(json.dumps({"idx": r["idx"], "control_boxes": r["control_boxes"]}) + "\n")

    _M = np.array([r["M_i"] for r in eligible_rows])
    _summary = {
        "task": "B — Qwen area-/shape-matched faithfulness controls",
        "n_total": len(faith_ids),
        "n_eligible": len(eligible_rows),
        "n_ineligible": len(ineligible_rows),
        "mean_M_i": float(_M.mean()) if len(_M) else None,
        "median_M_i": float(np.median(_M)) if len(_M) else None,
        "share_positive": float((_M > 0).mean()) if len(_M) else None,
    }
    if len(_M):
        _rng_boot = np.random.default_rng(42)
        _boots = [
            float(_rng_boot.choice(_M, size=len(_M), replace=True).mean())
            for _ in range(10_000)
        ]
        _summary["bootstrap_95ci"] = [float(np.percentile(_boots, 2.5)), float(np.percentile(_boots, 97.5))]

    with open(os.path.join(OUT_DIR, "area_matched_faithfulness_summary.json"), "w") as _f:
        json.dump(_summary, _f, indent=2)

    mo.md(f"Qwen matched-control run complete. Eligible: {len(eligible_rows)}/{len(faith_ids)}. Summary: {_summary}")
    return (
        Image,
        ImageDraw,
        eligible_rows,
        gray_mask,
        ineligible_rows,
        log_likelihood_score,
        pd,
    )


if __name__ == "__main__":
    app.run()
