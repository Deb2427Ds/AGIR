# Faithfulness score-definition audit (Task A)

**Verdict: MISMATCH (manuscript text only) -> NO_CHANGE to numeric results.**

`log_likelihood_score()` returns `-out.loss`, which under HuggingFace's default causal-LM loss reduction is already the token-mean negative log-likelihood over the (label-masked) answer span -- i.e. exactly manuscript Eq. (2)'s `c(I,q,a)`. `run_faithfulness_v3()` builds `s_original`, `s_relevant_deleted`, `s_irrelevant_deleted`, `s_region_only` by calling `log_likelihood_score()` directly, with no further division by answer word count anywhere in the function body. `c_del` and `c_irr` are plain differences of two such token-mean scores. Manuscript Eq. (22) as currently typeset (`S(I') = c(I',q,a_hat) / max(1, n_words(a_hat))`) therefore does not match the implementation: it implies an extra division that was never executed. This is a documentation bug, not a methodology bug -- no faithfulness numbers are affected and no rerun is required. Patch: retype Eq. (22) as `S(I') = c(I', q, a_hat)` and note explicitly that c is already a per-token mean.

Separately, the Qwen faithfulness code's existing 'irrelevant' control is NOT the fixed 0.2W x 0.2H corner box the pre-submission checklist assumed: it is already a box of the SAME width/height as the DINO box, placed via a single unseeded `np.random.uniform` translation with no enforced non-overlap. Task B below upgrades this to a seeded, non-overlapping, 5-draw matched-control protocol; it does not introduce area-matching that was wholly absent before.

# Task A — numerical spot check


## Qwen (faithfulness_v3_n500.jsonl)
- N rows checked: 500
- max |c_del - (s_original - s_relevant_deleted)|: 0.000e+00
- mean abs diff: 0.000e+00
- Interpretation: differences at float-precision level confirm `c_del` is an *undivided* difference of two token-mean scores, consistent with the code trace above (no word-count rescale).


## LLaVA (faithfulness_llava_dino_n500.jsonl): uploaded (500 rows) but required fields (`s_original`,`s_relevant_deleted`,`c_del`) not all present — this file only carries post-hoc scores (expected for the LLaVA summary cache, which never stored `s_original`/`s_relevant_deleted`).
