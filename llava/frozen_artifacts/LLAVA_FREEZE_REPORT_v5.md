# LLAVA_FREEZE_REPORT_v5.md

## Verdict: LLaVA replication PARTIAL — backbone-dependent result; the training-side policy prefers abstention

LLaVA replication PARTIAL — backbone-dependent result; the training-side policy prefers abstention

For LLaVA-v1.6-Mistral-7B, the held-out training-side policy search identifies no useful nonzero DINO-replay regime and selects tau=0.95, resulting in zero replay on all 1,145 validation examples. The frozen AGIR-DINO policy therefore matches Direct performance exactly (65.27%), while always-on DINO replay is significantly worse. A historical tau=0.50 operating point that forces nonzero replay also significantly degrades performance. These findings indicate that the utility of selective visual intervention is backbone-dependent and that AGIR can legitimately learn when not to intervene.

Corrected timing (order: answer -> confidence -> gate -> evidence/DINO/replay only if fired) on the same fixed N=200 sample IDs: Direct mean=0.1909s, AGIR-DINO mean=0.3839s (overhead vs Direct: 101.1%), DINO-always mean=0.8134s (AGIR speedup vs DINO-always: 52.8%). Because the frozen LLaVA policy always abstains, Direct inference is the practical deployment choice on this backbone; running the confidence gate adds overhead without triggering grounding.

## Sanity checklist
- [x] beneficial_1_harmful_0_neutral_0
- [x] gate_train_dev_disjoint
- [x] no_validation_labels_used_for_fit_or_threshold
- [x] threshold_grid_correct
- [x] selected_tau_in_grid
- [x] no_synthetic_tau_1_01_used_as_deployable_threshold
- [x] zero_replay_baseline_reported_separately_not_a_threshold
- [x] validation_decisions_from_saved_scaler_and_gate
- [x] validation_n_1145
- [x] ids_unique_and_aligned
- [x] direct_dino_predictions_reused_not_regenerated
- [x] actual_validation_replay_count_reported
- [x] paired_statistics_use_identical_samples
- [x] tau050_historical_negative_result_preserved
- [x] dino_always_timing_excludes_gate_overhead
- [x] direct_agir_dino_always_timing_same_fixed_n200
- [x] harmful_avoidance_100_described_as_trivial
- [x] beneficial_precision_undefined_when_zero_replay
- [x] refcoco_interpretation_limited
- [x] exact_checkpoint_name_reported
- [x] negative_results_preserved
- [x] direct_generates_answer_only
- [x] direct_does_not_generate_evidence_phrase
- [x] agir_computes_confidence_gate_before_evidence
- [x] agir_generates_evidence_only_if_gate_fires
- [x] dino_always_generates_evidence_for_every_sample
- [x] dino_always_does_not_compute_gate
- [x] same_fixed_n200_ids_used_for_all_policies
- [x] cuda_sync_used_around_timed_gpu_work
- [x] raw_n200_timing_rows_in_final_package
- [x] bootstrap_cis_labeled_unadjusted
- [x] holm_correction_attached_to_mcnemar_only
- [x] no_bootstrap_significant_holm_field
- [x] faithfulness_files_included_in_canonical_package
- [x] refcoco_files_included_in_canonical_package

## LLAVA FREEZE: YES

