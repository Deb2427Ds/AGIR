# QWEN_FREEZE_REPORT_v9.md

## Efficiency (corrected execution order, N=200)
On the fixed N=200 efficiency subset, measured with the true adaptive WHEN->WHERE->WHAT execution order (evidence-phrase generation and Grounding DINO invoked only after the confidence gate elects to replay), corrected AGIR-DINO runs 244.5% slower than Direct and 19.5% faster than always-on DINO replay, avoiding evidence-generation, DINO, and replay-generation calls on 46.0% of samples.

## Accuracy (unchanged from v8)
On Qwen, the corrected confidence-only AGIR-DINO policy achieves 64.63% accuracy versus 64.19% for Direct. The difference is numerical rather than statistically supported under the predeclared multiplicity-controlled paired testing.

## Fused comparison nuance
The unadjusted paired-bootstrap interval for AGIR-DINO versus always-on fused excludes zero, but the exact McNemar comparison does not survive the predeclared Holm correction (raw p=0.0595, Holm-adjusted p=0.1784); therefore we do not claim a statistically significant improvement.

## Sanity checklist
- [x] primary_target_remains_beneficial_1_harmful_0_neutral_0
- [x] tau_remains_0.35_not_retuned
- [x] frozen_n1145_predictions_unchanged
- [x] direct_timing_answer_generation_only
- [x] direct_timing_no_evidence_phrase
- [x] agir_evidence_only_when_gate_replayed_true
- [x] agir_executes_gate_before_evidence_generation
- [x] dino_always_generates_evidence_every_sample
- [x] dino_always_does_not_compute_gate
- [x] same_fixed_n200_ids_used_all_policies
- [x] cuda_sync_used_around_timed_gpu_work
- [x] n200_and_full_validation_replay_rates_reported_separately
- [x] old_v8_efficiency_numbers_marked_superseded
- [x] bootstrap_cis_labeled_unadjusted
- [x] holm_attached_to_mcnemar_family_only
- [x] richer8_remains_diagnostic_only
- [x] historical_v5_remains_legacy_only

## QWEN FREEZE: YES — CORRECTED CONFIDENCE-ONLY METHOD LOCKED

