# BLOCKED_INPUTS — Task B

## LLaVA area-/shape-matched faithfulness controls: BLOCKED

Searched every supplied LLaVA archive (llava_v5_final.zip, corrected_final_v4_llava.zip, and all of LLAVA_ALL.zip v1-v6 including outputs-3.zip and outputs-part4.zip). No file caches per-sample DINO box coordinates or the evidence phrase used to query the detector for the 500 A-OKVQA faithfulness samples -- only post-hoc scalar summaries (fused_box_area_ratio, dino_max_score, branch_agreement) survive from that run.

Recovering the real box requires re-deriving the evidence phrase, which requires re-running LLaVA's initial-answer generation step -- explicitly forbidden by the experiment fixation rules (no regenerating answers). This item is reported BLOCKED rather than approximated with a different box or a freshly generated answer.

**Recommended manuscript action:** narrow the LLaVA Section 6.7 / 8.x faithfulness-control claim explicitly to the existing unmatched-area protocol for LLaVA, while the Qwen result (if Task B below is run on a GPU) may carry the upgraded matched-control claim. Do not imply LLaVA received the same control upgrade as Qwen.
