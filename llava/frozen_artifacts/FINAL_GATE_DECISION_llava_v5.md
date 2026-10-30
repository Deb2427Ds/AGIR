# FINAL_GATE_DECISION_llava_v5.md

LLaVA replication PARTIAL — backbone-dependent result; the training-side policy prefers abstention

For LLaVA-v1.6-Mistral-7B, the held-out training-side policy search identifies no useful nonzero DINO-replay regime and selects tau=0.95, resulting in zero replay on all 1,145 validation examples. The frozen AGIR-DINO policy therefore matches Direct performance exactly (65.27%), while always-on DINO replay is significantly worse. A historical tau=0.50 operating point that forces nonzero replay also significantly degrades performance. These findings indicate that the utility of selective visual intervention is backbone-dependent and that AGIR can legitimately learn when not to intervene.

Corrected timing (order: answer -> confidence -> gate -> evidence/DINO/replay only if fired) on the same fixed N=200 sample IDs: Direct mean=0.1909s, AGIR-DINO mean=0.3839s (overhead vs Direct: 101.1%), DINO-always mean=0.8134s (AGIR speedup vs DINO-always: 52.8%). Because the frozen LLaVA policy always abstains, Direct inference is the practical deployment choice on this backbone; running the confidence gate adds overhead without triggering grounding.