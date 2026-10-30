"""
Fit the corrected confidence-only AGIR gate for Qwen2.5-VL-7B on A-OKVQA and
check it against the frozen artifacts.

Provenance note (read this): the original notebook that fitted the frozen gate
was not retained. This script is a reconstruction built only from retained
specification artifacts (CORRECTED_GATE_TARGET_SPEC.md, CORRECTED_GATE_SPLIT.json)
and the cached pool features/labels. It is verified numerically against the
frozen confidence_benefit_gate.pkl (see checks below).

Inputs (paths via CLI args or defaults in the current directory):
  intervention_cache_deduped.jsonl     (10,000-example pool: idx, label, ...)
  extended_features_cache.jsonl        (10,000 rows: idx, answer_confidence, ...)
  CORRECTED_GATE_SPLIT.json            (frozen train_ids / dev_ids, seed 42)
  full_validation_all_configs.jsonl    (1,145 validation rows: idx, answer_confidence)
Optional, for verification only:
  confidence_benefit_gate.pkl, corrected_gate_validation_predictions.csv

Usage: python 01b_fit_corrected_gate.py [data_dir]   (default: qwen/frozen_artifacts, searched recursively)
"""
import glob, json, os, pickle, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_artifacts")
TAU_FROZEN = 0.35  # frozen value from CORRECTED_GATE_FREEZE.json (see note in main())

def P(name):
    """Locate a file anywhere under the data directory."""
    hits = glob.glob(os.path.join(D, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

def jl(p): return [json.loads(l) for l in open(P(p))]

def main():
    pool = {r["idx"]: r for r in jl("intervention_cache_deduped.jsonl")}
    conf = {r["idx"]: r["answer_confidence"] for r in jl("extended_features_cache.jsonl")}
    split = json.load(open(P("CORRECTED_GATE_SPLIT.json")))
    tr, dv = split["train_ids"], split["dev_ids"]
    assert not set(tr) & set(dv) and len(tr) == 8000 and len(dv) == 2000

    # target: beneficial=1, harmful=0, neutral=0 (CORRECTED_GATE_TARGET_SPEC.md)
    Xtr = np.array([[conf[i]] for i in tr])
    ytr = np.array([1 if pool[i]["label"] == "beneficial" else 0 for i in tr])
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(Xtr), ytr)
    print("coef", model.coef_, "intercept", model.intercept_)
    print("scaler mean", scaler.mean_, "scale", scaler.scale_)

    # Validation decisions at the frozen tau
    val = {r["idx"]: r["answer_confidence"] for r in jl("full_validation_all_configs.jsonl")}
    ids = sorted(val)
    p = model.predict_proba(scaler.transform(np.array([[val[i]] for i in ids])))[:, 1]
    dec = p >= TAU_FROZEN
    print(f"validation replays at tau={TAU_FROZEN}: {int(dec.sum())}/{len(ids)}  (paper: 635/1145)")

    # Verification against frozen artifacts, if present
    try:
        b = pickle.load(open(P("confidence_benefit_gate.pkl"), "rb"))
        print("gate coef/intercept match frozen:",
              np.allclose(model.coef_, b["model"].coef_, atol=1e-6),
              np.allclose(model.intercept_, b["model"].intercept_, atol=1e-6))
        print("scaler match frozen:",
              np.allclose(scaler.mean_, b["scaler"].mean_, atol=1e-6),
              np.allclose(scaler.scale_, b["scaler"].scale_, atol=1e-6))
    except FileNotFoundError:
        print("frozen pickle not provided; skipping check")
    try:
        import pandas as pd
        c = pd.read_csv(P("corrected_gate_validation_predictions.csv")).set_index("sample_id")
        print("validation decisions identical to frozen CSV:",
              bool((c.loc[ids, "corrected_conf_decision"].values == dec).all()))
    except FileNotFoundError:
        print("frozen predictions CSV not provided; skipping check")

    # NOTE: tau=0.35 is taken from CORRECTED_GATE_FREEZE.json. Re-running the dev-grid
    # sweep from the retained pool caches does NOT reproduce the frozen dev policy
    # accuracy (0.5711667): the accuracy file used for the original dev sweep was not
    # retained, so the threshold is treated as a frozen constant, not re-derived.

if __name__ == "__main__":
    main()
