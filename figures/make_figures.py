"""Regenerates the Qwen2.5-VL-7B result figures from the frozen per-sample outputs."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(os.path.abspath(__file__))
FA = os.path.join(HERE, "..", "qwen", "frozen_artifacts") + "/"
OUT = os.path.join(HERE, "figures_out") + "/"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
C = {"conf": "#1f5fa8", "rich": "#c9731b", "direct": "#555555", "dino": "#8a8a8a",
     "ben": "#2a8f5a", "har": "#b3362c", "neu": "#b8b8b8"}

# ---------------- Fig: development threshold sweep (selection data only) ----------------
z = FA + "threshold_sweeps/"   # corrected_gate_threshold_sweep_{confidence,richer8}.csv (see README)
sc = pd.read_csv(z + "corrected_gate_threshold_sweep_confidence.csv")
sr = pd.read_csv(z + "corrected_gate_threshold_sweep_richer8.csv")
direct_dev = sr.loc[sr.replay_rate == 0, "policy_accuracy"].iloc[0] * 100   # zero-replay rows
dino_dev = sc.loc[sc.replay_rate == 1, "policy_accuracy"].iloc[0] * 100     # always-replay rows
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.55))
a = ax[0]
a.plot(sc.tau, sc.policy_accuracy * 100, "-o", ms=3, c=C["conf"], label="Confidence-only gate")
a.plot(sr.tau, sr.policy_accuracy * 100, "-s", ms=3, c=C["rich"], label="Eight-feature gate")
a.axhline(direct_dev, ls="--", c=C["direct"], lw=1); a.axhline(dino_dev, ls=":", c=C["dino"], lw=1.2)
a.text(0.955, direct_dev + 0.03, "Direct", ha="right", va="bottom", fontsize=7, color=C["direct"])
a.text(0.955, dino_dev - 0.03, "DINO-always", ha="right", va="top", fontsize=7, color=C["dino"])
a.axvline(0.35, c=C["conf"], lw=0.8, alpha=.5); a.axvline(0.50, c=C["rich"], lw=0.8, alpha=.5)
a.set_xlabel(r"Threshold $\tau$"); a.set_ylabel("Development policy accuracy (%)")
a.set_ylim(56.4, 57.5); a.legend(loc="lower left", frameon=False); a.set_title("(a) Accuracy surface is nearly flat", loc="left")
b = ax[1]
b.plot(sc.tau, sc.replay_rate * 100, "-o", ms=3, c=C["conf"], label="Replay rate")
b.plot(sc.tau, sc.harmful_avoidance * 100, "-^", ms=3, c=C["har"], label="Harmful avoidance")
b.plot(sc.tau, sc.beneficial_recall * 100, "-v", ms=3, c=C["ben"], label="Beneficial recall")
b.axvline(0.35, c=C["conf"], lw=0.8, alpha=.5); b.text(0.36, 3, r"$\tau^\star{=}0.35$", fontsize=7, color=C["conf"])
b.set_xlabel(r"Threshold $\tau$"); b.set_ylabel("Confidence-only gate (%)"); b.set_ylim(-3, 103)
b.legend(loc="center right", frameon=False); b.set_title("(b) Safety and cost move far more than accuracy", loc="left")
fig.tight_layout(); fig.savefig(OUT + "fig_qwen_dev_sweep.pdf"); plt.close(fig)
print("dev direct/dino:", round(direct_dev, 3), round(dino_dev, 3),
      "selected conf acc:", sc.loc[np.isclose(sc.tau, .35), "policy_accuracy"].iloc[0] * 100)

# ---------------- Fig: label composition, replayed vs bypassed (validation) ----------------
v = pd.read_csv(FA + "final_gate_v9/corrected_gate_validation_predictions.csv")
ben = (v.acc_direct == 0) & (v.acc_dino == 1); har = (v.acc_direct == 1) & (v.acc_dino == 0)
dec = v.corrected_conf_decision.astype(bool)
groups = [("All\n(n=1145)", np.ones(len(v), bool)), ("Replayed\n(n=%d)" % dec.sum(), dec.values), ("Bypassed\n(n=%d)" % (~dec).sum(), ~dec.values)]
fig, a = plt.subplots(figsize=(3.4, 2.4)); x = np.arange(3); w = .36
bv = [ben[m].mean() * 100 for _, m in groups]; hv = [har[m].mean() * 100 for _, m in groups]
bn = [int(ben[m].sum()) for _, m in groups]; hn = [int(har[m].sum()) for _, m in groups]
r1 = a.bar(x - w / 2, bv, w, color=C["ben"], label="Beneficial"); r2 = a.bar(x + w / 2, hv, w, color=C["har"], label="Harmful")
for r, n in zip(r1, bn): a.text(r.get_x() + r.get_width() / 2, r.get_height() + .15, str(n), ha="center", fontsize=7)
for r, n in zip(r2, hn): a.text(r.get_x() + r.get_width() / 2, r.get_height() + .15, str(n), ha="center", fontsize=7)
a.set_xticks(x); a.set_xticklabels([g for g, _ in groups]); a.set_ylabel("Share of samples in group (%)")
a.set_ylim(0, 11); a.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(OUT + "fig_qwen_outcomes.pdf"); plt.close(fig)
print("composition:", list(zip([g[0].split()[0] for g in groups], bn, hn)))

# ---------------- Fig: latency decomposition (N=200) ----------------
e = pd.DataFrame([json.loads(l) for l in open(FA + "final_gate_v9/efficiency_qwen_v9_true_gate_order_n200.jsonl")])
comp = {"Answer generation": [e.answer_generation_latency.mean()] * 3,
        "Confidence + gate": [0, (e.confidence_latency + e.gate_model_latency).mean(), 0],
        "Evidence phrase": [0, e.evidence_phrase_latency_agir.mean(), e.evidence_phrase_latency_always.mean()],
        "Grounding DINO": [0, e.dino_latency_agir.mean(), e.dino_latency_always.mean()],
        "Replay generation": [0, e.replay_latency_agir.mean(), e.replay_latency_always.mean()]}
cols = ["#8fb8e0", "#e8c46a", "#a8d3a0", "#d99a8f", "#b9a3d6"]
fig, a = plt.subplots(figsize=(3.4, 2.5)); bottom = np.zeros(3); names = ["Direct", "AGIR-DINO", "DINO-always"]
for (k, vals), c in zip(comp.items(), cols):
    a.bar(names, vals, bottom=bottom, color=c, label=k, width=.6, edgecolor="white", lw=.5); bottom += np.array(vals)
for i, t in enumerate(bottom): a.text(i, t + .008, f"{t:.3f} s", ha="center", fontsize=7.5)
a.set_ylabel("Mean latency per sample (s)"); a.set_ylim(0, .56); a.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig_qwen_latency.pdf"); plt.close(fig)
print("latency totals:", bottom)

# ---------------- Fig: paired faithfulness differences (N=500) ----------------
f = pd.DataFrame([json.loads(l) for l in open(FA + "faithfulness_refcoco/faithfulness_v3_n500.jsonl")])
d = (f.c_del - f.c_irr).values; lo, hi = -2, 5
fig, a = plt.subplots(figsize=(3.4, 2.4))
a.hist(np.clip(d, lo, hi), bins=np.linspace(lo, hi, 43), color="#1f5fa8", alpha=.85, edgecolor="white", lw=.4)
a.axvline(0, c="k", lw=.8); a.axvline(d.mean(), c=C["har"], lw=1.3, label=f"mean = {d.mean():.3f}")
a.axvline(np.median(d), c=C["ben"], lw=1.3, ls="--", label=f"median = {np.median(d):.3f}")
a.set_xlabel(r"Paired difference $C_{\mathrm{del}}-C_{\mathrm{irr}}$"); a.set_ylabel("Number of samples")
a.legend(frameon=False, loc="upper right"); fig.tight_layout(); fig.savefig(OUT + "fig_qwen_faithfulness.pdf"); plt.close(fig)
print("faith: frac>0", (d > 0).mean(), "outside [lo,hi]:", int(((d < lo) | (d > hi)).sum()), "min/max", d.min(), d.max())

# ---------------- Fig: RefCOCOg IoU distributions (N=400) ----------------
r = pd.DataFrame([json.loads(l) for l in open(FA + "faithfulness_refcoco/refcoco_v3_full_metrics.jsonl")])
ex = [("dino", "Detector"), ("native", "Native pointing"), ("fused", "Fusion"), ("attribution", "Grad. attribution"), ("clip", "CLIP baseline")]
fig, a = plt.subplots(figsize=(3.4, 2.5))
bp = a.boxplot([r[k + "_iou"] for k, _ in ex], vert=False, labels=[n for _, n in ex], widths=.55, patch_artist=True,
               showmeans=True, meanprops=dict(marker="D", mfc="white", mec="k", ms=4), medianprops=dict(c="k"),
               flierprops=dict(ms=2, alpha=.4))
for p, c in zip(bp["boxes"], ["#1f5fa8", "#c9731b", "#6a4c93", "#9a9a9a", "#c8c8c8"]): p.set_facecolor(c); p.set_alpha(.85)
a.invert_yaxis(); a.set_xlabel("IoU with ground-truth box"); a.set_xlim(-.02, 1.02)
fig.tight_layout(); fig.savefig(OUT + "fig_qwen_refcoco.pdf"); plt.close(fig)
print("done")
