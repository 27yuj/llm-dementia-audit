"""Paired subject-level bootstrap: is the fusion gain real or within noise?

Compares models on the SAME resampled subjects, so the shared difficulty of a
given draw cancels out. Reports the distribution of the AUROC difference and the
fraction of resamples where fusion wins.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

NAMES = ["phi3", "tfidf", "late", "early"]
LABEL = {"phi3": "Phi-3 L12 mean", "tfidf": "TF-IDF uni+bigram",
         "late": "late fusion", "early": "early fusion"}

oof = np.load("results/oof_fusion.npy")          # [model, repeat, n]
df = pd.read_csv("artifacts/manifest.csv")
df = df[(df.group == "Control") | (df.dx == "ProbableAD")].reset_index(drop=True)
y = (df.dx == "ProbableAD").astype(int).to_numpy()
subs = df.subject_id.unique()
by_sub = {s: np.flatnonzero(df.subject_id.to_numpy() == s) for s in subs}

rng = np.random.default_rng(0)
n_boot, n_repeats = 2000, oof.shape[1]
draws = {k: [] for k in NAMES}

for b in range(n_boot):
    pick = rng.choice(len(subs), len(subs), replace=True)
    idx = np.concatenate([by_sub[subs[p]] for p in pick])
    if len(set(y[idx])) < 2:
        continue
    r = b % n_repeats
    for i, k in enumerate(NAMES):
        draws[k].append(roc_auc_score(y[idx], oof[i, r][idx]))

draws = {k: np.array(v) for k, v in draws.items()}
print("subject bootstrap, 95% CI:")
for k in NAMES:
    lo, hi = np.percentile(draws[k], [2.5, 97.5])
    print(f"  {LABEL[k]:20s} {draws[k].mean():.4f}  [{lo:.4f}, {hi:.4f}]")

print("\npaired differences (same resampled subjects):")
rows = []
for a, b in [("late", "phi3"), ("late", "tfidf"), ("early", "phi3"),
             ("late", "early")]:
    d = draws[a] - draws[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    win = float((d > 0).mean())
    sig = "yes" if lo > 0 else "no"
    rows.append({"comparison": f"{LABEL[a]} - {LABEL[b]}",
                 "delta": round(float(d.mean()), 4),
                 "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                 "p_win": round(win, 3), "ci_excludes_zero": sig})
    print(f"  {LABEL[a]:14s} - {LABEL[b]:18s} "
          f"delta {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"wins {win * 100:.1f}%  significant: {sig}")

pd.DataFrame(rows).to_csv("results/fusion_significance.csv", index=False)
print("\nsaved -> results/fusion_significance.csv")
