"""Stacked fusion: learn the combination weights instead of assuming equal ones.

Equal-weight averaging penalises a weak member by construction. Here an inner CV
on each training fold produces out-of-fold base-model scores, a logistic
combiner is fit on those, and the combiner is applied to the outer test fold.
If surprisal carries no independent signal its weight collapses toward zero and
the stack simply matches the 2-way fusion -- so this is a fair test of whether
it adds anything.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

N_FOLDS, N_REPEATS, INNER = 5, 5, 5
PPL = ["nll_mean", "nll_median", "nll_std", "nll_p90", "frac_surprising"]

man = pd.read_csv("artifacts/manifest.csv")
df = man.merge(pd.read_csv("artifacts/perplexity.csv"), on="file_id")
mask = ((man.group == "Control") | (man.dx == "ProbableAD")).to_numpy()
X_emb = np.array(np.load("artifacts/L12_mean.npy")[mask], dtype=np.float64)
df = df[((df.group == "Control") | (df.dx == "ProbableAD")).to_numpy()].reset_index(drop=True)
df["label"] = (df.dx == "ProbableAD").astype(int)
y, groups = df.label.to_numpy(), df.subject_id.to_numpy()
texts = df.text.fillna("").to_numpy()
P = np.array(df[PPL].to_numpy(dtype=float), copy=True)


def base_scores(tr, te, ytr):
    """Three base models fit on `tr`, scoring `te`. Returns [n_te, 3]."""
    sc = StandardScaler().fit(X_emb[tr])
    s1 = LogisticRegression(max_iter=2000, C=0.01, class_weight="balanced").fit(
        sc.transform(X_emb[tr]), ytr).decision_function(sc.transform(X_emb[te]))
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(texts[tr])
    s2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
        vec.transform(texts[tr]), ytr).decision_function(vec.transform(texts[te]))
    scp = StandardScaler().fit(P[tr])
    s3 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        scp.transform(P[tr]), ytr).decision_function(scp.transform(P[te]))
    return np.c_[s1, s2, s3]


oof2 = np.zeros((N_REPEATS, len(y)))
oof3 = np.zeros((N_REPEATS, len(y)))
weights = []

for r in range(N_REPEATS):
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r)
    for tr, te in cv.split(np.zeros(len(y)), y, groups):
        ytr = y[tr]
        # inner CV -> honest base scores for the combiner's training set
        meta = np.zeros((len(tr), 3))
        inner = StratifiedGroupKFold(INNER, shuffle=True, random_state=200 + r)
        for itr, ite in inner.split(np.zeros(len(tr)), ytr, groups[tr]):
            meta[ite] = base_scores(tr[itr], tr[ite], ytr[itr])

        outer = base_scores(tr, te, ytr)
        ms = StandardScaler().fit(meta)

        c3 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            ms.transform(meta), ytr)
        oof3[r, te] = c3.decision_function(ms.transform(outer))
        weights.append(c3.coef_[0].copy())

        ms2 = StandardScaler().fit(meta[:, :2])
        c2 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            ms2.transform(meta[:, :2]), ytr)
        oof2[r, te] = c2.decision_function(ms2.transform(outer[:, :2]))
    print(f"  repeat {r} done", flush=True)

rows = []
for name, o in (("stacked phi3+tfidf", oof2), ("stacked phi3+tfidf+ppl", oof3)):
    a = [roc_auc_score(y, o[r]) for r in range(N_REPEATS)]
    rows.append({"model": name, "auroc": round(float(np.mean(a)), 4),
                 "auroc_sd": round(float(np.std(a)), 4)})
    print(f"  {name:26s} AUROC {np.mean(a):.4f} +/- {np.std(a):.4f}")

W = np.array(weights)
print(f"\nlearned combiner weights (mean over {len(W)} folds):")
for n, w, s in zip(["Phi-3", "TF-IDF", "surprisal"], W.mean(0), W.std(0)):
    print(f"  {n:10s} {w:+.3f} +/- {s:.3f}")
print(f"\nsurprisal weight as a share of total |weight|: "
      f"{abs(W.mean(0)[2]) / np.abs(W.mean(0)).sum() * 100:.1f}%")

pd.DataFrame(rows).to_csv("results/stacked_fusion.csv", index=False)
print("saved -> results/stacked_fusion.csv")
