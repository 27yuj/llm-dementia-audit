"""
Does combining Phi-3 geometry with lexical features beat either alone?

Two fusion schemes, both under the same subject-disjoint protocol:

  late  -- fit the two models separately, z-score each one's decision function
           on the training fold, average. No new hyperparameters, so nothing
           can be tuned into the test fold.
  early -- concatenate standardised Phi-3 features with TF-IDF and fit one
           model. The regularisation strength C is chosen by an INNER CV on the
           training fold, never on the outer test fold.

Phi-3 and TF-IDF scores correlate at r~0.86, so the ceiling here is limited by
construction: fusion can only exploit the ~25% of variance they do not share.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

LAYER, POOLING = 12, "mean"
N_FOLDS, N_REPEATS, INNER_FOLDS = 5, 5, 3
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# pre-cached slice (see artifacts/L12_mean.npy) keeps this script torch-free
df = pd.read_csv("artifacts/manifest.csv")
_all = np.load("artifacts/L12_mean.npy")
keep = ((df.group == "Control") | (df.dx == "ProbableAD")).to_numpy()
X = np.array(_all[keep], dtype=np.float64)
df = df[keep].reset_index(drop=True)
df["label"] = (df.dx == "ProbableAD").astype(int)
texts = df.text.fillna("").to_numpy()
y, groups = df.label.to_numpy(), df.subject_id.to_numpy()


def phi3_scores(tr, te_list, ytr):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=0.01,
                             class_weight="balanced").fit(sc.transform(X[tr]), ytr)
    return [clf.decision_function(sc.transform(X[i])) for i in te_list]


def tfidf_scores(tr, te_list, ytr):
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(texts[tr])
    clf = LogisticRegression(max_iter=3000, C=1.0,
                             class_weight="balanced").fit(vec.transform(texts[tr]), ytr)
    return [clf.decision_function(vec.transform(texts[i])) for i in te_list]


def build_early(tr, te, ytr, C):
    """Concatenate standardised Phi-3 with TF-IDF, fit one model."""
    sc = StandardScaler().fit(X[tr])
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(texts[tr])
    Ztr = np.hstack([sc.transform(X[tr]), vec.transform(texts[tr]).toarray()])
    Zte = np.hstack([sc.transform(X[te]), vec.transform(texts[te]).toarray()])
    clf = LogisticRegression(max_iter=3000, C=C,
                             class_weight="balanced").fit(Ztr, ytr)
    return clf.decision_function(Zte)


def z(a, ref):
    return (a - ref.mean()) / (ref.std() + 1e-9)


results = {k: np.zeros((N_REPEATS, len(y))) for k in
           ("phi3", "tfidf", "late", "early")}
chosen_C = []

for r in range(N_REPEATS):
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r)
    for tr, te in cv.split(np.zeros(len(y)), y, groups):
        ytr = y[tr]

        p_tr, p_te = phi3_scores(tr, [tr, te], ytr)
        t_tr, t_te = tfidf_scores(tr, [tr, te], ytr)
        results["phi3"][r, te] = p_te
        results["tfidf"][r, te] = t_te
        results["late"][r, te] = (z(p_te, p_tr) + z(t_te, t_tr)) / 2

        # inner CV picks C for the concatenated model, on training data only
        best_C, best_auc = C_GRID[0], -np.inf
        inner = StratifiedGroupKFold(INNER_FOLDS, shuffle=True, random_state=100 + r)
        for C in C_GRID:
            oof_in = np.zeros(len(tr))
            for itr, ite in inner.split(np.zeros(len(tr)), ytr, groups[tr]):
                oof_in[ite] = build_early(tr[itr], tr[ite], ytr[itr], C)
            auc = roc_auc_score(ytr, oof_in)
            if auc > best_auc:
                best_auc, best_C = auc, C
        chosen_C.append(best_C)
        results["early"][r, te] = build_early(tr, te, ytr, best_C)
    print(f"  repeat {r} done", flush=True)

print(f"\nControl vs ProbableAD, subject-disjoint {N_FOLDS}x{N_REPEATS}, "
      f"Phi-3 L{LAYER} {POOLING}\n")
label = {"phi3": "Phi-3 alone", "tfidf": "TF-IDF alone",
         "late": "late fusion (score average)", "early": "early fusion (concat)"}
rows = []
for k, oof in results.items():
    per = [roc_auc_score(y, oof[r]) for r in range(N_REPEATS)]
    rows.append({"model": label[k], "auroc": round(float(np.mean(per)), 4),
                 "auroc_sd": round(float(np.std(per)), 4)})
    print(f"  {label[k]:28s} AUROC {np.mean(per):.4f} +/- {np.std(per):.4f}")

base = max(rows[0]["auroc"], rows[1]["auroc"])
best_fusion = max(rows[2]["auroc"], rows[3]["auroc"])
print(f"\nbest single model {base:.4f} -> best fusion {best_fusion:.4f} "
      f"(delta {best_fusion - base:+.4f})")
print(f"C selected by inner CV: {pd.Series(chosen_C).value_counts().to_dict()}")

Path("results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("results/fusion.csv", index=False)
np.save("results/oof_fusion.npy", np.stack([results[k] for k in results]))
print("saved -> results/fusion.csv")
