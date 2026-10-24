"""Do Phi-3 representations track MMSE severity better than lexical features?

Classification AUROC says whether a model separates two groups. A biomarker
should also be graded. This compares out-of-fold scores from Phi-3 features and
from TF-IDF on the same subject-disjoint folds, correlated against MMSE --
overall, within the impaired group only (the hard test, since between-group
variance is removed), and partialling out age.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

LAYER, POOLING, N_REPEATS, N_FOLDS = 12, "mean", 5, 5

df = pd.read_csv("artifacts/manifest.csv")
blob = torch.load("artifacts/activations.pt", weights_only=False)
order = {str(f): i for i, f in enumerate(blob["file_ids"])}
df["_row"] = df.file_id.astype(str).map(order)
df = df[(df.group == "Control") | (df.dx == "ProbableAD")].reset_index(drop=True)
df["label"] = (df.dx == "ProbableAD").astype(int)

X = np.array(blob[POOLING][df["_row"].to_numpy()][:, LAYER, :].numpy(), dtype=np.float64)
texts = df.text.fillna("").to_numpy()
y, groups = df.label.to_numpy(), df.subject_id.to_numpy()


def oof_scores(kind):
    acc = np.zeros((N_REPEATS, len(y)))
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r)
        for tr, te in cv.split(np.zeros(len(y)), y, groups):
            if kind == "phi3":
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000, C=0.01,
                                         class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
                acc[r, te] = clf.decision_function(sc.transform(X[te]))
            else:
                pipe = make_pipeline(
                    TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True),
                    LogisticRegression(max_iter=3000, class_weight="balanced"))
                pipe.fit(texts[tr], y[tr])
                acc[r, te] = pipe.decision_function(texts[te])
    return acc


def partial_r(a, b, ctrl):
    A = np.c_[np.ones(len(ctrl)), ctrl]
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    return pearsonr(ra, rb)


results = {}
for kind in ("phi3", "tfidf"):
    oof = oof_scores(kind)
    auc = np.mean([roc_auc_score(y, oof[r]) for r in range(N_REPEATS)])
    s = oof.mean(0)
    results[kind] = (auc, s)

have = df.mmse.notna().to_numpy()
age_ok = have & df.age.notna().to_numpy()

label = {"phi3": f"Phi-3 L{LAYER} {POOLING}", "tfidf": "TF-IDF uni+bigram"}
print(f"Control vs ProbableAD, subject-disjoint {N_FOLDS}x{N_REPEATS}")
print(f"MMSE available: {have.sum()}/{len(df)} files\n")
print(f"{'model':22s} {'AUROC':>7s} {'r(MMSE) all':>13s} {'r impaired':>12s} "
      f"{'rho impaired':>13s} {'r|age':>8s}")
for kind, (auc, s) in results.items():
    m_imp = have & (y == 1)
    r_all = pearsonr(s[have], df.mmse.to_numpy()[have])[0]
    r_imp, p_imp = pearsonr(s[m_imp], df.mmse.to_numpy()[m_imp])
    rho_imp = spearmanr(s[m_imp], df.mmse.to_numpy()[m_imp])[0]
    r_age = partial_r(s[age_ok], df.mmse.to_numpy()[age_ok],
                      df.age.to_numpy(dtype=float)[age_ok])[0]
    print(f"{label[kind]:22s} {auc:7.3f} {r_all:13.3f} {r_imp:12.3f} "
          f"{rho_imp:13.3f} {r_age:8.3f}   (p_imp={p_imp:.1e}, n_imp={m_imp.sum()})")

# do they agree, or carry different information?
print(f"\ncorrelation between the two models' scores: "
      f"r={pearsonr(results['phi3'][1], results['tfidf'][1])[0]:.3f}")
