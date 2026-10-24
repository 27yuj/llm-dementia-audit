"""Evaluate transcript surprisal as a standalone marker, subject-disjoint."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

N_FOLDS, N_REPEATS = 5, 5

df = (pd.read_csv("artifacts/manifest.csv")
      .merge(pd.read_csv("artifacts/perplexity.csv"), on="file_id",
             suffixes=("", "_ppl")))
df = df[(df.group == "Control") | (df.dx == "ProbableAD")].reset_index(drop=True)
df["label"] = (df.dx == "ProbableAD").astype(int)
y, groups = df.label.to_numpy(), df.subject_id.to_numpy()

PPL = ["nll_mean", "nll_median", "nll_std", "nll_p90", "frac_surprising"]

print(f"Control vs ProbableAD: {len(df)} files, {df.subject_id.nunique()} subjects\n")
print("=== group means ===")
stats = df.groupby("label")[PPL + ["perplexity", "n_tokens"]].mean().T
stats.columns = ["Control", "ProbableAD"]
stats["delta_%"] = (100 * (stats.ProbableAD / stats.Control - 1)).round(1)
for c in PPL + ["perplexity"]:
    t, p = ttest_ind(df[df.label == 1][c], df[df.label == 0][c])
    stats.loc[c, "p"] = p
print(stats.round(4).to_string())


def cv_auroc(cols):
    X = np.array(df[cols].to_numpy(dtype=float), copy=True)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = np.nanmedian(X[:, j]) if (~m).any() else 0.0
    aucs = []
    for r in range(N_REPEATS):
        oof = np.zeros(len(y))
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r)
        for tr, te in cv.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                sc.transform(X[tr]), y[tr])
            oof[te] = clf.decision_function(sc.transform(X[te]))
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


print("\n=== subject-disjoint AUROC ===")
combos = [
    ("mean surprisal alone", ["nll_mean"]),
    ("all surprisal features", PPL),
    ("surprisal + length", PPL + ["n_tokens"]),
    ("length alone (reference)", ["n_tokens"]),
    ("surprisal + length + age", PPL + ["n_tokens", "age"]),
]
rows = []
for name, cols in combos:
    mu, sd = cv_auroc(cols)
    rows.append({"features": name, "auroc": round(mu, 4), "auroc_sd": round(sd, 4)})
    print(f"  {name:28s} AUROC {mu:.3f} +/- {sd:.3f}")

print("\n=== correlation with MMSE ===")
have = df.dropna(subset=["mmse"])
for c in ["nll_mean", "nll_std", "frac_surprising"]:
    r_all, p_all = pearsonr(have[c], have.mmse)
    imp = have[have.label == 1]
    r_imp, p_imp = pearsonr(imp[c], imp.mmse)
    print(f"  {c:16s} all r={r_all:+.3f} (p={p_all:.1e})   "
          f"impaired-only r={r_imp:+.3f} (p={p_imp:.1e})")

print("\n=== is surprisal just measuring length? ===")
r, p = pearsonr(df.nll_mean, df.n_tokens)
print(f"  corr(mean surprisal, n_tokens) = {r:+.3f} (p={p:.1e})")

pd.DataFrame(rows).to_csv("results/perplexity.csv", index=False)
print("\nsaved -> results/perplexity.csv")
