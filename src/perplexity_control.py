"""Is the surprisal difference independent of transcript length?

Mean surprisal correlates with length at r=-0.50 (more left context makes tokens
easier to predict), and impaired transcripts are ~10% shorter. So part of the
raw surprisal gap could be a length artifact. This regresses surprisal on length
and re-tests the group difference on the residual.

Also asks whether surprisal adds anything on top of the Phi-3 + TF-IDF fusion.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

N_FOLDS, N_REPEATS = 5, 5
PPL = ["nll_mean", "nll_median", "nll_std", "nll_p90", "frac_surprising"]

man = pd.read_csv("artifacts/manifest.csv")
ppl = pd.read_csv("artifacts/perplexity.csv")
df = man.merge(ppl, on="file_id")
keep = ((df.group == "Control") | (df.dx == "ProbableAD")).to_numpy()
X_emb = np.array(np.load("artifacts/L12_mean.npy")[
    ((man.group == "Control") | (man.dx == "ProbableAD")).to_numpy()], dtype=np.float64)
df = df[keep].reset_index(drop=True)
df["label"] = (df.dx == "ProbableAD").astype(int)
y, groups = df.label.to_numpy(), df.subject_id.to_numpy()
texts = df.text.fillna("").to_numpy()

print("=== surprisal difference, controlling for length ===")
Xd = sm.add_constant(pd.DataFrame({
    "label": y.astype(float),
    "log_tokens": np.log(df.n_tokens.to_numpy(dtype=float)),
}))
fit = sm.OLS(df.nll_mean.to_numpy(dtype=float), Xd).fit()
print(fit.summary2().tables[1].round(4).to_string())
print(f"\nlabel coefficient {fit.params['label']:+.4f} "
      f"(p={fit.pvalues['label']:.2e}) -- the group gap in mean surprisal that "
      f"survives length adjustment")

resid = fit.resid + fit.params["label"] * y      # keep group effect, remove length
t, p = ttest_ind(resid[y == 1], resid[y == 0])
print(f"length-adjusted residual: Control {resid[y==0].mean():+.4f}  "
      f"AD {resid[y==1].mean():+.4f}  (t={t:.2f}, p={p:.2e})")

print("\n=== does surprisal add to the Phi-3 + TF-IDF fusion? ===")
P = np.array(df[PPL].to_numpy(dtype=float), copy=True)


def z(a, ref):
    return (a - ref.mean()) / (ref.std() + 1e-9)


names = ["phi3+tfidf (2-way)", "phi3+tfidf+ppl (3-way)", "ppl alone"]
oof = {n: np.zeros((N_REPEATS, len(y))) for n in names}

for r in range(N_REPEATS):
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r)
    for tr, te in cv.split(np.zeros(len(y)), y, groups):
        ytr = y[tr]
        sc = StandardScaler().fit(X_emb[tr])
        m1 = LogisticRegression(max_iter=2000, C=0.01, class_weight="balanced").fit(
            sc.transform(X_emb[tr]), ytr)
        p1_tr, p1_te = (m1.decision_function(sc.transform(X_emb[i])) for i in (tr, te))

        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(texts[tr])
        m2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            vec.transform(texts[tr]), ytr)
        p2_tr, p2_te = (m2.decision_function(vec.transform(texts[i])) for i in (tr, te))

        scp = StandardScaler().fit(P[tr])
        m3 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            scp.transform(P[tr]), ytr)
        p3_tr, p3_te = (m3.decision_function(scp.transform(P[i])) for i in (tr, te))

        oof[names[0]][r, te] = (z(p1_te, p1_tr) + z(p2_te, p2_tr)) / 2
        oof[names[1]][r, te] = (z(p1_te, p1_tr) + z(p2_te, p2_tr) + z(p3_te, p3_tr)) / 3
        oof[names[2]][r, te] = z(p3_te, p3_tr)

rows = []
for n in names:
    a = [roc_auc_score(y, oof[n][r]) for r in range(N_REPEATS)]
    rows.append({"model": n, "auroc": round(float(np.mean(a)), 4),
                 "auroc_sd": round(float(np.std(a)), 4)})
    print(f"  {n:26s} AUROC {np.mean(a):.4f} +/- {np.std(a):.4f}")

d = rows[1]["auroc"] - rows[0]["auroc"]
print(f"\nadding surprisal to the fusion: {d:+.4f} AUROC")
pd.DataFrame(rows).to_csv("results/perplexity_fusion.csv", index=False)
print("saved -> results/perplexity_fusion.csv")
