"""
Non-neural reference baselines, all under the same subject-disjoint protocol as
the main evaluation.

The question these answer: how much of the Control/ProbableAD separation is
available WITHOUT the LLM representation? If TF-IDF over word counts, or the
patient's age, matches the Phi-3 probe, the representation is not doing the work.

Every vectorizer, scaler and threshold is fit inside the training fold. Fitting
TF-IDF on the full corpus would leak test vocabulary and IDF statistics, which
is the same class of error as the file-level split.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from disfluency import ANNOT_MARKERS, ASR_MARKERS  # noqa: E402

N_REPEATS, N_FOLDS = 5, 5


def cv_auroc(build_fit_predict, y, groups, n_repeats=N_REPEATS):
    """Repeated subject-disjoint CV. Returns (mean, sd) AUROC over repeats."""
    aurocs = []
    for r in range(n_repeats):
        oof = np.zeros(len(y), dtype=float)
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r)
        for tr, te in cv.split(np.zeros(len(y)), y, groups):
            oof[te] = build_fit_predict(tr, te, y[tr])
        aurocs.append(roc_auc_score(y, oof))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-dx", default="ProbableAD")
    args = ap.parse_args()

    df = pd.read_csv("artifacts/manifest.csv").merge(
        pd.read_csv("artifacts/disfluency.csv"), on="file_id"
    )
    df = df[(df.group == "Control") | (df.dx == args.cohort_dx)].reset_index(drop=True)
    df["label"] = (df.dx == args.cohort_dx).astype(int)
    df["sex_m"] = (df.sex == "male").astype(float)

    y = df.label.to_numpy()
    groups = df.subject_id.to_numpy()
    texts = df.text.fillna("").to_numpy()

    print(f"Control vs {args.cohort_dx}: {len(df)} files, "
          f"{df.subject_id.nunique()} subjects, "
          f"{int(y.sum())} impaired / {int((1 - y).sum())} control")
    print(f"Subject-disjoint {N_FOLDS}-fold x {N_REPEATS} repeats\n")

    # ---- dense tabular features -------------------------------------------
    asr_cols = [f"{k}_per100" for k in ASR_MARKERS]
    annot_cols = [f"{k}_per100" for k in ANNOT_MARKERS]

    def dense(cols):
        X = np.array(df[cols].to_numpy(dtype=float), copy=True)
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            X[m, j] = np.nanmedian(X[:, j]) if (~m).any() else 0.0

        def run(tr, te, ytr):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                sc.transform(X[tr]), ytr
            )
            return clf.decision_function(sc.transform(X[te]))

        return run

    # ---- text features -----------------------------------------------------
    def textual(vec_factory, C=1.0):
        def run(tr, te, ytr):
            pipe = make_pipeline(
                vec_factory(),
                LogisticRegression(max_iter=3000, C=C, class_weight="balanced"),
            )
            pipe.fit(texts[tr], ytr)          # vectorizer fit on TRAIN ONLY
            return pipe.decision_function(texts[te])

        return run

    baselines = [
        ("--- demographic / surface ---", None),
        ("age only", dense(["age"])),
        ("transcript length only", dense(["n_words"])),
        ("age + length + sex", dense(["age", "n_words", "sex_m"])),
        ("--- disfluency ---", None),
        ("disfluency (ASR-recoverable)", dense(asr_cols)),
        ("disfluency (all, incl. annotator)", dense(asr_cols + annot_cols)),
        ("disfluency + age + length", dense(asr_cols + annot_cols + ["age", "n_words"])),
        ("--- lexical (bag of words) ---", None),
        ("bag-of-words counts", textual(
            lambda: CountVectorizer(min_df=2, max_features=5000), C=0.1)),
        ("TF-IDF unigram", textual(
            lambda: TfidfVectorizer(min_df=2, sublinear_tf=True), C=1.0)),
        ("TF-IDF uni+bigram", textual(
            lambda: TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True),
            C=1.0)),
        ("TF-IDF char 3-5gram", textual(
            lambda: TfidfVectorizer(min_df=2, analyzer="char_wb", ngram_range=(3, 5),
                                    sublinear_tf=True), C=1.0)),
    ]

    rows = []
    for name, fn in baselines:
        if fn is None:
            print(f"\n{name}")
            continue
        mu, sd = cv_auroc(fn, y, groups)
        rows.append({"baseline": name, "auroc": round(mu, 4), "auroc_sd": round(sd, 4)})
        print(f"  {name:36s} AUROC {mu:.3f} +/- {sd:.3f}", flush=True)

    Path("results").mkdir(exist_ok=True)
    out = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    out.to_csv("results/baselines.csv", index=False)
    print(f"\nstrongest non-neural baseline: {out.iloc[0].baseline} "
          f"(AUROC {out.iloc[0].auroc:.3f})")
    print("saved -> results/baselines.csv")


if __name__ == "__main__":
    raise SystemExit(main())
