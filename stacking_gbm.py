"""
Does a NON-LINEAR meta-learner beat plain score averaging?

Equal-weight averaging of Phi-3 and TF-IDF reaches 0.9246; a learned LINEAR
combiner reached 0.9222, i.e. slightly worse (src/stacked_fusion.py). The
obvious next hypothesis is that the combination is non-linear -- that the
lexical model should be trusted more when the transcript is long, or the
embedding model more when disfluency is high -- and that a gradient-boosted
tree could exploit that.

This script tests it under the same subject-disjoint protocol, with two things
kept scrupulously honest, because the paper's whole claim is that sloppy
evaluation manufactures gains:

  1. The meta-learner's TRAINING matrix is out-of-fold. An INNER
     StratifiedGroupKFold on the training portion (grouped on subject) produces
     base-model scores from models that never saw the row they are scoring.
     Fitting the combiner on in-sample base scores is the classic stacking bug:
     the strongest base model looks near-perfect on its own training rows, the
     combiner hands it all the weight, and the ensemble degenerates.
  2. Every meta-learner hyperparameter is chosen by a further inner CV on that
     same training matrix (again grouped on subject). Nothing is picked by
     looking at outer-fold performance.

The prior expectation is that boosting LOSES here, and the design makes that a
meaningful result rather than an artefact: 3-4 input features and ~380 training
rows is a regime where a tree ensemble has plenty of capacity to memorise the
inner-fold noise and nothing extra to learn. The GBMs are therefore held to
num_leaves<=4, max_depth<=3, n_estimators<=200, learning_rate<=0.05 and a large
min_child_samples. If a heavily constrained GBM cannot beat the mean of two
numbers, an unconstrained one certainly is not doing so honestly.

TWO comparators, not one. A stack that is handed four bases must not be credited
with a gain that comes from feature COUNT rather than from non-linearity -- the
plain unweighted mean of all four bases already beats the mean of two. So every
learned stack is tested against BOTH averages, and the delta that matters is the
one against the average over the SAME bases the stack was given.

Scale note: each base model's scores are z-scored by the TRAINING score
distribution of the same model, exactly as in src/fusion.py. That puts the
inner-CV meta features and the outer-fold meta features on a comparable scale
without any test-fold statistic entering.

Thresholds are out-of-sample in EVERY arm. The averages take Youden's J on the
inner-CV base scores; the learned stacks take it on the meta-learner's own
out-of-fold scores from the hyperparameter-selection CV -- never on the matrix
the final meta-learner was just fitted to. In-sample meta scores would flatter
every stack's accuracy by one to two points (and its sensitivity by more) while
the averages got an honest threshold, i.e. a bias pointing in exactly the
direction of the hypothesis under test.

Two limitations, stated rather than left implicit:

  - Base-model hyperparameters (Phi-3 C=0.01; TF-IDF C=1.0, min_df=2, uni+bigram)
    are frozen constants inherited verbatim from src/fusion.py, where they were
    chosen on the full dataset. They are NOT re-selected inside each fold, which
    is a pipeline-level researcher-degrees-of-freedom leak and sets the absolute
    level of every number below. It cannot manufacture a STACKING gain: the
    references and the meta-learners consume identical base scores. They are kept
    fixed so these numbers stay comparable with the established 0.9246 reference.
  - Hyperparameter selection re-partitions rows that a different inner partition
    generated, so its folds are correlated through shared base-model parameters.
    This inflates every candidate config's apparent quality by roughly the same
    amount and imports no outer-test information; removing it entirely would
    require a third level of nesting.

Multiplicity is Holm-corrected across the whole comparison table, because the
paper's own argument is that an uncorrected family of five-plus comparisons
manufactures "significant" findings at a ~23% family-wise rate.

Outputs: results/stacking_gbm.csv, results/stacking_significance.csv,
         results/oof_stacking.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from disfluency import ASR_MARKERS  # noqa: E402
from evaluate import metrics_at, subject_bootstrap_ci, youden_threshold  # noqa: E402

LAYER, POOLING = 12, "mean"
REFERENCE = "average (phi3+tfidf)"      # the 0.9246 number every stack must beat
AVG_ALL = "average (all bases)"         # guards against a feature-COUNT gain

# Heavily constrained on purpose -- see module docstring.
GRIDS = {
    "logreg": [{"C": c} for c in (0.03, 0.1, 0.3, 1.0, 3.0)],
    "lightgbm": [
        {"num_leaves": 2, "max_depth": 2, "n_estimators": 200,
         "learning_rate": 0.05, "min_child_samples": 20},
        {"num_leaves": 2, "max_depth": 2, "n_estimators": 100,
         "learning_rate": 0.05, "min_child_samples": 40},
        {"num_leaves": 3, "max_depth": 3, "n_estimators": 200,
         "learning_rate": 0.05, "min_child_samples": 20},
        {"num_leaves": 3, "max_depth": 3, "n_estimators": 200,
         "learning_rate": 0.02, "min_child_samples": 40},
        {"num_leaves": 4, "max_depth": 3, "n_estimators": 200,
         "learning_rate": 0.05, "min_child_samples": 40},
        {"num_leaves": 4, "max_depth": 3, "n_estimators": 100,
         "learning_rate": 0.02, "min_child_samples": 20},
    ],
    "xgboost": [
        {"max_depth": 1, "n_estimators": 200, "learning_rate": 0.05,
         "min_child_weight": 10},
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05,
         "min_child_weight": 20},
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.05,
         "min_child_weight": 10},
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.02,
         "min_child_weight": 40},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.02,
         "min_child_weight": 20},
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.05,
         "min_child_weight": 20},
    ],
    "randomforest": [
        {"n_estimators": 300, "max_depth": 2, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 20},
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 20},
    ],
}
LABEL = {
    REFERENCE: "simple average (reference)",
    AVG_ALL: "simple average, all bases",
    "logreg": "logistic meta-learner",
    "lightgbm": "LightGBM meta-learner",
    "xgboost": "XGBoost meta-learner",
    "randomforest": "random forest meta-learner",
}
SHORT = {REFERENCE: "avg2", AVG_ALL: "avgN", "logreg": "logreg",
         "lightgbm": "lgbm", "xgboost": "xgb", "randomforest": "rf"}
# feature_importances_ means something different for every library; the
# attribution table below mixes these three incommensurable scales by row.
IMPORTANCE_KIND = {
    "logreg": "|coefficient|",
    "lightgbm": "split count",
    "xgboost": "gain",
    "randomforest": "mean impurity decrease",
}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_cohort(manifest, embeddings, disfluency, order=None):
    """The primary cohort: Control vs ProbableAD, 477 files / 246 subjects.

    The embedding matrix is joined POSITIONALLY, so a re-sorted manifest would
    silently pair every transcript with someone else's vector. A length check
    catches truncation and merge fan-out but not a permutation, so the file_id
    order file written alongside the embeddings is verified element-wise.

    The label-purity assertion pins down a latent hazard in build_manifest.py:
    subject_id is parsed from the filename stem with no corpus-directory
    namespace, so id 172 spans a Control file and three MCI files. Harmless for
    this cohort (only the Control visit survives the filter), but widening the
    cohort to MCI would put one GROUP id across two label classes and quietly
    corrupt every subject-disjoint split.
    """
    man = pd.read_csv(manifest)
    if order is not None and Path(order).exists():
        ids = pd.read_csv(order).file_id.astype(str).to_numpy()
        assert len(ids) == len(man) and (ids == man.file_id.astype(str).to_numpy()).all(), \
            f"{order} disagrees with {manifest}: embeddings are positionally misaligned"

    keep = ((man.group == "Control") | (man.dx == "ProbableAD")).to_numpy()
    X_emb = np.array(np.load(embeddings)[keep], dtype=np.float64)

    df = man[keep].reset_index(drop=True)
    df = df.merge(pd.read_csv(disfluency), on="file_id", how="left")
    df["label"] = (df.dx == "ProbableAD").astype(int)
    assert len(df) == len(X_emb), "manifest/embedding row order diverged"
    assert df.groupby("subject_id").label.nunique().max() == 1, \
        "a subject_id spans both classes -- grouped splits would be meaningless"
    return df, X_emb


def z(a, ref):
    return (a - ref.mean()) / (ref.std() + 1e-9)


def make_base_fns(df, X_emb, names):
    """Base models as fn(tr, te, ytr) -> (train scores, test scores).

    Each closure fits EVERYTHING it needs -- scaler, TF-IDF vocabulary and IDF,
    median imputation, coefficients -- on `tr` alone.
    """
    texts = df.text.fillna("").to_numpy()
    demo = df[["age", "n_words"]].to_numpy(dtype=float)
    disfl_cols = [f"{k}_per100" for k in ASR_MARKERS]
    disfl = df[disfl_cols].to_numpy(dtype=float)

    def dense(M, C=1.0):
        def run(tr, te, ytr):
            # per-fold median imputation; nothing from `te` informs the median
            med = np.array([np.median(c[~np.isnan(c)]) if (~np.isnan(c)).any()
                            else 0.0 for c in M[tr].T])
            Atr = np.where(np.isnan(M[tr]), med, M[tr])
            Ate = np.where(np.isnan(M[te]), med, M[te])
            sc = StandardScaler().fit(Atr)
            clf = LogisticRegression(max_iter=2000, C=C,
                                     class_weight="balanced").fit(sc.transform(Atr), ytr)
            return (clf.decision_function(sc.transform(Atr)),
                    clf.decision_function(sc.transform(Ate)))
        return run

    def phi3(tr, te, ytr):
        sc = StandardScaler().fit(X_emb[tr])
        clf = LogisticRegression(max_iter=2000, C=0.01,
                                 class_weight="balanced").fit(sc.transform(X_emb[tr]), ytr)
        return (clf.decision_function(sc.transform(X_emb[tr])),
                clf.decision_function(sc.transform(X_emb[te])))

    def tfidf(tr, te, ytr):
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2),
                              sublinear_tf=True).fit(texts[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0,
                                 class_weight="balanced").fit(vec.transform(texts[tr]), ytr)
        return (clf.decision_function(vec.transform(texts[tr])),
                clf.decision_function(vec.transform(texts[te])))

    available = {"phi3": phi3, "tfidf": tfidf,
                 "demo": dense(demo), "disfl": dense(disfl)}
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"unknown base model(s): {sorted(unknown)}")
    return [(n, available[n]) for n in names]


def base_matrix(base_fns, tr, te, ytr):
    """Fit all bases on `tr`; return z-scored meta features for tr and te.

    Each column is standardised by the model's own TRAINING score distribution
    (as in src/fusion.py), so inner-CV columns and outer-fold columns share a
    scale. No test-fold statistic is used anywhere.
    """
    cols_tr, cols_te = [], []
    for _, fn in base_fns:
        s_tr, s_te = fn(tr, te, ytr)
        cols_tr.append(z(s_tr, s_tr))
        cols_te.append(z(s_te, s_tr))
    return np.column_stack(cols_tr), np.column_stack(cols_te)


# ----------------------------------------------------------------------------
# meta-learners
# ----------------------------------------------------------------------------
def make_meta(name, params, ytr, seed=0):
    pos = max(int((ytr == 1).sum()), 1)
    neg = max(int((ytr == 0).sum()), 1)
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", **params))
    if name == "lightgbm":
        return LGBMClassifier(objective="binary", class_weight="balanced",
                              subsample=1.0, colsample_bytree=1.0,
                              min_split_gain=0.0, reg_lambda=1.0,
                              n_jobs=1, verbose=-1, force_col_wise=True,
                              random_state=seed, **params)
    if name == "xgboost":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             scale_pos_weight=neg / pos, tree_method="hist",
                             subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
                             n_jobs=1, verbosity=0, random_state=seed, **params)
    if name == "randomforest":
        return RandomForestClassifier(class_weight="balanced", n_jobs=1,
                                      random_state=seed, **params)
    raise ValueError(name)


def meta_score(est, Z):
    """Rank-comparable score; sign convention is 'higher == more impaired'."""
    if hasattr(est, "decision_function"):
        return est.decision_function(Z)
    return est.predict_proba(Z)[:, 1]


def meta_attribution(est):
    """Per-base share of the meta-learner's attention, normalised to sum to 1.

    The underlying quantity is NOT the same across rows: |coefficient| for the
    logistic combiner, split COUNT for LightGBM, GAIN for XGBoost (its sklearn
    default), and mean impurity decrease for the random forest. Normalising each
    row to sum to 1 makes the columns readable WITHIN a row; it does not make the
    rows commensurable with each other. See IMPORTANCE_KIND, which is printed
    beside every row for exactly this reason.

    Diagnostic only -- it never feeds back into any fit. Its job is to show
    whether a stack that fails to beat averaging is failing because it ignores
    the weak bases or because it over-trusts them.
    """
    w = est[-1].coef_[0] if hasattr(est, "steps") else getattr(
        est, "feature_importances_", None)
    if w is None:
        return None
    w = np.abs(np.asarray(w, dtype=float))
    return w / max(w.sum(), 1e-12)


def select_params(name, Zmeta, ytr, gtr, n_folds, seed):
    """Pick hyperparameters by CV on the TRAINING meta matrix, grouped on subject.

    Returns (best params, the winning config's out-of-fold score vector).

    The rows here are already out-of-fold base scores drawn only from the
    training portion, so this is model selection on training data -- the outer
    test fold is untouched. Re-splitting rows that were generated by a different
    inner partition slightly under-states the difficulty of the selection
    problem, but it cannot import any test information.

    The returned OOF vector is what the caller takes Youden's J from. It costs
    nothing extra (it is computed anyway to rank the configs) and it is the only
    honest training-side score available for a meta-learner: `Zmeta` itself is
    out-of-fold with respect to the BASE models but fully in-sample once the
    meta-learner has been fitted to it.
    """
    grid = GRIDS[name]
    cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(ytr)), ytr, gtr))
    best, best_auc, best_oof = grid[0], -np.inf, None
    for params in grid:
        oof = np.zeros(len(ytr))
        for itr, ite in splits:
            est = make_meta(name, params, ytr[itr], seed).fit(Zmeta[itr], ytr[itr])
            oof[ite] = meta_score(est, Zmeta[ite])
        auc = roc_auc_score(ytr, oof) if len(set(ytr)) > 1 else 0.5
        if auc > best_auc:
            best_auc, best, best_oof = auc, params, oof
    return best, best_oof


# ----------------------------------------------------------------------------
# main experiment
# ----------------------------------------------------------------------------
def run(df, X_emb, args):
    base_fns = make_base_fns(df, X_emb, args.bases.split(","))
    names = [n for n, _ in base_fns]
    y = df.label.to_numpy()
    groups = df.subject_id.to_numpy()

    try:
        i_phi3, i_tfidf = names.index("phi3"), names.index("tfidf")
    except ValueError:
        raise SystemExit("--bases must include both 'phi3' and 'tfidf' "
                         "(the reference average is defined over those two)")

    metas = [m for m in args.metas.split(",") if m]
    unknown = set(metas) - set(GRIDS)
    if unknown:
        raise SystemExit(f"unknown meta-learner(s): {sorted(unknown)}")
    # avgN is a distinct comparator only when a stack sees more than phi3+tfidf;
    # with exactly those two it IS avg2 and would just duplicate the row.
    references = [REFERENCE] + ([AVG_ALL] if len(names) > 2 else [])
    methods = references + metas
    oof = {m: np.zeros((args.repeats, len(y))) for m in methods}
    thr_acc = {m: [[] for _ in range(args.repeats)] for m in methods}
    chosen = {m: [] for m in metas}
    attrib = {m: [] for m in metas}

    for r in range(args.repeats):
        outer = StratifiedGroupKFold(args.folds, shuffle=True, random_state=r)
        for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
            ytr, gtr = y[tr], groups[tr]

            # ---- honest, out-of-fold training matrix for the meta-learner ----
            Zmeta = np.zeros((len(tr), len(names)))
            inner = StratifiedGroupKFold(args.inner_folds, shuffle=True,
                                         random_state=200 + r)
            for itr, ite in inner.split(np.zeros(len(tr)), ytr, gtr):
                _, Zmeta[ite] = base_matrix(base_fns, tr[itr], tr[ite], ytr[itr])

            # ---- bases refit on the whole training fold, applied to the test --
            _, Zte = base_matrix(base_fns, tr, te, ytr)

            for m in methods:
                if m == REFERENCE:
                    s_tr = Zmeta[:, [i_phi3, i_tfidf]].mean(1)
                    oof[m][r, te] = Zte[:, [i_phi3, i_tfidf]].mean(1)
                elif m == AVG_ALL:
                    s_tr = Zmeta.mean(1)
                    oof[m][r, te] = Zte.mean(1)
                else:
                    params, s_tr = select_params(m, Zmeta, ytr, gtr,
                                                 args.meta_folds, 300 + r)
                    chosen[m].append(str(params))
                    est = make_meta(m, params, ytr, seed=r).fit(Zmeta, ytr)
                    imp = meta_attribution(est)
                    if imp is not None:
                        attrib[m].append(imp)
                    oof[m][r, te] = meta_score(est, Zte)
                # Youden's J on training-fold scores that are OUT-OF-SAMPLE for
                # whatever produced them: inner-CV base scores for the averages,
                # the selection CV's own held-out meta scores for the stacks.
                # Never meta_score(est, Zmeta) -- est was just fitted to Zmeta.
                thr_acc[m][r].append(youden_threshold(ytr, s_tr))

            print(f"  repeat {r} fold {k} done", flush=True)
        print(f"repeat {r} complete   "
              + "  ".join(f"{SHORT.get(m, m)}={roc_auc_score(y, oof[m][r]):.4f}"
                          for m in methods), flush=True)

    thrs = {m: np.array([float(np.mean(thr_acc[m][r])) for r in range(args.repeats)])
            for m in methods}
    return methods, references, names, oof, thrs, chosen, attrib


def summarize(df, methods, oof, thrs, n_boot):
    y = df.label.to_numpy()
    rows = []
    for m in methods:
        per = [metrics_at(y, oof[m][r], thrs[m][r]) for r in range(oof[m].shape[0])]
        ci = subject_bootstrap_ci(df, oof[m], thrs[m], n_boot=n_boot)
        row = {"meta_learner": LABEL.get(m, m)}
        for key in per[0]:
            row[key] = round(float(np.mean([p[key] for p in per])), 4)
            row[f"{key}_sd"] = round(float(np.std([p[key] for p in per])), 4)
        row["auroc_ci_lo"] = round(ci["auroc_ci"][0], 4)
        row["auroc_ci_hi"] = round(ci["auroc_ci"][1], 4)
        rows.append(row)
        print(f"  {LABEL.get(m, m):28s} AUROC {row['auroc']:.4f} +/- {row['auroc_sd']:.4f} "
              f"[{row['auroc_ci_lo']:.3f}, {row['auroc_ci_hi']:.3f}]   "
              f"acc {row['acc']:.3f}  sens {row['sens']:.3f}  spec {row['spec']:.3f}",
              flush=True)
    return rows


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values: monotone, capped at 1."""
    p = np.asarray(pvals, dtype=float)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired_bootstrap(df, methods, references, oof, n_boot, seed=0):
    """Paired subject-level bootstrap against BOTH averaging baselines.

    Resamples SUBJECTS (the unit of independence) and scores every meta-learner
    on the IDENTICAL resample, so the shared difficulty of a draw cancels.
    Mirrors src/fusion_test.py.

    Every learned stack is compared twice: against the two-base average (the
    established 0.9246 reference) and against the plain mean of the same bases it
    was handed. Without that second comparison, a stack fed four bases can post a
    pure feature-COUNT gain and have it read as evidence for non-linearity --
    the unweighted four-way mean already beats the two-way one.

    p_boot is the bootstrap achieved significance level (twice the smaller tail
    mass of the paired difference), floored at 1/n_draws. Holm-Bonferroni is
    applied across the whole family, because at an uncorrected 95% level five
    comparisons carry a ~23% chance of one spurious finding under the global
    null -- precisely the mechanism this paper criticises.
    """
    y = df.label.to_numpy()
    subs = df.subject_id.unique()
    by_sub = {s: np.flatnonzero(df.subject_id.to_numpy() == s) for s in subs}
    n_repeats = oof[methods[0]].shape[0]
    rng = np.random.default_rng(seed)

    draws = {m: [] for m in methods}
    for b in range(n_boot):
        pick = rng.choice(len(subs), len(subs), replace=True)
        idx = np.concatenate([by_sub[subs[p]] for p in pick])
        if len(set(y[idx])) < 2:
            continue
        r = b % n_repeats
        for m in methods:
            draws[m].append(roc_auc_score(y[idx], oof[m][r][idx]))
    draws = {m: np.asarray(v) for m, v in draws.items()}

    print("\nsubject bootstrap, 95% CI:")
    for m in methods:
        lo, hi = np.percentile(draws[m], [2.5, 97.5])
        print(f"  {LABEL.get(m, m):28s} {draws[m].mean():.4f}  [{lo:.4f}, {hi:.4f}]")

    # family 1: everything against the two-base reference.
    # family 2: every LEARNED stack against the mean of the bases it was given.
    pairs = [(m, REFERENCE) for m in methods if m != REFERENCE]
    if AVG_ALL in references:
        pairs += [(m, AVG_ALL) for m in methods if m not in references]

    rows = []
    for m, ref in pairs:
        d = draws[m] - draws[ref]
        lo, hi = np.percentile(d, [2.5, 97.5])
        gt, lt = float((d > 0).mean()), float((d < 0).mean())
        rows.append({"comparison": f"{LABEL.get(m, m)} - {LABEL.get(ref, ref)}",
                     "method": LABEL.get(m, m), "baseline": LABEL.get(ref, ref),
                     "delta": round(float(d.mean()), 4),
                     "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                     "frac_wins": round(gt, 3),
                     "p_boot": min(1.0, max(2 * min(gt, lt), 1.0 / max(len(d), 1))),
                     "ci_excludes_zero": "yes" if (lo > 0 or hi < 0) else "no"})
    for row, padj in zip(rows, holm([r0["p_boot"] for r0 in rows])):
        row["p_boot"] = round(row["p_boot"], 4)
        row["p_holm"] = round(float(padj), 4)
        row["sig_holm_05"] = "yes" if padj < 0.05 else "no"

    print(f"\npaired differences on the same resampled subjects "
          f"({len(rows)} comparisons, Holm-corrected):")
    for row in rows:
        print(f"  {row['comparison']:58s} delta {row['delta']:+.4f}  "
              f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  "
              f"wins {row['frac_wins'] * 100:5.1f}%  "
              f"p={row['p_boot']:.3f}  holm={row['p_holm']:.3f}  "
              f"significant: {row['sig_holm_05']}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--manifest", default="artifacts/manifest.csv")
    ap.add_argument("--embeddings", default="artifacts/L12_mean.npy")
    ap.add_argument("--embedding-order", default="artifacts/L12_mean_order.csv",
                    help="file_id order of --embeddings; verified element-wise "
                         "against the manifest (the join is positional)")
    ap.add_argument("--disfluency", default="artifacts/disfluency.csv")
    ap.add_argument("--bases", default="phi3,tfidf,demo,disfl",
                    help="comma list from phi3,tfidf,demo,disfl "
                         "(phi3 and tfidf are required)")
    ap.add_argument("--metas", default="logreg,lightgbm,xgboost,randomforest")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=5,
                    help="inner CV that builds the meta-learner training matrix")
    ap.add_argument("--meta-folds", type=int, default=3,
                    help="inner CV that selects meta-learner hyperparameters")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    df, X_emb = load_cohort(args.manifest, args.embeddings, args.disfluency,
                            order=args.embedding_order)
    y = df.label.to_numpy()
    print(f"Control vs ProbableAD: {len(df)} files, {df.subject_id.nunique()} subjects, "
          f"{int(y.sum())} impaired / {int((1 - y).sum())} control")
    print(f"subject-disjoint {args.folds}-fold x {args.repeats} repeats, "
          f"Phi-3 L{LAYER} {POOLING}")
    print(f"bases: {args.bases}   meta-learners: {args.metas}\n")

    methods, references, base_names, oof, thrs, chosen, attrib = run(df, X_emb, args)

    print("\nstacking results:")
    rows = summarize(df, methods, oof, thrs, args.n_boot)
    sig = paired_bootstrap(df, methods, references, oof, args.n_boot)

    print("\nhyperparameters selected by inner CV (modal choice):")
    for m, picks in chosen.items():
        if picks:
            vc = pd.Series(picks).value_counts()
            print(f"  {LABEL.get(m, m):28s} {vc.index[0]}  "
                  f"({vc.iloc[0]}/{len(picks)} folds)")

    print(f"\nshare of meta-learner attention per base "
          f"(mean over folds; equal weight = {1 / len(base_names):.2f} each).")
    print("   Rows are NOT commensurable -- the importance type differs by library:")
    print(f"  {'':28s} " + "  ".join(f"{b:>7s}" for b in base_names))
    for m in attrib:
        if attrib[m]:
            w = np.mean(np.stack(attrib[m]), axis=0)
            print(f"  {LABEL.get(m, m):28s} " + "  ".join(f"{v:7.3f}" for v in w)
                  + f"   [{IMPORTANCE_KIND.get(m, '?')}]")

    # LEARNED stacks only: 'simple average, all bases' is a baseline, not a
    # meta-learner, and it is the front-runner often enough to be picked here.
    by_method = dict(zip(methods, rows))
    learned = [m for m in methods if m not in references]
    print()
    for m in references:
        print(f"{LABEL.get(m, m):28s} {by_method[m]['auroc']:.4f}")
    if learned:
        best = max(learned, key=lambda m: by_method[m]["auroc"])
        deltas = "   ".join(
            f"vs {SHORT.get(m, m)} {by_method[best]['auroc'] - by_method[m]['auroc']:+.4f}"
            for m in references)
        print(f"best learned meta-learner: {LABEL.get(best, best)} "
              f"{by_method[best]['auroc']:.4f}   {deltas}")
    else:
        print("no learned meta-learner requested (--metas empty): averages only.")

    any_sig = [s for s in sig if s["sig_holm_05"] == "yes"]
    if any_sig:
        print("differences surviving Holm correction: "
              + "; ".join(f"{s['comparison']} ({s['delta']:+.4f})" for s in any_sig))
    else:
        marginal = [s for s in sig if s["ci_excludes_zero"] == "yes"]
        tail = (f"{len(marginal)} would have passed an UNCORRECTED 95% CI: "
                + "; ".join(s["comparison"] for s in marginal)) if marginal else \
               "every paired CI contains zero even before correction."
        print(f"no comparison survives Holm correction across {len(sig)} "
              f"test{'s' if len(sig) != 1 else ''}; " + tail)

    pd.DataFrame(rows).to_csv("results/stacking_gbm.csv", index=False)
    pd.DataFrame(sig).to_csv("results/stacking_significance.csv", index=False)
    np.save("results/oof_stacking.npy", np.stack([oof[m] for m in methods]))
    Path("results/oof_stacking_methods.txt").write_text("\n".join(methods), encoding="utf-8")
    print("\nsaved -> results/stacking_gbm.csv, results/stacking_significance.csv, "
          "results/oof_stacking.npy")


if __name__ == "__main__":
    raise SystemExit(main())
