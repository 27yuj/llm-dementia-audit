"""
Subject-disjoint evaluation of Phi-3 representations for cognitive-impairment
detection on the DementiaBank Pitt Cookie Theft task.

Every split groups by subject_id, so no speaker ever appears in both train and
test. Every decision threshold, cPCA sign, scaler, and direction is fit on the
training fold alone.

Modes
  sweep : cPCA + logistic regression across all layers x {last, mean} pooling
  main  : full method comparison at one layer, including the trained probe
  mmse  : does the concept-vector projection track MMSE severity?
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from models import fit_cpca_scorer, probe_scores, train_probe  # noqa: E402

COHORTS = {
    "control_vs_probablead": (["Control"], ["ProbableAD"]),
    "control_vs_ad_spectrum": (["Control"], ["ProbableAD", "PossibleAD"]),
    "control_vs_all_dementia": (["Control"], None),   # None = whole Dementia folder
    "control_vs_mci": (["Control"], ["MCI"]),
}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_data(manifest, activations, cohort):
    df = pd.read_csv(manifest)
    blob = torch.load(activations, weights_only=False)

    order = {str(f): i for i, f in enumerate(blob["file_ids"])}
    df = df[df.file_id.astype(str).isin(order)].copy()
    df["_row"] = df.file_id.astype(str).map(order)

    ctrl_dx, case_dx = COHORTS[cohort]
    is_ctrl = df.group.eq("Control") & (df.dx.isin(ctrl_dx) | df.dx.isna())
    is_case = df.group.eq("Dementia") & (
        df.dx.isin(case_dx) if case_dx is not None else True
    )
    df = df[is_ctrl | is_case].copy()
    df["label"] = is_case[df.index].astype(int)

    feats = {p: blob[p][df["_row"].to_numpy()].numpy() for p in ("last", "mean")}
    return df.reset_index(drop=True), feats, blob["last"].shape[1]


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def metrics_at(y, scores, thr):
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "auroc": float(roc_auc_score(y, scores)) if len(set(y)) > 1 else float("nan"),
        "acc": (tp + tn) / max(len(y), 1),
        "bal_acc": (sens + spec) / 2,
        "sens": sens,
        "spec": spec,
        "f1": 2 * prec * sens / max(prec + sens, 1e-9),
    }


def subject_bootstrap_ci(df, oof, thrs, n_boot=2000, seed=0):
    """95% CI resampling SUBJECTS (not files) -- the unit of independence.

    Bootstraps WITHIN each repeat and pools the draws, so the interval describes
    the same estimator as the reported point estimate (mean per-repeat metric).
    Bootstrapping the repeat-averaged score instead would describe an ensemble of
    all repeats, which is a different -- and better-performing -- predictor.
    """
    rng = np.random.default_rng(seed)
    subs = df.subject_id.unique()
    by_sub = {s: np.flatnonzero(df.subject_id.to_numpy() == s) for s in subs}
    y = df.label.to_numpy()
    n_repeats = oof.shape[0]
    per_repeat_boot = max(n_boot // n_repeats, 1)

    aurocs, accs = [], []
    for r in range(n_repeats):
        for _ in range(per_repeat_boot):
            pick = rng.choice(len(subs), len(subs), replace=True)
            idx = np.concatenate([by_sub[subs[p]] for p in pick])
            if len(set(y[idx])) < 2:
                continue
            aurocs.append(roc_auc_score(y[idx], oof[r][idx]))
            accs.append(((oof[r][idx] >= thrs[r]).astype(int) == y[idx]).mean())

    def q(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    return {"auroc_ci": q(aurocs), "acc_ci": q(accs)}


def youden_threshold(y, s):
    order = np.argsort(s)
    cands = (s[order][:-1] + s[order][1:]) / 2
    pos, neg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    best, thr = -np.inf, float(np.median(s))
    for c in cands:
        p = s >= c
        j = (p & (y == 1)).sum() / pos - (p & (y == 0)).sum() / neg
        if j > best:
            best, thr = j, float(c)
    return thr


def _fit_predict(method, Xtr, ytr, Xte):
    """Fit `method` on the training rows and score the given rows."""
    if method == "cpca":
        fn, _, _ = fit_cpca_scorer(Xtr, ytr)
        return fn(Xte)
    if method == "logreg":
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=0.01,
                                 class_weight="balanced").fit(sc.transform(Xtr), ytr)
        return clf.decision_function(sc.transform(Xte))
    if method == "probe":
        m = train_probe(Xtr, ytr)
        return probe_scores(m, Xte)
    raise ValueError(method)



def run_cv(X, y, groups, method, n_splits=5, n_repeats=5, probe_kwargs=None):
    """Returns (oof_scores [n_repeats, N], oof_thresholds [n_repeats])."""
    probe_kwargs = probe_kwargs or {}
    oof = np.zeros((n_repeats, len(y)))
    thrs = np.zeros(n_repeats)

    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=r)
        train_thr = []
        for tr, te in cv.split(X, y, groups):
            Xtr, Xte, ytr = X[tr], X[te], y[tr]

            if method == "cpca":
                score_fn, thr, _ = fit_cpca_scorer(Xtr, ytr)
                oof[r, te] = score_fn(Xte)
                train_thr.append(thr)

            elif method == "logreg":
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(
                    max_iter=2000, C=0.01, class_weight="balanced"
                ).fit(sc.transform(Xtr), ytr)
                s_tr = clf.decision_function(sc.transform(Xtr))
                oof[r, te] = clf.decision_function(sc.transform(Xte))
                train_thr.append(youden_threshold(ytr, s_tr))

            elif method == "probe":
                m = train_probe(Xtr, ytr, seed=r, **probe_kwargs)
                s_tr = probe_scores(m, Xtr)
                oof[r, te] = probe_scores(m, Xte)
                train_thr.append(youden_threshold(ytr, s_tr))

            else:
                raise ValueError(method)

        thrs[r] = float(np.mean(train_thr))
    return oof, thrs


def summarize(df, oof, thrs, label):
    per_repeat = [metrics_at(df.label.to_numpy(), oof[r], thrs[r])
                  for r in range(len(thrs))]
    agg = {k: (np.mean([m[k] for m in per_repeat]),
               np.std([m[k] for m in per_repeat])) for k in per_repeat[0]}
    ci = subject_bootstrap_ci(df, oof, thrs)
    return {
        "method": label,
        **{k: round(v[0], 4) for k, v in agg.items()},
        **{f"{k}_sd": round(v[1], 4) for k, v in agg.items()},
        "auroc_ci_lo": round(ci["auroc_ci"][0], 4),
        "auroc_ci_hi": round(ci["auroc_ci"][1], 4),
        "acc_ci_lo": round(ci["acc_ci"][0], 4),
        "acc_ci_hi": round(ci["acc_ci"][1], 4),
    }


# ----------------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------------
def mode_sweep(df, feats, n_layers, args):
    """cPCA + logistic regression at every layer, both pooling schemes."""
    y, groups = df.label.to_numpy(), df.subject_id.to_numpy()
    rows = []
    for pooling in ("last", "mean"):
        for layer in range(n_layers):
            X = feats[pooling][:, layer, :].astype(np.float64)
            for method in ("cpca", "logreg"):
                oof, thrs = run_cv(X, y, groups, method,
                                   n_splits=args.folds, n_repeats=args.sweep_repeats)
                per_repeat = [metrics_at(y, oof[r], thrs[r]) for r in range(len(thrs))]
                rows.append({
                    "pooling": pooling, "layer": layer, "method": method,
                    "auroc": round(float(np.mean([m["auroc"] for m in per_repeat])), 4),
                    "auroc_sd": round(float(np.std([m["auroc"] for m in per_repeat])), 4),
                    "bal_acc": round(float(np.mean([m["bal_acc"] for m in per_repeat])), 4),
                })
            print(f"  {pooling} L{layer:02d}  "
                  f"cpca={rows[-2]['auroc']:.3f}  logreg={rows[-1]['auroc']:.3f}",
                  flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("results/layer_sweep.csv", index=False)
    print("\nBest configurations by AUROC:")
    print(out.sort_values("auroc", ascending=False).head(10).to_string(index=False))
    print("\nsaved -> results/layer_sweep.csv")


def mode_main(df, feats, n_layers, args):
    """Head-to-head comparison at one layer, subject-disjoint."""
    layer = args.layer if args.layer >= 0 else n_layers + args.layer
    X = feats[args.pooling][:, layer, :].astype(np.float64)
    y, groups = df.label.to_numpy(), df.subject_id.to_numpy()

    print(f"layer {layer} / pooling {args.pooling} / cohort {args.cohort}")
    print(f"{len(df)} files, {df.subject_id.nunique()} subjects, "
          f"{int(y.sum())} impaired / {int((1-y).sum())} control\n")

    rows = []
    for method in ("cpca", "logreg", "probe"):
        oof, thrs = run_cv(X, y, groups, method,
                           n_splits=args.folds, n_repeats=args.repeats,
                           probe_kwargs={"rank": args.rank, "epochs": args.epochs})
        rows.append(summarize(df, oof, thrs, method))
        np.save(f"results/oof_{method}_L{layer}_{args.pooling}.npy", oof)
        r = rows[-1]
        print(f"{method:8s} AUROC {r['auroc']:.3f} +/- {r['auroc_sd']:.3f} "
              f"[{r['auroc_ci_lo']:.3f}, {r['auroc_ci_hi']:.3f}]   "
              f"acc {r['acc']:.3f}  sens {r['sens']:.3f}  spec {r['spec']:.3f}",
              flush=True)

    pd.DataFrame(rows).to_csv(f"results/main_L{layer}_{args.pooling}.csv", index=False)
    print(f"\nsaved -> results/main_L{layer}_{args.pooling}.csv")


def mode_leakage(df, feats, n_layers, args):
    """The same probe, evaluated with file-level vs subject-disjoint splitting."""
    from sklearn.model_selection import StratifiedKFold

    layer = args.layer if args.layer >= 0 else n_layers + args.layer
    X = feats[args.pooling][:, layer, :].astype(np.float64)
    y, groups = df.label.to_numpy(), df.subject_id.to_numpy()
    rows = []

    for regime in ("file-level (leaky)", "subject-disjoint"):
        oof = np.zeros((args.repeats, len(y)))
        thrs = np.zeros(args.repeats)
        for r in range(args.repeats):
            if regime.startswith("file"):
                cv = StratifiedKFold(args.folds, shuffle=True, random_state=r)
                splits = cv.split(X, y)
            else:
                cv = StratifiedGroupKFold(args.folds, shuffle=True, random_state=r)
                splits = cv.split(X, y, groups)
            tt = []
            for tr, te in splits:
                m = train_probe(X[tr], y[tr], rank=args.rank,
                                epochs=args.epochs, seed=r)
                oof[r, te] = probe_scores(m, X[te])
                tt.append(youden_threshold(y[tr], probe_scores(m, X[tr])))
            thrs[r] = float(np.mean(tt))
        rows.append(summarize(df, oof, thrs, regime))
        r0 = rows[-1]
        print(f"{regime:22s} AUROC {r0['auroc']:.3f} +/- {r0['auroc_sd']:.3f}   "
              f"acc {r0['acc']:.3f}", flush=True)

    d_auroc = rows[0]["auroc"] - rows[1]["auroc"]
    d_acc = rows[0]["acc"] - rows[1]["acc"]
    print(f"\ninflation from speaker leakage: "
          f"+{d_auroc:.3f} AUROC, +{d_acc*100:.1f} accuracy points")
    pd.DataFrame(rows).to_csv("results/leakage_ablation.csv", index=False)
    print("saved -> results/leakage_ablation.csv")


def mode_mmse(df, feats, n_layers, args):
    """Does the cPCA projection track symptom severity, not just class?"""
    from scipy.stats import pearsonr, spearmanr

    layer = args.layer if args.layer >= 0 else n_layers + args.layer
    X = feats[args.pooling][:, layer, :].astype(np.float64)
    y, groups = df.label.to_numpy(), df.subject_id.to_numpy()

    oof, _ = run_cv(X, y, groups, args.mmse_method,
                    n_splits=args.folds, n_repeats=args.repeats)
    df = df.assign(model_score=oof.mean(0))

    have = df.dropna(subset=["mmse"])
    print(f"MMSE available for {len(have)}/{len(df)} files "
          f"({have.subject_id.nunique()} subjects)\n")

    for name, sub in (("all", have),
                      ("controls only", have[have.label == 0]),
                      ("impaired only", have[have.label == 1])):
        if len(sub) < 10:
            continue
        r, pr = pearsonr(sub.model_score, sub.mmse)
        rho, ps = spearmanr(sub.model_score, sub.mmse)
        print(f"{name:15s} n={len(sub):3d}   "
              f"Pearson r={r:+.3f} (p={pr:.2e})   Spearman rho={rho:+.3f} (p={ps:.2e})")

    # does the score track severity beyond simply tracking age?
    import numpy as _np
    sub = have.dropna(subset=["age"])
    def _resid(a, b):
        A = _np.c_[_np.ones(len(b)), b]
        return a - A @ _np.linalg.lstsq(A, a, rcond=None)[0]
    rp, pp = pearsonr(_resid(sub.model_score.to_numpy(dtype=float),
                             sub.age.to_numpy(dtype=float)),
                      _resid(sub.mmse.to_numpy(dtype=float),
                             sub.age.to_numpy(dtype=float)))
    print()
    print(f"partial correlation controlling for age: r={rp:+.3f} (p={pp:.2e}), n={len(sub)}")

    df.to_csv("results/mmse_scores.csv", index=False)
    print("\nsaved -> results/mmse_scores.csv")


def mode_nested(df, feats, n_layers, args):
    """Layer/pooling selection done honestly.

    Picking the best layer by looking at cross-validated AUROC and then
    reporting that same AUROC is selection bias -- with 66 candidate
    configurations the winner is partly winning by luck. Here the choice is made
    by an INNER CV on the training portion of each outer fold, so the reported
    outer score estimates the whole 'select a layer, then fit' pipeline.

    The gap between this and the raw sweep maximum is the optimism the sweep
    would have baked in.
    """
    y, groups = df.label.to_numpy(), df.subject_id.to_numpy()
    candidates = [(p, l) for p in args.nested_poolings.split(",")
                  for l in range(0, n_layers, args.layer_stride)]
    print(f"{len(candidates)} candidate configurations, "
          f"inner {args.inner_folds}-fold selection\n")

    chosen, outer_scores = [], np.zeros((args.nested_repeats, len(y)))
    thrs = np.zeros(args.nested_repeats)

    for r in range(args.nested_repeats):
        outer = StratifiedGroupKFold(args.folds, shuffle=True, random_state=r)
        fold_thr = []
        for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
            best_cfg, best_auc = None, -np.inf
            for pooling, layer in candidates:
                Xc = feats[pooling][:, layer, :].astype(np.float64)
                inner = StratifiedGroupKFold(args.inner_folds, shuffle=True,
                                             random_state=100 + r)
                inner_oof = np.zeros(len(tr))
                for itr, ite in inner.split(np.zeros(len(tr)), y[tr], groups[tr]):
                    inner_oof[ite] = _fit_predict(
                        args.nested_method, Xc[tr][itr], y[tr][itr], Xc[tr][ite])
                auc = roc_auc_score(y[tr], inner_oof)
                if auc > best_auc:
                    best_auc, best_cfg = auc, (pooling, layer)

            pooling, layer = best_cfg
            chosen.append(best_cfg)
            Xb = feats[pooling][:, layer, :].astype(np.float64)
            outer_scores[r, te] = _fit_predict(
                args.nested_method, Xb[tr], y[tr], Xb[te])
            s_tr = _fit_predict(args.nested_method, Xb[tr], y[tr], Xb[tr])
            fold_thr.append(youden_threshold(y[tr], s_tr))
            print(f"  repeat {r} fold {k}: selected {pooling} L{layer} "
                  f"(inner AUROC {best_auc:.3f})", flush=True)
        thrs[r] = float(np.mean(fold_thr))

    row = summarize(df, outer_scores, thrs,
                    f"{args.nested_method} (nested layer selection)")
    print(f"\nnested-CV AUROC {row['auroc']:.3f} +/- {row['auroc_sd']:.3f} "
          f"[{row['auroc_ci_lo']:.3f}, {row['auroc_ci_hi']:.3f}]")
    counts = pd.Series([f"{p} L{l}" for p, l in chosen]).value_counts()
    print("\nconfigurations selected across outer folds:")
    print(counts.to_string())
    pd.DataFrame([row]).to_csv("results/nested_selection.csv", index=False)
    print("\nsaved -> results/nested_selection.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["sweep", "main", "leakage", "mmse", "nested"],
                    default="main")
    ap.add_argument("--manifest", default="artifacts/manifest.csv")
    ap.add_argument("--activations", default="artifacts/activations.pt")
    ap.add_argument("--cohort", choices=list(COHORTS), default="control_vs_probablead")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sweep-repeats", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--nested-repeats", type=int, default=3)
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--nested-poolings", default="last,mean")
    ap.add_argument("--nested-method", choices=["cpca", "logreg", "probe"],
                    default="logreg")
    ap.add_argument("--mmse-method", choices=["cpca", "logreg", "probe"],
                    default="logreg")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    df, feats, n_layers = load_data(args.manifest, args.activations, args.cohort)
    print(f"cohort '{args.cohort}': {len(df)} files, "
          f"{df.subject_id.nunique()} subjects, {n_layers} layers cached\n")

    {"sweep": mode_sweep, "main": mode_main, "leakage": mode_leakage,
     "mmse": mode_mmse, "nested": mode_nested}[args.mode](df, feats, n_layers, args)


if __name__ == "__main__":
    raise SystemExit(main())
