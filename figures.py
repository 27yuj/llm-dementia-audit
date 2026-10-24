"""Generate every figure in the paper. Outputs PDF (for LaTeX) and PNG."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from disfluency import ANNOT_MARKERS, ASR_MARKERS  # noqa: E402

FIG = Path("figures")
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C_MEAN, C_LAST, C_REF, C_FUSE = "#1f5f8b", "#c0392b", "#7f8c8d", "#16a085"
TFIDF_REF, AGE_REF = 0.9087, 0.7060


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"))
    plt.close(fig)
    print("  wrote figures/" + name + ".pdf|.png")


# --------------------------------------------------------------- shared data
man = pd.read_csv("artifacts/manifest.csv")
dfl = man.merge(pd.read_csv("artifacts/disfluency.csv"), on="file_id")
dfl = dfl.merge(pd.read_csv("artifacts/perplexity.csv"), on="file_id")
keep = ((dfl.group == "Control") | (dfl.dx == "ProbableAD")).to_numpy()
d = dfl[keep].reset_index(drop=True)
d["label"] = (d.dx == "ProbableAD").astype(int)
y = d.label.to_numpy()
groups = d.subject_id.to_numpy()
subs = d.subject_id.unique()
by_sub = {s: np.flatnonzero(groups == s) for s in subs}


def boot_ci(oof, n_boot=1500, seed=0):
    """Subject-level bootstrap CI, matching the point estimate's estimator."""
    rng = np.random.default_rng(seed)
    vals = []
    for b in range(n_boot):
        pick = rng.choice(len(subs), len(subs), replace=True)
        idx = np.concatenate([by_sub[subs[p]] for p in pick])
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], oof[b % oof.shape[0]][idx]))
    point = float(np.mean([roc_auc_score(y, oof[r]) for r in range(oof.shape[0])]))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def tabular_oof(cols, n_repeats=5):
    """Out-of-fold scores for a dense tabular baseline, subject-disjoint."""
    X = np.array(d[cols].to_numpy(dtype=float), copy=True)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = np.nanmedian(X[:, j]) if (~m).any() else 0.0
    oof = np.zeros((n_repeats, len(y)))
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=r)
        for tr, te in cv.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                sc.transform(X[tr]), y[tr])
            oof[r, te] = clf.decision_function(sc.transform(X[te]))
    return oof


# ------------------------------------------------- Fig 1: layer/pooling sweep
# Column-width, single panel. The cPCA curve that used to be panel (b) is flat
# and uninformative to look at; its numbers are in the text and in fig3.
sw = pd.read_csv("results/layer_sweep.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.45))
for pool, col, mk in (("mean", C_MEAN, "o"), ("last", C_LAST, "s")):
    g = sw[(sw.method == "logreg") & (sw.pooling == pool)].sort_values("layer")
    ax.plot(g.layer, g.auroc, marker=mk, ms=2.4, lw=1.2, color=col,
            label=pool + "-token pooling")
    ax.fill_between(g.layer, g.auroc - g.auroc_sd, g.auroc + g.auroc_sd,
                    color=col, alpha=0.15, lw=0)
ax.axhline(TFIDF_REF, ls="--", lw=0.9, color=C_REF)
ax.axhline(AGE_REF, ls=":", lw=0.9, color=C_REF)
ax.set_xlabel("Phi-3 hidden layer")
ax.set_ylabel("AUROC (subject-disjoint)")
ax.set_xlim(-0.5, 32.5)
ax.set_ylim(0.60, 0.96)
ax.annotate("TF-IDF baseline (0.909)", xy=(15, TFIDF_REF), xytext=(7.5, 0.925),
            fontsize=7.0, color=C_REF)
ax.annotate("age-only (0.706)", xy=(15, AGE_REF), xytext=(0.6, 0.716),
            fontsize=7.0, color=C_REF)
best = sw[(sw.method == "logreg") & (sw.pooling == "mean")].nlargest(1, "auroc").iloc[0]
ax.plot([best.layer], [best.auroc], marker="*", ms=10, color="#f39c12",
        mec="k", mew=0.4, zorder=5, ls="none")
ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=7.0)
save(fig, "fig1_layer_sweep")

# ------------------------------------------------ Fig 2: single-split variance
v = np.load("results/single_split_variance.npz")
auc = v["auroc"]
fig, ax = plt.subplots(figsize=(3.4, 2.35))
ax.hist(auc, bins=14, color=C_MEAN, alpha=0.75, edgecolor="white", lw=0.5)
ax.axvline(auc.mean(), color="k", lw=1.3, label="mean %.3f" % auc.mean())
ax.axvline(auc.max(), color=C_LAST, lw=1.3, ls="--",
           label="best seed %.3f" % auc.max())
ax.axvline(0.5, color="k", lw=0.6, alpha=0.4)
ax.set_xlabel("AUROC from one random 80/20 file-level split")
ax.set_ylabel("number of seeds")
ax.set_title("40 seeds, identical model (SD = %.3f)" % auc.std(), fontsize=8)
ax.legend(frameon=True, loc="upper left")
save(fig, "fig2_single_split_variance")

# ----------------------------------------------------- Fig 3: the landscape
# Reads CIs straight from the result CSVs so the figure cannot drift from the
# tables. Colour encodes the paper's argument: grey = non-neural reference,
# blue = frozen + simple, red = an elaboration that was expected to help,
# green = the winning combination.
tp = pd.read_csv("results/temporal_probe.csv").set_index("arm")
sg = pd.read_csv("results/stacking_gbm.csv")
ql = pd.read_csv("results/qlora.csv").set_index("model")
m12 = pd.read_csv("results/main_L12_mean.csv").set_index("method")
m32 = pd.read_csv("results/main_L32_last.csv").set_index("method")

C_BASE, C_SIMPLE, C_ELAB, C_WIN = "#7f8c8d", "#1f5f8b", "#c0392b", "#16a085"

oof_f = np.load("results/oof_fusion.npy")      # phi3, tfidf, late, early
dis_cols = [k + "_per100" for k in list(ASR_MARKERS) + list(ANNOT_MARKERS)]
ppl_cols = ["nll_mean", "nll_median", "nll_std", "nll_p90", "frac_surprising"]


def row(lbl, auroc, lo, hi, col):
    return (lbl, float(auroc), float(lo), float(hi), col)


age_m, age_lo, age_hi = boot_ci(tabular_oof(["age"]))
dis_m, dis_lo, dis_hi = boot_ci(tabular_oof(dis_cols))
ppl_m, ppl_lo, ppl_hi = boot_ci(tabular_oof(ppl_cols))
tf_m, tf_lo, tf_hi = boot_ci(oof_f[1])

ENTRIES = [
    row("Age only", age_m, age_lo, age_hi, C_BASE),
    row("LLM surprisal", ppl_m, ppl_lo, ppl_hi, C_BASE),
    row("Disfluency counts", dis_m, dis_lo, dis_hi, C_BASE),
    row("TF-IDF uni+bigram", tf_m, tf_lo, tf_hi, C_BASE),
    None,
    row("cPCA, zero-shot (RepE)", m12.loc["cpca", "auroc"],
        m12.loc["cpca", "auroc_ci_lo"], m12.loc["cpca", "auroc_ci_hi"], C_ELAB),
    row("GRU over utterances", tp.loc["gru", "auroc"],
        tp.loc["gru", "auroc_ci_lo"], tp.loc["gru", "auroc_ci_hi"], C_ELAB),
    row("Transformer over utterances", tp.loc["transformer", "auroc"],
        tp.loc["transformer", "auroc_ci_lo"], tp.loc["transformer", "auroc_ci_hi"], C_ELAB),
    row("1D-CNN over utterances", tp.loc["cnn", "auroc"],
        tp.loc["cnn", "auroc_ci_lo"], tp.loc["cnn", "auroc_ci_hi"], C_ELAB),
    row("QLoRA fine-tune (4-bit)", ql.loc["Phi-3 QLoRA 4-bit fine-tune", "auroc_pooled"],
        ql.loc["Phi-3 QLoRA 4-bit fine-tune", "auroc_ci_lo"],
        ql.loc["Phi-3 QLoRA 4-bit fine-tune", "auroc_ci_hi"], C_ELAB),
    row("Trajectory statistics", tp.loc["traj", "auroc"],
        tp.loc["traj", "auroc_ci_lo"], tp.loc["traj", "auroc_ci_hi"], C_ELAB),
    row("XGBoost meta-learner", sg.iloc[4]["auroc"],
        sg.iloc[4]["auroc_ci_lo"], sg.iloc[4]["auroc_ci_hi"], C_ELAB),
    row("LightGBM meta-learner", sg.iloc[3]["auroc"],
        sg.iloc[3]["auroc_ci_lo"], sg.iloc[3]["auroc_ci_hi"], C_ELAB),
    None,
    row("Phi-3 L32, last-token", m32.loc["logreg", "auroc"],
        m32.loc["logreg", "auroc_ci_lo"], m32.loc["logreg", "auroc_ci_hi"], C_SIMPLE),
    row("Phi-3 L12, mean-pooled", m12.loc["logreg", "auroc"],
        m12.loc["logreg", "auroc_ci_lo"], m12.loc["logreg", "auroc_ci_hi"], C_SIMPLE),
    row("Late fusion (average)", sg.iloc[0]["auroc"],
        sg.iloc[0]["auroc_ci_lo"], sg.iloc[0]["auroc_ci_hi"], C_WIN),
    row("Linear meta-learner, 4 bases", sg.iloc[2]["auroc"],
        sg.iloc[2]["auroc_ci_lo"], sg.iloc[2]["auroc_ci_hi"], C_WIN),
]

fig, ax = plt.subplots(figsize=(3.4, 4.15))
ypos, labels = [], []
i = 0
for e in ENTRIES:
    if e is None:
        i += 0.6
        continue
    lbl, m, lo, hi, col = e
    ax.errorbar([m], [i], xerr=[[m - lo], [hi - m]], fmt="o", ms=3.6, color=col,
                capsize=1.8, lw=1.1, mec="k", mew=0.3)
    ax.text(hi + 0.012, i, "%.3f" % m, va="center", fontsize=7.0)
    ypos.append(i)
    labels.append(lbl)
    i += 1
ax.set_yticks(ypos)
ax.set_yticklabels(labels)
ax.axvline(TFIDF_REF, ls="--", lw=0.8, color=C_BASE, alpha=0.8)
ax.set_xlabel("AUROC (95% subject-bootstrap CI)")
ax.set_xlim(0.42, 1.08)
ax.invert_yaxis()
for lbl, col, xx, yy in (("non-neural", C_BASE, 0.70, 0.985),
                         ("elaborations", C_ELAB, 0.46, 0.738),
                         ("frozen + simple", C_SIMPLE, 0.035, 0.238)):
    ax.text(xx, yy, lbl, transform=ax.transAxes, fontsize=7.0, color=col,
            style="italic", va="top")
save(fig, "fig3_method_comparison")

# ------------------------------------------------- Fig 4: perplexity paradox
# Column-width (not full-width): a `figure*` can only be placed at the top of a
# page in a two-column layout, and when no top slot is free it defers to a page
# of its own. Both panels still fit at \columnwidth with reduced label sizes.
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.62))
ax = axes[0]
for lab, grp, col in (("Control", 0, C_MEAN), ("ProbableAD", 1, C_LAST)):
    ax.hist(d[d.label == grp].nll_mean, bins=22, alpha=0.6, color=col,
            label=lab, density=True, edgecolor="white", lw=0.25)
ax.set_xlabel("surprisal (nats/token)", fontsize=7.0, labelpad=1.5)
ax.set_ylabel("density", fontsize=7.0, labelpad=1.5)
ax.set_title("(a) significant shift", fontsize=6.8, pad=2.5)
ax.tick_params(labelsize=5.4, pad=1.2)
ax.legend(frameon=True, fontsize=5.2, handlelength=1.0, borderpad=0.25,
          labelspacing=0.25, loc="upper right")

ax = axes[1]
names = ["Phi-3", "TF-IDF", "surp."]
w = [1.408, 1.235, 0.192]
err = [0.244, 0.176, 0.087]
bars = ax.bar(names, w, yerr=err, capsize=1.8, color=[C_MEAN, C_LAST, C_REF],
              width=0.62, edgecolor="k", lw=0.35, error_kw={"lw": 0.7})
for b, val in zip(bars, w):
    ax.text(b.get_x() + b.get_width() / 2, val + err[bars.index(b)] + 0.12,
            "%.2f" % val, ha="center", fontsize=5.8)
ax.set_ylabel("stacking weight", fontsize=7.0, labelpad=1.5)
ax.set_title("(b) 6.8% of weight", fontsize=6.8, pad=2.5)
ax.tick_params(labelsize=5.4, pad=1.2)
ax.set_ylim(0, 2.0)
fig.subplots_adjust(wspace=0.42)
save(fig, "fig4_perplexity_paradox")

print("all figures written to figures/")
