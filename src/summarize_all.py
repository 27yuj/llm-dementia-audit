"""Collect every result file into one table for the paper."""

from pathlib import Path

import pandas as pd

base = Path("results")
rows = []


def add(family, model, r, **extra):
    rows.append({"family": family, "model": model,
                 "auroc": getattr(r, "auroc", None),
                 "auroc_sd": getattr(r, "auroc_sd", None), **extra})


if (base / "baselines.csv").exists():
    for r in pd.read_csv(base / "baselines.csv").itertuples():
        add("non-neural baseline", r.baseline, r)

for f in sorted(base.glob("main_L*.csv")):
    layer, pooling = f.stem.replace("main_L", "").split("_")
    for r in pd.read_csv(f).itertuples():
        add(f"Phi-3 L{layer} {pooling}", r.method, r, acc=r.acc, sens=r.sens,
            spec=r.spec, ci_lo=r.auroc_ci_lo, ci_hi=r.auroc_ci_hi)

if (base / "nested_selection.csv").exists():
    for r in pd.read_csv(base / "nested_selection.csv").itertuples():
        add("Phi-3 nested layer selection", r.method, r, acc=r.acc, sens=r.sens,
            spec=r.spec, ci_lo=r.auroc_ci_lo, ci_hi=r.auroc_ci_hi)

for fname, family in (("temporal_probe.csv", "temporal"),
                      ("stacking_gbm.csv", "stacking"),
                      ("qlora.csv", "qlora"),
                      ("fusion.csv", "fusion"),
                      ("stacked_fusion.csv", "fusion (stacked)"),
                      ("perplexity.csv", "surprisal"),
                      ("perplexity_fusion.csv", "fusion + surprisal")):
    if (base / fname).exists():
        d = pd.read_csv(base / fname)
        if "auroc" not in d.columns and "auroc_pooled" in d.columns:
            d = d.rename(columns={"auroc_pooled": "auroc",
                                  "auroc_pooled_sd": "auroc_sd"})
        col = next((c for c in ("model", "arm", "features", "baseline")
                    if c in d.columns), d.columns[0])
        for r in d.itertuples():
            add(family, getattr(r, col), r)

out = pd.DataFrame(rows).sort_values("auroc", ascending=False)
out.to_csv(base / "ALL_RESULTS.csv", index=False)
cols = [c for c in ("family", "model", "auroc", "auroc_sd", "ci_lo", "ci_hi",
                    "acc", "sens", "spec") if c in out.columns]
print(out[cols].head(18).to_string(index=False, na_rep="-"))
print(f"\n({len(out)} rows total) saved -> {base / 'ALL_RESULTS.csv'}")
