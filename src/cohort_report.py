"""Cohort description + a direct measurement of speaker leakage under file-level splitting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("artifacts/manifest.csv")

print("=" * 68)
print("COHORT SUMMARY  (Pitt Cookie Theft, PAR-only)")
print("=" * 68)
tbl = df.groupby("group").agg(
    files=("file_id", "count"),
    subjects=("subject_id", "nunique"),
    age_mean=("age", "mean"),
    age_sd=("age", "std"),
    mmse_mean=("mmse", "mean"),
    mmse_sd=("mmse", "std"),
    mmse_missing=("mmse", lambda s: s.isna().sum()),
    words_median=("n_words", "median"),
)
tbl["visits_per_subject"] = (tbl["files"] / tbl["subjects"]).round(2)
print(tbl.round(2).to_string())

print("\n-- sex --")
print(pd.crosstab(df["group"], df["sex"]).to_string())

print("\n-- per-visit clinical label inside the 'Dementia' folder --")
print(df[df.group == "Dementia"]["dx"].value_counts(dropna=False).to_string())

print("\n-- subjects appearing in BOTH folders (converters) --")
both = (
    df.groupby("subject_id")["group"].nunique().pipe(lambda s: s[s > 1]).index.tolist()
)
for sid in both:
    sub = df[df.subject_id == sid].sort_values("visit_id")
    print(f"  subject {sid}: " + ", ".join(
        f"v{r.visit_id}={r.dx}(mmse {r.mmse})" for r in sub.itertuples()
    ))

print("\n" + "=" * 68)
print("LEAKAGE UNDER RANDOM FILE-LEVEL SPLIT  (the original Colab setup)")
print("=" * 68)
rng_overlaps, rng_leaked_rows = [], []
for seed in range(200):
    tr, te = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    shared = set(tr.subject_id) & set(te.subject_id)
    rng_overlaps.append(len(shared))
    rng_leaked_rows.append(te.subject_id.isin(shared).sum() / len(te))

print(f"test set size                     : {len(te)} files")
print(f"subjects present in train AND test: "
      f"{np.mean(rng_overlaps):.1f} +/- {np.std(rng_overlaps):.1f}  (over 200 seeds)")
print(f"fraction of TEST files whose speaker was also seen in training: "
      f"{np.mean(rng_leaked_rows)*100:.1f}%")
print("\nInterpretation: with 80/20 file-level shuffling, roughly "
      f"{np.mean(rng_leaked_rows)*100:.0f}% of the held-out transcripts come from a "
      "speaker\nthe probe already trained on. Reported accuracy/AUROC are inflated by "
      "speaker\nidentity, not purely by cognitive status.")
