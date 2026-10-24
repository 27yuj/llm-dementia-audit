"""Distribution of results under single-split evaluation (the common protocol).

Reproduces the widely used setup -- one random file-level 80/20 split, final
layer, last-token pooling -- across many seeds, to show how wide the outcome
distribution is on a corpus this size.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from models import probe_scores, train_probe  # noqa: E402

N_SEEDS = 40
df = pd.read_csv("artifacts/manifest.csv")
blob = torch.load("artifacts/activations.pt", weights_only=False)
order = {str(f): i for i, f in enumerate(blob["file_ids"])}
rows = df.file_id.astype(str).map(order).to_numpy()
X = np.array(blob["last"][rows][:, -1, :].numpy(), dtype=np.float64)
y = (df.group == "Dementia").astype(int).to_numpy()

aucs, accs = [], []
for seed in range(N_SEEDS):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              stratify=y, random_state=seed)
    m = train_probe(X[tr], y[tr], rank=16, epochs=100, seed=seed)
    s = probe_scores(m, X[te])
    aucs.append(roc_auc_score(y[te], s))
    accs.append(accuracy_score(y[te], (s >= 0.5).astype(int)))
    if (seed + 1) % 10 == 0:
        print(f"  {seed + 1}/{N_SEEDS}", flush=True)

aucs, accs = np.array(aucs), np.array(accs)
np.savez("results/single_split_variance.npz", auroc=aucs, acc=accs)
print(f"AUROC  mean {aucs.mean():.3f}  sd {aucs.std():.3f}  "
      f"range [{aucs.min():.3f}, {aucs.max():.3f}]")
print(f"acc    mean {accs.mean():.3f}  sd {accs.std():.3f}  "
      f"range [{accs.min():.3f}, {accs.max():.3f}]")
print(f"seeds >= 0.832 AUROC: {(aucs >= 0.832).sum()}/{N_SEEDS}")
print("saved -> results/single_split_variance.npz")
