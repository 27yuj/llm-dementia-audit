"""Contrastive PCA direction-finder and the streamlined linear probe."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Contrastive PCA
# ----------------------------------------------------------------------------
def contrastive_pca(target: np.ndarray, background: np.ndarray, alpha: float = 1.0,
                    n_pre_pca: int | None = None) -> np.ndarray:
    """Direction maximising target variance while penalising background variance.

    The contrastive covariance C = cov(T) - alpha*cov(B) is 3072x3072 but built
    from only ~450 samples, so range(C) lies inside the span of the centred
    samples. We therefore solve the eigenproblem in that span and map back --
    numerically identical to the full 3072x3072 eigendecomposition, but ~100x
    faster, which is what makes the layer sweep tractable.

    `n_pre_pca` optionally truncates that span further (Abid et al.'s
    regularised cPCA), trading exactness for variance reduction.
    Returns a unit vector in the ORIGINAL feature space.
    """
    target = np.asarray(target, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)

    tc = target - target.mean(0, keepdims=True)
    bc = background - background.mean(0, keepdims=True)

    # orthonormal basis for the span of the centred samples
    _, sv, vt = np.linalg.svd(np.vstack([tc, bc]), full_matrices=False)
    rank = int((sv > sv[0] * 1e-10).sum()) if sv[0] > 0 else 1
    if n_pre_pca:
        rank = min(rank, n_pre_pca)
    Q = vt[:rank].T                                   # [H, r]

    tq, bq = tc @ Q, bc @ Q
    cov_t = tq.T @ tq / max(len(tq) - 1, 1)
    cov_b = bq.T @ bq / max(len(bq) - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov_t - alpha * cov_b)
    vec = Q @ eigvecs[:, int(np.argmax(eigvals))]
    return vec / (np.linalg.norm(vec) + 1e-12)


def fit_cpca_scorer(X_tr, y_tr, alpha=1.0, n_pre_pca=None):
    """Fit direction, centring, and sign on TRAINING data only.

    Returns a callable mapping features -> score where higher == more impaired.
    Sign and centre are locked from train; nothing is chosen on the test fold.
    """
    mu = X_tr.mean(0, keepdims=True)
    vec = contrastive_pca(X_tr[y_tr == 1], X_tr[y_tr == 0], alpha, n_pre_pca)

    proj_tr = (X_tr - mu) @ vec
    # orient so the impaired class sits on the positive side
    if proj_tr[y_tr == 1].mean() < proj_tr[y_tr == 0].mean():
        vec = -vec
        proj_tr = -proj_tr

    # threshold by Youden's J on train only
    order = np.argsort(proj_tr)
    cands = (proj_tr[order][:-1] + proj_tr[order][1:]) / 2
    best_j, thr = -np.inf, float(np.median(proj_tr))
    pos, neg = (y_tr == 1).sum(), (y_tr == 0).sum()
    for c in cands:
        pred = proj_tr >= c
        tpr = (pred & (y_tr == 1)).sum() / max(pos, 1)
        fpr = (pred & (y_tr == 0)).sum() / max(neg, 1)
        if tpr - fpr > best_j:
            best_j, thr = tpr - fpr, float(c)

    def score(X):
        return (np.asarray(X, dtype=np.float64) - mu) @ vec

    return score, thr, vec


# ----------------------------------------------------------------------------
# Streamlined probe
# ----------------------------------------------------------------------------
class StreamlinedProbe(nn.Module):
    """LayerNorm -> low-rank bottleneck -> GELU -> classifier.

    GELU (not ReLU) and bias=True on both projections: the unnormalised Phi-3
    activations killed a bias-free ReLU bottleneck outright.
    """

    def __init__(self, input_dim=3072, rank=16, num_classes=2, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, rank, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(rank, num_classes, bias=True)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.down_proj(h))
        h = self.dropout(h)
        return self.up_proj(h)


def train_probe(X_tr, y_tr, rank=16, epochs=100, lr=1e-3, weight_decay=0.01,
                batch_size=32, dropout=0.2, seed=0):
    """Train the probe on one fold. Class weights counter the label imbalance."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)

    counts = torch.bincount(yt, minlength=2).float()
    weights = counts.sum() / (2.0 * counts.clamp(min=1))

    model = StreamlinedProbe(Xt.shape[1], rank=rank, dropout=dropout)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(Xt)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model


@torch.no_grad()
def probe_scores(model, X):
    logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.softmax(logits, dim=1)[:, 1].numpy()
