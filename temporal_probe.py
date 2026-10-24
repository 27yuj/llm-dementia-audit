"""
Does the TRAJECTORY of a narrative carry signal that mean pooling throws away?

Mean pooling a transcript answers "what is this person talking about". It cannot
answer "did they hold the thread". AD narratives on the Cookie Theft picture
often open competently and then derail, loop back to an object already
described, or trail off -- all of which are properties of the SEQUENCE of
utterances, not of their average.

Arms, all under the same subject-disjoint protocol:

  mean            control. TOKEN-WEIGHTED mean of the utterance vectors ->
                  logistic regression. Token weighting is what makes this the
                  same estimator as the cached artifacts/L12_mean.npy behind the
                  established 0.9135; an unweighted mean over utterances is a
                  different pooling scheme and would not be a reproduction.
                  Every headline number in this file is a delta against THIS,
                  not against the number in the paper's table.

  mean-matched    the same control refitted on the subset of the training fold
                  that the sequence models actually train on (the inner-train
                  split, ~80% of the fold). The sequence arms hold out an inner
                  validation split to pick their stopping epoch and are never
                  refitted on the full fold, so they see less data than `mean`
                  and `traj` do. Comparing a sequence arm to `mean` alone
                  confounds "recurrence does not help" with "recurrence trained
                  on 20% fewer sequences". `mean-matched` measures that penalty
                  directly. It is a confound diagnostic, not a hypothesis test,
                  so it is excluded from the multiple-comparison family.

  traj            hand-built statistics over the utterance sequence: spread
                  across utterances, first-half minus second-half drift, the
                  projection onto the training-fold class direction and its
                  shape over time, cosine distance between consecutive
                  utterances (a derailment / jitter proxy), similarity between
                  non-adjacent utterances (a looping proxy), and length.

  traj-temporal   the same feature set with every CONTENT-LEVEL term removed.
                  What survives is exactly the set of features that is invariant
                  to adding a constant vector to all of a transcript's utterance
                  vectors: `Z.std`, the half-to-half drift, the shift-invariant
                  moments of the class-direction projection (sd, slope, first
                  minus last, range), the local-centred jitter and looping
                  blocks, and length. Dropped: `Z.mean` and the LEVEL of the
                  class projection (mean/max/min), which are the static content
                  mean read off in one number and nothing to do with narrative
                  shape. Invariance here is a property of the algebra, not an
                  empirical claim -- `--self-test` checks it numerically. It
                  does NOT mean the arm is content-blind in general: content
                  that expresses itself as within-transcript variation can still
                  be read. It means the arm cannot see where the transcript sits
                  in embedding space.

  cnn/gru/transformer
                  tiny sequence models over the utterance sequence, mean+max
                  pooled over time with a linear head.

GEOMETRY. Cosine features are taken on TRANSCRIPT-LOCALLY centred vectors,
`seq - seq.mean(0)`. An earlier version centred on the training-fold grand mean
`mu`; that leaves every utterance of a transcript sharing the offset
`(transcript mean - mu)`, so every pairwise cosine collapses to a monotone
function of ||transcript mean - mu|| -- a static content statistic wearing a
derailment label. Local centring removes the transcript's position exactly and
uses no labels and no other transcript, so it is fold-safe. It also makes the
"similarity to the transcript centroid" feature identically zero (the local
centroid IS zero), so that block is replaced by the non-adjacent-pair block plus
a lag-2 return term, which is the looping statistic that block was reaching for.

OVERFITTING IS THE POINT OF FAILURE HERE, not the metric. ~381 training
sequences against 3072-dim vectors will memorise anything, so every arm that
touches the sequence first goes through a PCA basis fitted ON THE TRAINING FOLD
ONLY (64 components), the models are kept at hidden size ~48 with one layer,
dropout on the pooled representation and weight decay on everything (the
transformer additionally has dropout inside its encoder layer; for the CNN and
GRU the real regulariser is early stopping plus weight decay), and the epoch
count is chosen by early stopping on an inner subject-disjoint split OF THE
TRAINING FOLD. Nothing -- PCA basis, scaler, class direction, regularisation
strength, stopping epoch, decision threshold -- ever sees the outer test fold.

THRESHOLDS deviate deliberately from the convention in src/evaluate.py, which
takes Youden's J on resubstitution training scores and then averages the
threshold across folds. Both halves of that hurt here. Resubstitution scores
from a near-separating GRU live on a completely different scale from those of a
C=0.01 logistic regression, so the two arms' acc/sens/spec would not be
comparable even though their AUROCs are; and averaging thresholds across folds
applies to each test sample a threshold that four of the five folds chose while
that sample's label was in their training set. Here each fold's threshold comes
from the INNER-VALIDATION scores already computed for model selection (out of
sample within the training fold) and is applied only to that fold's own test
rows. AUROC is threshold-free and therefore identical in definition to every
other number in the paper; only acc/sens/spec differ, and they differ in the
honest direction.

SIGNIFICANCE is a paired subject-level bootstrap (the same instrument as
src/fusion_test.py): subjects are resampled with replacement, every arm is
scored on the SAME resampled subjects within the same repeat, and the reported
interval is the interval of the DELTA against the control. The across-repeat SD
is split-seed noise on a fixed 477 files and is not a sampling interval; it must
not be used to decide whether an arm beats the control. p-values are Holm
corrected across the arms in the family.

Requires artifacts/utterance_embeddings.npz from src/extract_utterances.py.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import metrics_at, youden_threshold  # noqa: E402

PUBLISHED_MEANPOOL = 0.9135          # Phi-3 L12 mean-pooled + logreg, this corpus
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
CONTROL = "mean"
MATCHED = "mean-matched"
ARMS = [CONTROL, MATCHED, "traj", "traj-temporal", "cnn", "gru", "transformer"]
# mean-matched is a confound diagnostic, not a hypothesis about sequence
# structure; including it in the correction family would make every real test
# harder to pass, which is the direction that flatters this paper's null.
FAMILY = ["traj", "traj-temporal", "cnn", "gru", "transformer"]
LABEL = {
    CONTROL: "mean-pool baseline (control)",
    MATCHED: "mean-pool, sequence-model train subset",
    "traj": "trajectory statistics",
    "traj-temporal": "trajectory statistics, content level removed",
    "cnn": "1D-CNN over utterances",
    "gru": "GRU over utterances",
    "transformer": "Transformer encoder over utterances",
}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_sequences(manifest, utterances):
    """Cohort dataframe plus one [n_utterances, hidden] array per transcript."""
    blob = np.load(utterances, allow_pickle=False)
    emb, counts = blob["emb"], blob["counts"].astype(int)
    file_ids = [str(f) for f in blob["file_ids"]]
    bounds = np.concatenate([[0], np.cumsum(counts)])
    span = {f: (int(bounds[i]), int(bounds[i + 1])) for i, f in enumerate(file_ids)}

    df = pd.read_csv(manifest)
    df = df[(df.group == "Control") | (df.dx == "ProbableAD")].reset_index(drop=True)
    df["label"] = (df.dx == "ProbableAD").astype(int)

    missing = [f for f in df.file_id.astype(str) if f not in span]
    if missing:
        raise SystemExit(f"{len(missing)} cohort files absent from {utterances}: "
                         f"{missing[:5]}")

    seqs = [emb[span[str(f)][0]:span[str(f)][1]].astype(np.float64)
            for f in df.file_id.astype(str)]
    if "utt_tokens" in blob:
        t = blob["utt_tokens"].astype(np.float64)
        ntok = [t[span[str(f)][0]:span[str(f)][1]] for f in df.file_id.astype(str)]
    else:
        print("[warn] no utt_tokens in the npz; the control falls back to an "
              "UNWEIGHTED utterance mean and is no longer the published "
              "estimator", flush=True)
        ntok = [np.ones(len(s)) for s in seqs]
    return df, seqs, ntok, int(blob["layer"]) if "layer" in blob else -1


def pool_mean(seq, ntok):
    """Token-weighted mean of a transcript's utterance vectors.

    This is the transcript-level vector artifacts/L12_mean.npy holds: a mean
    over TOKEN positions, which weights an utterance by how many tokens it
    contributed. Averaging the utterance vectors unweighted would up-weight
    three-word utterances to parity with thirty-word ones and would be a
    different estimator from the 0.9135 baseline this arm has to reproduce.
    """
    w = np.asarray(ntok, dtype=np.float64)
    if w.sum() <= 0:
        return seq.mean(0)
    return np.average(seq, axis=0, weights=w)


# ----------------------------------------------------------------------------
# arm 2 -- trajectory statistics
# ----------------------------------------------------------------------------
def _slope(v):
    """Least-squares slope of v against time normalised to [0, 1]."""
    n = len(v)
    if n < 2:
        return 0.0
    t = np.linspace(0.0, 1.0, n)
    t = t - t.mean()
    return float(t @ (v - v.mean()) / (t @ t))


def traj_shape_features(seq, Z, ntok, include_content):
    """Label-free half of the trajectory features for one transcript.

    `Z` is the sequence projected onto the training-fold PCA basis. Cosines are
    taken on transcript-locally centred vectors so that no feature here can read
    the transcript's absolute position in embedding space (see module docstring).
    With `include_content=False` every returned feature is exactly invariant to
    `seq -> seq + c` for any constant vector c.
    """
    n = len(seq)
    feats = []
    if include_content:
        feats.append(Z.mean(0))                       # what was said
    feats.append(Z.std(0))                            # how much it moved
    half = int(np.ceil(n / 2))
    feats.append(Z[:half].mean(0) - Z[half:].mean(0) if n >= 2 else np.zeros(Z.shape[1]))

    L = seq - seq.mean(0)                             # transcript-local centring
    Ln = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-9)

    # jitter / derailment: how far each utterance jumps from the previous one
    scal = []
    if n >= 2:
        d = 1.0 - (Ln[:-1] * Ln[1:]).sum(1)
        scal += [float(d.mean()), float(d.std()), float(d.max()), float(d.min()),
                 _slope(d) if n >= 3 else 0.0]
    else:
        scal += [0.0] * 5

    # looping: similarity between non-adjacent utterances, and the lag-2 return
    if n >= 3:
        G = Ln @ Ln.T
        far = np.triu(np.ones_like(G, dtype=bool), k=2)   # skip self + neighbours
        g = G[far]
        scal += [float(g.max()), float(g.mean()), float(g.std()),
                 float((Ln[:-2] * Ln[2:]).sum(1).mean())]
    else:
        scal += [0.0] * 4

    # narrative length, in utterances and in tokens per utterance
    scal += [float(np.log1p(n)), float(np.log1p(np.mean(ntok))),
             float(np.std(ntok)) if n > 1 else 0.0]

    feats.append(np.nan_to_num(np.array(scal, dtype=np.float64)))
    return np.concatenate(feats)


def traj_projection_features(seq, w, include_level):
    """Shape of the class-direction projection over time.

    `w` is the training-fold class direction, so this block is label-derived and
    must be rebuilt inside every inner fold that selects on it. The LEVEL terms
    (mean/max/min of the projection) are content, not shape: `p.mean()` is the
    transcript's mean utterance vector projected onto the class direction, i.e.
    the mean-pool control's own score computed inline. They are gated off for
    the content-free arm; what remains is invariant to shifting p by a constant,
    hence to shifting the transcript by any constant vector.
    """
    p = seq @ w
    n = len(p)
    scal = [float(p.std()) if n > 1 else 0.0, _slope(p),
            float(p[0] - p[-1]), float(p.max() - p.min())]
    if include_level:
        scal += [float(p.mean()), float(p.max()), float(p.min())]
    return np.nan_to_num(np.array(scal, dtype=np.float64))


def class_direction(pooled, y):
    """Unit difference-of-class-means direction. Uses labels -- fold-local only."""
    if not (y == 1).any() or not (y == 0).any():
        return np.zeros(pooled.shape[1])
    w = pooled[y == 1].mean(0) - pooled[y == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-12)


def inner_select_C(ytr, gtr, inner_folds, seed, build, grid=None):
    """Pick the regularisation strength by inner subject-disjoint CV on train.

    `build(fit_idx, eval_idx) -> (X_fit, X_eval)` rebuilds the design matrix for
    each inner split, which is what lets the caller refit label-derived
    transforms (the class direction) INSIDE the inner fold. Fitting the
    direction once on the whole training fold and then selecting C on inner
    validation rows whose features were built with it would make the projection
    block look more discriminative during selection than it is at test time, and
    push the choice toward weaker regularisation.

    Returns (best_C, inner-OOF decision values at best_C). Those OOF values are
    out of sample within the training fold, which is what the decision threshold
    is taken on. A single-element `grid` pins C and still returns the OOF scores,
    so the threshold never has to come from resubstitution.
    """
    grid = list(grid or C_GRID)
    inner = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=seed)
    oof = {C: np.zeros(len(ytr)) for C in grid}
    for itr, ive in inner.split(np.zeros(len(ytr)), ytr, gtr):
        Xf, Xv = build(itr, ive)
        sc = StandardScaler().fit(Xf)
        Xf, Xv = sc.transform(Xf), sc.transform(Xv)
        for C in grid:
            clf = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
            clf.fit(Xf, ytr[itr])
            oof[C][ive] = clf.decision_function(Xv)
    best_C = max(grid, key=lambda C: roc_auc_score(ytr, oof[C]))
    return best_C, oof[best_C]


def fit_logreg(Xtr, ytr, C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=4000, C=C,
                             class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return lambda X: clf.decision_function(sc.transform(X))


# ----------------------------------------------------------------------------
# arm 3 -- tiny sequence models
# ----------------------------------------------------------------------------
def pad_batch(seqs, max_len=None):
    """[n, L, k] float32 tensor plus a [n, L] bool validity mask."""
    L = max_len or max(len(s) for s in seqs)
    k = seqs[0].shape[1]
    X = np.zeros((len(seqs), L, k), dtype=np.float32)
    M = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        n = min(len(s), L)
        X[i, :n] = s[:n]
        M[i, :n] = True
    return torch.from_numpy(X), torch.from_numpy(M)


def masked_pool(h, m):
    """Mean+max over time with padded positions excluded from both."""
    mf = m.unsqueeze(-1).to(h.dtype)
    mean = (h * mf).sum(1) / mf.sum(1).clamp(min=1.0)
    mx = h.masked_fill(~m.unsqueeze(-1), float("-inf")).max(1).values
    return torch.cat([mean, torch.nan_to_num(mx, neginf=0.0)], dim=-1)


class SeqCNN(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        self.conv = nn.Conv1d(d_in, hidden, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x, m):
        x = x * m.unsqueeze(-1).to(x.dtype)      # keep padding at exactly zero
        h = F.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        return self.head(self.drop(masked_pool(h, m))).squeeze(-1)


class SeqGRU(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x, m):
        lens = m.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.shape[1])
        return self.head(self.drop(masked_pool(h, m))).squeeze(-1)


class SeqTransformer(nn.Module):
    """Positions past the longest TRAINING sequence reuse the last trained one.

    Positional rows the training fold never reaches get no gradient and are
    decayed toward zero by AdamW, so a longer test transcript would be encoded
    with near-zero positional information exactly where the trained rows carry
    signal. Clamping the index is not leakage: `trained_len` is a property of
    the training split only.
    """

    def __init__(self, d_in, hidden, dropout, max_len=256, nhead=4):
        super().__init__()
        while hidden % nhead:
            nhead -= 1
        self.inp = nn.Linear(d_in, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.trained_len = max_len
        layer = nn.TransformerEncoderLayer(
            hidden, nhead, dim_feedforward=2 * hidden, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1,
                                         enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x, m):
        L = x.shape[1]
        if L > self.pos.shape[1]:
            raise ValueError(f"{L} utterances exceeds the positional table "
                             f"({self.pos.shape[1]}); raise max_len")
        idx = torch.arange(L, device=x.device).clamp(max=max(self.trained_len - 1, 0))
        h = self.inp(x) + self.pos[:, idx]
        h = self.enc(h, src_key_padding_mask=~m)
        return self.head(self.drop(masked_pool(h, m))).squeeze(-1)


BUILDERS = {"cnn": SeqCNN, "gru": SeqGRU, "transformer": SeqTransformer}


@torch.no_grad()
def seq_predict(model, X, M, device, batch=64):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(X[i:i + batch].to(device),
                         M[i:i + batch].to(device)).float().cpu())
    return torch.cat(out).numpy()


def train_sequence_model(kind, Ztr, ytr, Zte, split, args, seed):
    """Fit one sequence model on the training fold; score the test fold.

    The stopping epoch is chosen on an inner subject-disjoint split of the
    TRAINING fold, passed in by the caller so that every arm which claims to use
    "the sequence models' training subset" uses the identical rows, and the
    best-validation weights are restored. We keep the model fitted on that
    inner-train subset rather than refitting on the full fold at the chosen
    epoch: refitting would change the loss landscape the epoch was selected
    under. The cost is that this arm trains on ~20% fewer sequences than the
    `mean` and `traj` arms; the `mean-matched` arm exists to price that.

    Returns (test scores, threshold, epochs run, best inner-validation AUROC).
    The threshold is Youden's J on the inner-VALIDATION scores at the selected
    epoch -- out of sample within the training fold, and on the same logit scale
    as the test scores, unlike a threshold read off resubstitution scores.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    itr, iva = split

    # each split is padded to its own maximum: padded positions are masked out of
    # every pooling op, so the padded length has no effect on any prediction, and
    # keeping them independent means no property of the test fold -- not even its
    # longest sequence -- touches training
    Xtr, Mtr = pad_batch([Ztr[i] for i in itr])
    Xva, Mva = pad_batch([Ztr[i] for i in iva])
    Xte, Mte = pad_batch(Zte)
    ya = torch.tensor(ytr[itr], dtype=torch.float32)

    model = BUILDERS[kind](Ztr[0].shape[1], args.hidden, args.dropout).to(device)
    if hasattr(model, "trained_len"):
        model.trained_len = int(Xtr.shape[1])
    n_pos = float(max(ya.sum().item(), 1.0))
    pos_weight = torch.tensor([(len(ya) - n_pos) / n_pos], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    gen = torch.Generator().manual_seed(seed)
    best_auc, best_state, stale = -np.inf, copy.deepcopy(model.state_dict()), 0
    best_va, best_epoch, ran = np.zeros(len(iva)), 0, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = crit(model(Xtr[idx].to(device), Mtr[idx].to(device)),
                        ya[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        ran = ep + 1
        s_va = seq_predict(model, Xva, Mva, device)
        auc = roc_auc_score(ytr[iva], s_va) if len(set(ytr[iva])) > 1 else 0.5
        if auc > best_auc + 1e-5:
            best_auc, best_state, stale = auc, copy.deepcopy(model.state_dict()), 0
            best_va, best_epoch = s_va.copy(), ran
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    thr = youden_threshold(ytr[iva], best_va)
    return seq_predict(model, Xte, Mte, device), thr, best_epoch, float(best_auc)


# ----------------------------------------------------------------------------
# protocol
# ----------------------------------------------------------------------------
def run(df, seqs, ntoks, args):
    y = df.label.to_numpy()
    groups = df.subject_id.to_numpy()
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm '{a}'; choose from {ARMS}")

    oof = {a: np.zeros((args.repeats, len(y))) for a in arms}
    # one threshold per TEST ROW: the threshold its own fold chose. Averaging
    # thresholds across folds would apply to a sample a cut that four of the
    # five folds picked with that sample's label in their training set.
    thr = {a: np.zeros((args.repeats, len(y))) for a in arms}
    chosen_C = {a: [] for a in arms if a.startswith("traj") or a.startswith("mean")}
    epochs_used = {a: [] for a in arms if a in BUILDERS}
    t0 = time.time()

    for r in range(args.repeats):
        cv = StratifiedGroupKFold(args.folds, shuffle=True, random_state=r)

        for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            ytr, gtr = y[tr], groups[tr]
            seq_tr = [seqs[i] for i in tr]
            seq_te = [seqs[i] for i in te]
            tok_tr = [ntoks[i] for i in tr]
            tok_te = [ntoks[i] for i in te]

            # ---- everything below is fitted on the training fold only ----
            # the inner split the sequence models train against, computed once
            # so that mean-matched provably uses the identical rows
            seq_seed = 1000 * r + k
            inner = StratifiedGroupKFold(args.inner_folds, shuffle=True,
                                         random_state=seq_seed)
            split = next(iter(inner.split(np.zeros(len(ytr)), ytr, gtr)))
            itr_seq = split[0]

            Ptr = np.vstack([pool_mean(s, t) for s, t in zip(seq_tr, tok_tr)])
            Pte = np.vstack([pool_mean(s, t) for s, t in zip(seq_te, tok_te)])

            for arm in (CONTROL, MATCHED):
                if arm not in arms:
                    continue
                sub = np.arange(len(ytr)) if arm == CONTROL else itr_seq
                Psub = Ptr[sub]
                C, in_oof = inner_select_C(
                    ytr[sub], gtr[sub], args.inner_folds, 100 + r,
                    lambda a, b, P=Psub: (P[a], P[b]),
                    grid=[args.control_c] if args.control_c > 0 else None)
                score = fit_logreg(Psub, ytr[sub], C)
                oof[arm][r, te] = score(Pte)
                thr[arm][r, te] = youden_threshold(ytr[sub], in_oof)
                chosen_C[arm].append(C)

            need_pca = any(a not in (CONTROL, MATCHED) for a in arms)
            if need_pca:
                U = np.vstack(seq_tr)
                n_comp = int(min(args.pca, U.shape[0] - 1, U.shape[1]))
                pca = PCA(n_components=n_comp, svd_solver="randomized",
                          random_state=r).fit(U)
                Ztr = [pca.transform(s) for s in seq_tr]
                Zte = [pca.transform(s) for s in seq_te]

            for arm in arms:
                if arm.startswith("traj"):
                    keep = arm == "traj"
                    Str = np.vstack([traj_shape_features(s, Z, t, keep)
                                     for s, Z, t in zip(seq_tr, Ztr, tok_tr)])
                    Ste = np.vstack([traj_shape_features(s, Z, t, keep)
                                     for s, Z, t in zip(seq_te, Zte, tok_te)])

                    def design(fit_idx, eval_idx, S=Str, keep=keep):
                        # the class direction is refit on the inner-training
                        # rows only, so inner validation never sees a feature
                        # built with its own label
                        wi = class_direction(Ptr[fit_idx], ytr[fit_idx])
                        proj = {j: traj_projection_features(seq_tr[j], wi, keep)
                                for j in np.concatenate([fit_idx, eval_idx])}
                        return (np.hstack([S[fit_idx],
                                           np.vstack([proj[j] for j in fit_idx])]),
                                np.hstack([S[eval_idx],
                                           np.vstack([proj[j] for j in eval_idx])]))

                    C, in_oof = inner_select_C(ytr, gtr, args.inner_folds,
                                               100 + r, design)
                    w = class_direction(Ptr, ytr)
                    Ftr = np.hstack([Str, np.vstack(
                        [traj_projection_features(s, w, keep) for s in seq_tr])])
                    Fte = np.hstack([Ste, np.vstack(
                        [traj_projection_features(s, w, keep) for s in seq_te])])
                    score = fit_logreg(Ftr, ytr, C)
                    oof[arm][r, te] = score(Fte)
                    thr[arm][r, te] = youden_threshold(ytr, in_oof)
                    chosen_C[arm].append(C)

                elif arm in BUILDERS:
                    zsc = StandardScaler().fit(np.vstack(Ztr))
                    Str_z = [zsc.transform(z).astype(np.float32) for z in Ztr]
                    Ste_z = [zsc.transform(z).astype(np.float32) for z in Zte]
                    s_te, t_arm, ep, _ = train_sequence_model(
                        arm, Str_z, ytr, Ste_z, split, args, seed=seq_seed)
                    oof[arm][r, te] = s_te
                    thr[arm][r, te] = t_arm
                    epochs_used[arm].append(ep)

            print(f"  repeat {r} fold {k} done "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

        line = "  ".join(f"{a}={roc_auc_score(y, oof[a][r]):.3f}" for a in arms)
        print(f"repeat {r}: {line}", flush=True)

    return oof, thr, chosen_C, epochs_used, arms


# ----------------------------------------------------------------------------
# metrics and significance
# ----------------------------------------------------------------------------
def metrics_rows(y, s, thr):
    """metrics_at with a PER-ROW threshold; AUROC still taken on raw scores."""
    m = metrics_at(y, s - thr, 0.0)
    m["auroc"] = float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
    return m


def subject_bootstrap(df, oof, thr, arms, n_boot, seed):
    """Paired subject-level bootstrap: every arm scored on the SAME draw.

    Subjects, not files, are the unit of independence, and resampling within
    each repeat keeps the interval describing the same estimator as the reported
    point estimate (the mean per-repeat metric). Because all arms share the draw,
    differences of the stored draws are paired and the shared difficulty of a
    given resample cancels -- which is the whole point, since the arms are
    correlated and a marginal interval per arm says nothing about the delta.
    """
    rng = np.random.default_rng(seed)
    subs = df.subject_id.unique()
    by_sub = {s: np.flatnonzero(df.subject_id.to_numpy() == s) for s in subs}
    y = df.label.to_numpy()
    n_repeats = oof[arms[0]].shape[0]
    per_repeat = max(n_boot // n_repeats, 1)

    auc = {a: [] for a in arms}
    acc = {a: [] for a in arms}
    for r in range(n_repeats):
        for _ in range(per_repeat):
            pick = rng.choice(len(subs), len(subs), replace=True)
            idx = np.concatenate([by_sub[subs[p]] for p in pick])
            if len(set(y[idx])) < 2:
                continue
            for a in arms:
                auc[a].append(roc_auc_score(y[idx], oof[a][r][idx]))
                acc[a].append((((oof[a][r][idx] >= thr[a][r][idx]).astype(int)
                                == y[idx]).mean()))
    return ({a: np.array(v) for a, v in auc.items()},
            {a: np.array(v) for a, v in acc.items()})


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def delta_stats(draws, arm, ref):
    """Paired delta of AUROC draws, its 95% interval, and a two-sided p."""
    d = draws[arm] - draws[ref]
    n = max(len(d), 1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_win = float((d > 0).mean())
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {"delta": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_win": p_win, "p": float(min(max(p, 1.0 / n), 1.0))}


def summarize(df, oof, thr, arm, auc_draws, acc_draws):
    y = df.label.to_numpy()
    per = [metrics_rows(y, oof[arm][r], thr[arm][r])
           for r in range(oof[arm].shape[0])]
    row = {"model": LABEL[arm], "arm": arm}
    for key in ("auroc", "acc", "sens", "spec", "bal_acc"):
        row[key] = round(float(np.mean([m[key] for m in per])), 4)
        row[f"{key}_sd"] = round(float(np.std([m[key] for m in per])), 4)
    if auc_draws is not None:
        lo, hi = np.percentile(auc_draws[arm], [2.5, 97.5])
        row["auroc_ci_lo"], row["auroc_ci_hi"] = round(float(lo), 4), round(float(hi), 4)
        lo, hi = np.percentile(acc_draws[arm], [2.5, 97.5])
        row["acc_ci_lo"], row["acc_ci_hi"] = round(float(lo), 4), round(float(hi), 4)
    return row


# ----------------------------------------------------------------------------
# self test -- the content-invariance claim, checked rather than asserted
# ----------------------------------------------------------------------------
def self_test():
    """traj-temporal features must not move when a transcript is translated.

    The arm's entire claim is that it cannot read where a transcript sits in
    embedding space. That is an algebraic property of the feature set, so it is
    checkable exactly: shift every utterance of a transcript by a constant
    vector and every feature must be unchanged to floating-point noise.
    """
    rng = np.random.default_rng(0)
    ok = True
    for n in (1, 2, 3, 7, 25):
        seq = rng.normal(size=(n, 40))
        ntok = rng.integers(2, 30, size=n).astype(float)
        w = rng.normal(size=40)
        w /= np.linalg.norm(w)
        pca = PCA(n_components=8).fit(rng.normal(size=(200, 40)))
        shift = rng.normal(size=40) * 5.0

        for keep in (False, True):
            a = np.concatenate([traj_shape_features(seq, pca.transform(seq), ntok, keep),
                                traj_projection_features(seq, w, keep)])
            b = np.concatenate([traj_shape_features(seq + shift,
                                                    pca.transform(seq + shift),
                                                    ntok, keep),
                                traj_projection_features(seq + shift, w, keep)])
            gap = float(np.max(np.abs(a - b)))
            tag = "traj" if keep else "traj-temporal"
            if keep:
                print(f"  n={n:3d} {tag:14s} max|delta| {gap:.3e} (expected nonzero)")
            else:
                bad = gap > 1e-8
                ok = ok and not bad
                print(f"  n={n:3d} {tag:14s} max|delta| {gap:.3e} "
                      f"{'FAIL' if bad else 'ok'}")
    print("\ncontent-invariance:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifest.csv")
    ap.add_argument("--utterances", default="artifacts/utterance_embeddings.npz")
    ap.add_argument("--out", default="results/temporal_probe.csv")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=5,
                    help="inner subject-disjoint split for C / early stopping")
    ap.add_argument("--pca", type=int, default=64,
                    help="components fitted on the training fold's utterances")
    ap.add_argument("--control-c", type=float, default=0.0,
                    help="fixed C for the mean-pool control; 0 (default) selects "
                         "it by the same inner CV the traj arms get, so the "
                         "control is not handicapped relative to what it is "
                         "measured against. Pass 0.01 to reproduce the published "
                         "estimator exactly.")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--boot", type=int, default=2000,
                    help="paired subject bootstrap draws, split across repeats")
    ap.add_argument("--device", default="cpu",
                    help="the sequence models are tiny; cpu keeps the GPU free")
    ap.add_argument("--no-bootstrap", dest="bootstrap", action="store_false",
                    default=True)
    ap.add_argument("--save-oof", default="results/oof_temporal.npy")
    ap.add_argument("--self-test", action="store_true",
                    help="check the traj-temporal invariance claim and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    Path("results").mkdir(exist_ok=True)
    df, seqs, ntoks, layer = load_sequences(args.manifest, args.utterances)
    lens = np.array([len(s) for s in seqs])
    print(f"{len(df)} files, {df.subject_id.nunique()} subjects, "
          f"{int(df.label.sum())} ProbableAD / {int((1 - df.label).sum())} Control")
    print(f"utterance sequences: layer {layer}, dim {seqs[0].shape[1]}, "
          f"median {np.median(lens):.0f}  max {lens.max()}  total {lens.sum()}")
    # hyperparameters are pre-registered constants, not selected on the outer
    # test folds; printing and storing them is the audit trail that says so
    cfg = {k: v for k, v in vars(args).items() if k != "self_test"}
    cfg["argv"] = " ".join(sys.argv[1:])
    print(f"config: {json.dumps(cfg, sort_keys=True)}\n", flush=True)

    oof, thr, chosen_C, epochs_used, arms = run(df, seqs, ntoks, args)

    auc_draws = acc_draws = None
    if args.bootstrap:
        print("\npaired subject bootstrap ...", flush=True)
        auc_draws, acc_draws = subject_bootstrap(df, oof, thr, arms,
                                                 args.boot, seed=0)

    rows = [summarize(df, oof, thr, a, auc_draws, acc_draws) for a in arms]
    for row in rows:
        row.update({k: v for k, v in cfg.items()
                    if k in ("folds", "repeats", "inner_folds", "pca", "hidden",
                             "dropout", "lr", "weight_decay", "batch_size",
                             "epochs", "patience", "control_c", "argv")})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    if args.save_oof:
        np.save(args.save_oof, np.stack([oof[a] for a in arms]))
        thr_path = Path(args.save_oof).with_name(
            Path(args.save_oof).stem + "_thr.npy")
        np.save(thr_path, np.stack([thr[a] for a in arms]))

    print("\nControl vs ProbableAD, subject-disjoint "
          f"{args.folds}x{args.repeats}\n")
    ref = next((r for r in rows if r["arm"] == CONTROL), None)
    for r in rows:
        ci = (f"[{r['auroc_ci_lo']:.3f}, {r['auroc_ci_hi']:.3f}]"
              if "auroc_ci_lo" in r else "")
        print(f"  {r['model']:42s} AUROC {r['auroc']:.4f} +/- {r['auroc_sd']:.4f} "
              f"{ci:18s} acc {r['acc']:.3f}  sens {r['sens']:.3f}  "
              f"spec {r['spec']:.3f}")

    print(f"\npublished mean-pool baseline: {PUBLISHED_MEANPOOL:.4f}")
    if ref is not None:
        print(f"mean-pool control here:      {ref['auroc']:.4f} "
              f"(delta {ref['auroc'] - PUBLISHED_MEANPOOL:+.4f})")

    sig_rows = []
    if auc_draws is not None and CONTROL in arms:
        fam = [a for a in arms if a in FAMILY]
        stats = {a: delta_stats(auc_draws, a, CONTROL) for a in arms if a != CONTROL}
        adj = dict(zip(fam, holm(np.array([stats[a]["p"] for a in fam]))))
        print("\ndelta vs the mean-pool control, paired subject bootstrap "
              f"({len(auc_draws[CONTROL])} draws), Holm corrected over "
              f"{len(fam)} arms:")
        for a in arms:
            if a == CONTROL:
                continue
            s = stats[a]
            pa = adj.get(a)
            verdict = ("beats control" if pa is not None and pa < 0.05 and s["ci_lo"] > 0
                       else "below control" if pa is not None and pa < 0.05
                       else "not distinguishable from control")
            if a not in fam:
                verdict = "confound diagnostic, not in the test family"
            print(f"  {LABEL[a]:42s} {s['delta']:+.4f} "
                  f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  "
                  f"wins {s['p_win'] * 100:5.1f}%  "
                  f"p={s['p']:.3f}" + (f" p_holm={pa:.3f}" if pa is not None else "")
                  + f"  ({verdict})")
            sig_rows.append({"arm": a, "reference": CONTROL, "in_family": a in fam,
                             **{k: round(v, 4) for k, v in s.items()},
                             "p_holm": round(float(pa), 4) if pa is not None else None,
                             "verdict": verdict})

        # the sequence arms train on ~80% of the fold; against mean-matched the
        # comparison is data-matched and the recurrence question is isolated
        if MATCHED in arms:
            print("\ndata-matched delta (vs the control refit on the sequence "
                  "models' own training subset):")
            for a in arms:
                if a not in BUILDERS:
                    continue
                s = delta_stats(auc_draws, a, MATCHED)
                print(f"  {LABEL[a]:42s} {s['delta']:+.4f} "
                      f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  "
                      f"wins {s['p_win'] * 100:5.1f}%  p={s['p']:.3f}")
                sig_rows.append({"arm": a, "reference": MATCHED, "in_family": False,
                                 **{k: round(v, 4) for k, v in s.items()},
                                 "p_holm": None, "verdict": ""})

        # a null is only a result if it comes with the effect it could have seen
        worst = max(stats[a]["ci_hi"] for a in fam) if fam else float("nan")
        print(f"\nequivalence: no arm's 95% delta interval reaches above "
              f"{worst:+.4f} AUROC, so this design excludes gains larger than "
              f"that. Failure to reject is not equivalence; the interval is.")

    if sig_rows:
        sig_path = Path(args.out).with_name(Path(args.out).stem + "_significance.csv")
        pd.DataFrame(sig_rows).to_csv(sig_path, index=False)
        print(f"saved -> {sig_path}")

    for a, cs in chosen_C.items():
        if cs:
            print(f"\nC selected by inner CV for {a}: "
                  f"{pd.Series(cs).value_counts().to_dict()}")
    for a, es in epochs_used.items():
        if es:
            print(f"epochs to best inner-val AUROC for {a}: "
                  f"median {np.median(es):.0f}  max {max(es)}")
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
