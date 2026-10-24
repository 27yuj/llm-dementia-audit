"""
The ceiling test: does actually fine-tuning Phi-3 beat frozen mid-layer extraction?

Every number in this paper so far comes from a FROZEN model -- hidden states are
extracted once and a linear model is fit on top (L12 mean-pooled + logreg =
0.9135 AUROC). The obvious objection is that we never let the network adapt. So
here we fine-tune the whole thing end-to-end: 4-bit NF4 quantised Phi-3-mini with
LoRA adapters on every attention and MLP projection, a trainable classification
head, and a class-weighted loss.

WHY THIS IS STRUCTURED THE WAY IT IS
------------------------------------
1. PAIRED BY CONSTRUCTION. A fine-tune scored against a number computed on some
   other partition is not a comparison, it is a coincidence. The frozen L12
   logistic-regression probe is refit inside THIS script on THE SAME
   StratifiedGroupKFold folds, from the same seeds, and we report the per-fold
   difference. That is the only version of "QLoRA vs frozen" that means anything.

1b. ONE ESTIMATOR, NOT TWO. The headline AUROC is POOLED out-of-fold per repeat,
   averaged over repeats -- identical to evaluate.py:summarize, which is what
   produced the 0.9135 this is measured against. Mean-of-per-fold-AUROCs is a
   different and systematically ~+0.003 more optimistic estimator on this
   cohort; it is still reported, in its own clearly-named column, but it is
   never the number placed beside a cross-script reference. acc/sens/spec use
   evaluate.py's convention too (mean of the fold training thresholds applied to
   the pooled OOF vector) so the rows are comparable to results/main_L12_mean.csv.

2. NO EARLY STOPPING. Early stopping needs a validation set, and the only honest
   place to carve one out is inside the training fold -- which at ~381 training
   transcripts costs real signal. Rather than spend it, the epoch budget is a
   fixed hyperparameter chosen a priori (3) and applied identically to every
   fold. Nothing about the test fold touches any training decision. Peeking at
   test AUROC to pick an epoch is exactly the bug this paper documents.

3. SUBJECT-DISJOINT, ALWAYS. The Pitt corpus is longitudinal; many speakers
   appear at several visits. Grouping on subject_id is what keeps this from
   being a speaker-identification task wearing a diagnosis costume.

4. A NULL MUST BE DISTINGUISHABLE FROM A FAILED RUN. "Fine-tuning did not help"
   and "the fine-tune silently did nothing" produce the same test AUROC. So the
   TRAIN-fold AUROC of both arms is recorded every fold, along with the count of
   optimiser steps the GradScaler skipped on inf/nan. If train AUROC is not
   ~1.0, the adapters did not learn and the null is uninterpretable, not
   evidence.

5. RUN IT ONCE. Every invocation appends its full argument vector to
   results/qlora_runs.jsonl and the script shouts if the hyperparameters differ
   from a previous run. Nothing here stops an operator sweeping --epochs and
   keeping the best table, but that would make the number test-selected, and the
   log makes it visible rather than deniable. Smoke runs (--limit / --max-steps)
   are forced onto a separate out-prefix so they can never overwrite the real
   results file.

HONEST READING OF A NULL RESULT
-------------------------------
If QLoRA does not beat 0.9135 here, that shows fine-tuning does not help AT THIS
SAMPLE SIZE -- roughly 381 training transcripts of ~110 tokens each, against a
3.8B-parameter model. It does NOT show that fine-tuning is useless for dementia
detection in general. With an order of magnitude more data the conclusion could
easily flip. The claim this experiment can support is narrow and should be
reported narrowly: at DementiaBank-Pitt scale, frozen mid-layer features are not
the bottleneck.

And "no difference" is not what a null here licenses either. Five folds paired
by fold give t(4) and a minimum detectable |delta| of roughly 1.24 x SD of the
per-fold difference, which on this cohort lands near 0.03 AUROC. A null should
therefore be phrased "not better by more than about 0.03 AUROC", never "no
difference". The paired subject-level bootstrap on the pooled OOF scores is the
interval to quote; it captures test-set sampling noise only, not fine-tuning
seed variance (that would need repeated fine-tunes on identical folds).

MEMORY
------
Sized for a single 8.59 GB RTX 4060 Laptop (~7.45 GB free): 4-bit base weights
(~1.9 GB), gradient checkpointing, per-device batch 4 with accumulation to an
effective 8, max_length 512, fp16 autocast, paged 8-bit AdamW. Budget is roughly
1.9 GB quantised weights + 0.4 GB embeddings (peft upcasts the fp16 embedding
table to fp32) + 0.2 GB LoRA weights and grads + 0.05 GB optimiser state +
activations, so ~4 GB allocated and ~5 GB reserved against 7.4 GB free.

Batch 1 fits with more room but is ~3x slower: each micro-step re-dequantises
all 1.9 GB of NF4 weights to run a GEMM over ~120 tokens, so this loop is
dequantisation-bandwidth bound and batching is nearly free. Fall back to
--batch-size 1 --grad-accum 8 if anything OOMs; lower --max-length or
--eval-batch-size after that.

max_length 512 covers 99.8% of transcripts uncut (exactly 1 of 477 is clipped).
That percentage was measured over the whole cohort, test folds included, but it
is a label-independent memory knob fixed a priori and cannot move AUROC.

The model is explicitly deleted and the allocator emptied between folds, because
five sequential fine-tunes in one process is exactly how you end up fragmented.

A plain PyTorch loop is used rather than transformers.Trainer on purpose: the
v5 TrainingArguments surface has churned (`evaluation_strategy` is now
`eval_strategy`, `warmup_ratio` is gone entirely on the installed 5.15.1), and a
hand-written loop has no version-dependent argument names to get wrong.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# This machine's antivirus intercepts TLS with its own root CA, which lives in the
# Windows trust store but not in certifi's bundle. Use the OS store so HuggingFace
# lookups work with certificate verification still enabled.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import metrics_at, subject_bootstrap_ci, youden_threshold  # noqa: E402

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# Phi-3 names its fused projections differently from Llama: one qkv_proj instead
# of q/k/v_proj, one gate_up_proj instead of gate_proj/up_proj.
TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_cohort(manifest, l12, order_csv):
    """The primary cohort: Control vs ProbableAD, 477 files / 246 subjects.

    L12_mean.npy is a positional cache with no keys of its own, so if the
    manifest were ever rebuilt in a different order the probe would silently be
    fit on mismatched features. Verify against the recorded row order and refuse
    to run rather than produce a plausible wrong number.
    """
    df = pd.read_csv(manifest)
    X12 = np.load(l12)
    if len(X12) != len(df):
        raise SystemExit(f"{l12} has {len(X12)} rows but {manifest} has {len(df)}")

    order_p = Path(order_csv)
    if order_p.exists():
        order = pd.read_csv(order_p)
        same = (order.file_id.astype(str).to_numpy() == df.file_id.astype(str).to_numpy())
        if not same.all():
            raise SystemExit(
                f"{l12} row order disagrees with {manifest} at {int((~same).sum())} "
                "rows -- re-run extract_activations / the L12 cache before trusting this"
            )
    else:
        print(f"WARNING: {order_csv} missing, cannot verify L12 row alignment", flush=True)

    keep = ((df.group == "Control") | (df.dx == "ProbableAD")).to_numpy()
    df = df[keep].reset_index(drop=True)
    df["label"] = (df.dx == "ProbableAD").astype(int)
    return df, np.asarray(X12[keep], dtype=np.float64)


def subsample_subjects(df, X12, limit, folds, seed=0):
    """Smoke-test shrink that removes SUBJECTS, never rows.

    Dropping random rows would leave subjects half in and half out, which still
    splits correctly but stops resembling the real problem. Keep whole speakers.
    """
    rng = np.random.default_rng(seed)
    subs = df.subject_id.unique().copy()
    rng.shuffle(subs)
    take, n = [], 0
    for s in subs:
        take.append(s)
        n += int((df.subject_id == s).sum())
        sub_lab = df.label[df.subject_id.isin(take)]
        if n >= limit and sub_lab.nunique() == 2 and len(take) >= 2 * folds:
            break
    m = df.subject_id.isin(take).to_numpy()
    return df[m].reset_index(drop=True), X12[m]


def tokenize_all(tok, texts, max_length):
    enc = [tok(t, truncation=True, max_length=max_length,
               add_special_tokens=True)["input_ids"] for t in texts]
    trunc = sum(len(e) == max_length for e in enc)
    print(f"tokenised {len(enc)} transcripts | median {int(np.median([len(e) for e in enc]))} "
          f"tokens | {trunc} hit the {max_length}-token cap", flush=True)
    return enc


def make_batch(seqs, pad_id, device):
    """RIGHT padding.

    Phi3ForSequenceClassification pools the rightmost non-pad token, so with
    right padding each example's pooled position is its own true final token and
    predictions are batch-composition invariant. Right padding also keeps RoPE
    positions contiguous from 0, which left padding does not (HF derives
    position_ids from arange, not from the mask).
    """
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        att[i, : len(s)] = 1
    return ids.to(device), att.to(device)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def build_model(args, pad_id):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=2,
        quantization_config=bnb_cfg,
        dtype=torch.float16,                 # transformers 5.x name; torch_dtype is legacy
        device_map={"": 0},
        attn_implementation=args.attn,
    )
    # Without this the pooling head cannot find the final real token and errors
    # out on any batch larger than 1.
    model.config.pad_token_id = pad_id
    model.config.get_text_config().pad_token_id = pad_id
    model.config.use_cache = False

    # use_reentrant=False is the non-reentrant checkpointing path: it propagates
    # grads to the LoRA layers without the enable_input_require_grads hack.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False

    lora = LoraConfig(
        task_type="SEQ_CLS",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=TARGET_MODULES,
        modules_to_save=["score"],           # the randomly-init head must train
    )
    model = get_peft_model(model, lora)
    return model


def free():
    """Collect and hand the arena back to the driver.

    Callers must `del` their own references FIRST -- deleting a name inside this
    function would only drop the argument binding, leaving the caller's name
    still holding the tensors alive. Five sequential fine-tunes in one process
    is exactly the situation where that mistake shows up as an OOM on fold 3.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ----------------------------------------------------------------------------
# train / predict
# ----------------------------------------------------------------------------
def train_fold(model, seqs, y_tr, tr, args, pad_id, device, seed):
    """One fold's fine-tune. Class weights come from the TRAINING fold only.

    The loss is weighted BY HAND rather than through F.cross_entropy(weight=...).
    That is not a stylistic choice. `weight=` with the default mean reduction
    computes sum_i w_i l_i / sum_i w_i, so on a single-example micro-batch the
    weight cancels exactly and class weighting becomes a silent no-op -- which is
    precisely the regime gradient accumulation puts us in. Here each accumulation
    group is normalised by its own EXAMPLE COUNT, not by sum(w) and not by
    grad_accum, so (a) the weights survive, (b) the ragged final group of an
    epoch is scaled by its true size instead of a full group's, and (c) the
    gradient magnitude stays on the same scale as an unweighted mean, because
    balanced weights sum to n over the training fold by construction.
    """
    import bitsandbytes as bnb
    from transformers import get_linear_schedule_with_warmup

    counts = np.bincount(y_tr, minlength=2).astype(np.float64)
    cw = torch.tensor(counts.sum() / (2.0 * np.clip(counts, 1.0, None)),
                      dtype=torch.float32, device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in params)
    # Paged states live in CUDA unified memory, which is unreliable under the
    # Windows WDDM driver. Only ~25M params are trainable so the states are tiny
    # and paging is a convenience, not a requirement -- fall back with --optim.
    opt_cls = {"paged_adamw_8bit": bnb.optim.PagedAdamW8bit,
               "adamw_8bit": bnb.optim.AdamW8bit,
               "adamw": torch.optim.AdamW}[args.optim]
    opt = opt_cls(params, lr=args.lr, weight_decay=args.weight_decay)

    n = len(tr)
    starts = list(range(0, n, args.batch_size))          # micro-batch offsets
    micro_n = [min(args.batch_size, n - s) for s in starts]
    # examples in the accumulation group each micro-batch belongs to; the last
    # group of an epoch is usually short and must be normalised by its own size
    grp_n = [sum(micro_n[g : g + args.grad_accum])
             for g in range(0, len(starts), args.grad_accum)]

    per_epoch = len(grp_n)
    total_steps = per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    sched = get_linear_schedule_with_warmup(
        opt, int(args.warmup_frac * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    print(f"    trainable {n_train_p/1e6:.2f}M params | {n} examples | "
          f"{total_steps} optimiser steps | class weights "
          f"[{cw[0]:.3f}, {cw[1]:.3f}]", flush=True)

    gen = torch.Generator().manual_seed(seed)
    model.train()
    step, skipped, t0 = 0, 0, time.time()
    stop = False

    def _optimiser_step():
        """Returns True if the weights actually moved.

        GradScaler.step is a no-op when the unscaled grads contain inf/nan, and
        it signals that by shrinking the scale in update(). Stepping the LR
        schedule anyway would desynchronise it from the real step count and hide
        an overflow that ate part of the run, so the schedule advances only on a
        step that landed and the skips are counted and reported.
        """
        nonlocal step, skipped
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        took = scaler.get_scale() >= before
        if took:
            sched.step()
            step += 1
        else:
            skipped += 1
        opt.zero_grad(set_to_none=True)
        return took

    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=gen).tolist()
        opt.zero_grad(set_to_none=True)
        run_loss, run_k = 0.0, 0

        for m, i in enumerate(starts):
            sel = perm[i : i + args.batch_size]
            ids, att = make_batch([seqs[tr[j]] for j in sel], pad_id, device)
            lab = torch.tensor(y_tr[sel], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                logits = model(input_ids=ids, attention_mask=att).logits
            # weighted by hand -- see the docstring; weight= would cancel at bs=1
            per_ex = F.cross_entropy(logits.float(), lab, reduction="none") * cw[lab]
            loss = per_ex.sum() / grp_n[m // args.grad_accum]
            scaler.scale(loss).backward()
            run_loss += float(per_ex.detach().sum()); run_k += len(sel)

            last_of_group = (m + 1) % args.grad_accum == 0 or m == len(starts) - 1
            if last_of_group:
                _optimiser_step()
                if args.log_every and step % args.log_every == 0 and run_k:
                    print(f"      step {step}/{total_steps} loss {run_loss/run_k:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    run_loss, run_k = 0.0, 0
                if args.max_steps > 0 and step >= args.max_steps:
                    stop = True
                    break
        if stop:
            break

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"    trained {step} steps ({skipped} skipped on inf/nan) in "
          f"{time.time()-t0:.0f}s | peak VRAM {peak:.2f} GB", flush=True)
    del opt, scaler, params
    free()
    return model, {"steps": step, "skipped_steps": skipped, "peak_vram_gb": round(peak, 2)}


@torch.no_grad()
def predict(model, seqs, idx, args, pad_id, device):
    """P(ProbableAD) for the given rows. Batches are length-sorted purely to cut
    padding waste -- with right padding and last-real-token pooling the score of
    an example does not depend on what it is batched with."""
    was_training = model.training
    model.eval()
    out = np.zeros(len(idx), dtype=np.float64)
    order = np.argsort([len(seqs[i]) for i in idx])
    for i in range(0, len(order), args.eval_batch_size):
        sel = order[i : i + args.eval_batch_size]
        ids, att = make_batch([seqs[idx[j]] for j in sel], pad_id, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
            logits = model(input_ids=ids, attention_mask=att).logits
        out[sel] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    model.train(was_training)
    return out


def probe_fold(X12, tr, te, y_tr):
    """The frozen L12 baseline, refit on this exact fold.

    Identical estimator to evaluate.py's `logreg` (C=0.01, balanced,
    max_iter=2000) -- the one that produced 0.9135 -- so the paired difference
    is attributable to fine-tuning and not to a changed baseline.
    """
    sc = StandardScaler().fit(X12[tr])
    clf = LogisticRegression(max_iter=2000, C=0.01,
                             class_weight="balanced").fit(sc.transform(X12[tr]), y_tr)
    return (clf.decision_function(sc.transform(X12[tr])),
            clf.decision_function(sc.transform(X12[te])))


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------
def paired_delta_bootstrap(df, oof_a, oof_b, n_boot=2000, seed=0):
    """Subject-level paired bootstrap of AUROC(a) - AUROC(b).

    Five folds that share training data are not five independent observations,
    so the paired t-test over folds is anti-conservative and cannot carry the
    claim on its own. This resamples SUBJECTS -- the unit of independence -- and
    scores both arms on the IDENTICAL resample, so the shared difficulty of a
    draw cancels and what is left is the difference. Mirrors
    stacking_gbm.paired_bootstrap and fusion_test.py.

    Draws are spread across repeats (r = b % n_repeats) exactly as
    evaluate.subject_bootstrap_ci does, so the interval describes the same
    estimator as the reported point estimate.

    IMPORTANT: this interval covers test-set sampling noise ONLY. It says
    nothing about fine-tuning seed variance -- that would need repeated
    fine-tunes on identical folds, which is not what this script runs.

    p_boot is the achieved significance level: twice the smaller tail mass of
    the paired difference, floored at 1/n_draws.
    """
    y = df.label.to_numpy()
    subs = df.subject_id.unique()
    by_sub = {s: np.flatnonzero(df.subject_id.to_numpy() == s) for s in subs}
    n_repeats = oof_a.shape[0]
    rng = np.random.default_rng(seed)

    diffs = []
    for b in range(n_boot):
        pick = rng.choice(len(subs), len(subs), replace=True)
        idx = np.concatenate([by_sub[subs[p]] for p in pick])
        if len(set(y[idx])) < 2:
            continue
        r = b % n_repeats
        diffs.append(roc_auc_score(y[idx], oof_a[r][idx])
                     - roc_auc_score(y[idx], oof_b[r][idx]))

    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    tail = min((d <= 0).mean(), (d >= 0).mean())
    return {"delta_mean": float(d.mean()), "delta_lo": float(lo), "delta_hi": float(hi),
            "p_boot": float(max(2 * tail, 1.0 / max(len(d), 1))), "n_draws": int(len(d))}


TUNABLE = ("epochs", "lr", "lora_r", "lora_alpha", "lora_dropout", "weight_decay",
           "warmup_frac", "max_length", "batch_size", "grad_accum", "repeats",
           "folds", "optim", "fp16", "max_grad_norm", "model_id")


def record_run(args, smoke):
    """Append this invocation to results/qlora_runs.jsonl and flag reruns.

    This script has no hyperparameter search and no early stopping, so there is
    nothing to correct for WITHIN a run. The exposure is ACROSS runs: after a
    null there is real pressure to try another learning rate, and results/qlora.csv
    on its own cannot distinguish run 1 from run 4. Selecting the best of several
    configurations by their test AUROC is test-set selection no matter how few
    lines of code it takes -- exactly the failure mode this paper documents. The
    log makes the choice visible; it deliberately does not prevent it.
    """
    log = Path("results/qlora_runs.jsonl")
    cfg = {k: v for k, v in vars(args).items() if k in TUNABLE}
    prior = []
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                prior.append(json.loads(line))

    real_prior = [p for p in prior if not p.get("smoke")]
    differing = [p for p in real_prior
                 if {k: p["config"].get(k) for k in TUNABLE} != cfg]
    if not smoke and real_prior:
        print(f"\n  NOTE: {len(real_prior)} previous non-smoke run(s) logged in {log}.",
              flush=True)
        if differing:
            print("  WARNING: a previous run used DIFFERENT hyperparameters. Reporting", flush=True)
            print("  whichever run scored better is test-set selection. Report the", flush=True)
            print("  pre-registered configuration, or report all of them.", flush=True)

    with log.open("a") as fh:
        fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                             "smoke": smoke, "config": cfg,
                             "argv": sys.argv[1:]}) + "\n")
    Path(f"{args.out_prefix}_config.json").write_text(json.dumps(vars(args), indent=2))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="4-bit QLoRA fine-tune of Phi-3 vs the frozen L12 probe, same folds")
    ap.add_argument("--manifest", default="artifacts/manifest.csv")
    ap.add_argument("--l12", default="artifacts/L12_mean.npy")
    ap.add_argument("--l12-order", default="artifacts/L12_mean_order.csv")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3,
                    help="each repeat is 5 full fine-tunes. 3 by default: at 1 "
                         "the reported SD is fold-to-fold within a single "
                         "partition, and a single partition's baseline can sit "
                         "0.006 off the 5x5 mean by luck of the split")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="4 is ~3x faster than 1 at the same effective batch; "
                         "drop to 1 --grad-accum 8 if VRAM is tight")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--optim", choices=["paged_adamw_8bit", "adamw_8bit", "adamw"],
                    default="paged_adamw_8bit")
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke test: cap optimiser steps per fold (0 = no cap)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: shrink the cohort to ~N files, whole subjects only")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="subject-level bootstrap draws for the CIs")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate splits/data/probe/IO without touching the GPU")
    ap.add_argument("--out-prefix", default="results/qlora")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)

    # A smoke run is not the primary cohort and its AUROC must never reach the
    # paper, so it is mechanically prevented from landing on the real filenames
    # rather than merely discouraged. --limit in particular picks subjects with a
    # label-aware stopping rule (both classes must appear), so its cohort is not
    # the 477-file one and its numbers mean nothing. A --dry-run writes a
    # probe-only fold table, which would silently clobber a real run's CSV, so it
    # is redirected too.
    smoke = bool(args.limit or args.max_steps)
    if (smoke or args.dry_run) and args.out_prefix == "results/qlora":
        args.out_prefix = "results/qlora_smoke" if smoke else "results/qlora_dryrun"
        why = "SMOKE RUN (--limit/--max-steps)" if smoke else "DRY RUN"
        print(f"{why}: NOT the primary cohort/protocol.\n"
              f"  output redirected -> {args.out_prefix}*  (do not report these numbers)\n",
              flush=True)

    df, X12 = load_cohort(args.manifest, args.l12, args.l12_order)
    if args.limit:
        df, X12 = subsample_subjects(df, X12, args.limit, args.folds)
    y = df.label.to_numpy()
    groups = df.subject_id.to_numpy()
    N = len(df)
    print(f"cohort: {N} files, {df.subject_id.nunique()} subjects, "
          f"{int((y == 0).sum())} Control / {int(y.sum())} ProbableAD", flush=True)
    print(f"protocol: StratifiedGroupKFold({args.folds}) on subject_id x "
          f"{args.repeats} repeat(s), {args.epochs} epochs, no early stopping", flush=True)

    if not args.dry_run:
        record_run(args, smoke)
    print("", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_id, seqs = 0, None
    if not args.dry_run:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token          # Phi-3 has no dedicated pad token
        tok.padding_side = "right"
        pad_id = tok.pad_token_id
        seqs = tokenize_all(tok, df.text.fillna("").tolist(), args.max_length)

    oof = {k: np.full((args.repeats, N), np.nan) for k in ("qlora", "probe")}
    # per-fold training thresholds; evaluate.py averages these within a repeat and
    # applies the average to the pooled OOF vector, and so do we (see summary)
    thr_folds = {k: [[] for _ in range(args.repeats)] for k in ("qlora", "probe")}
    fold_rows = []

    for r in range(args.repeats):
        cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=r)
        for k, (tr, te) in enumerate(cv.split(np.zeros(N), y, groups)):
            assert not set(groups[tr]) & set(groups[te]), "subject leaked across the split"
            ytr, yte = y[tr], y[te]
            t0 = time.time()
            print(f"repeat {r} fold {k}: {len(tr)} train / {len(te)} test files, "
                  f"{len(set(groups[te]))} test subjects", flush=True)

            row = {"repeat": r, "fold": k, "n_train": len(tr), "n_test": len(te),
                   "n_test_subjects": len(set(groups[te]))}

            # --- frozen L12 probe, this exact fold -----------------------------
            s_tr, s_te = probe_fold(X12, tr, te, ytr)
            thr_p = youden_threshold(ytr, s_tr)
            thr_folds["probe"][r].append(thr_p)
            oof["probe"][r, te] = s_te
            row["auroc_probe"] = float(roc_auc_score(yte, s_te))
            row["auroc_probe_train"] = float(roc_auc_score(ytr, s_tr))
            row["thr_probe"] = float(thr_p)
            for m, v in metrics_at(yte, s_te, thr_p).items():
                if m in ("acc", "sens", "spec"):
                    row[f"{m}_probe"] = round(float(v), 4)

            # --- QLoRA fine-tune, same fold ------------------------------------
            if args.dry_run:
                print(f"  [dry-run] skipping fine-tune | "
                      f"probe fold AUROC {row['auroc_probe']:.4f}", flush=True)
                fold_rows.append(row)
                continue

            torch.manual_seed(1000 * r + k)
            model = build_model(args, pad_id)
            model, diag = train_fold(model, seqs, ytr, tr, args, pad_id, device,
                                     seed=1000 * r + k)
            q_tr = predict(model, seqs, tr, args, pad_id, device)
            q_te = predict(model, seqs, te, args, pad_id, device)
            del model                      # drop the caller's reference, THEN collect
            free()

            thr_q = youden_threshold(ytr, q_tr)
            thr_folds["qlora"][r].append(thr_q)
            oof["qlora"][r, te] = q_te
            row["auroc_qlora"] = float(roc_auc_score(yte, q_te))
            # the ONLY thing separating "fine-tuning does not help" from "the
            # fine-tune silently did nothing": if this is not ~1.0 the adapters
            # never learned and the null below is uninterpretable
            row["auroc_qlora_train"] = float(roc_auc_score(ytr, q_tr))
            row["thr_qlora"] = float(thr_q)
            for m, v in metrics_at(yte, q_te, thr_q).items():
                if m in ("acc", "sens", "spec"):
                    row[f"{m}_qlora"] = round(float(v), 4)
            row["delta"] = row["auroc_qlora"] - row["auroc_probe"]
            row.update(diag)
            row["secs"] = round(time.time() - t0, 1)
            fold_rows.append(row)

            print(f"  fold AUROC  qlora {row['auroc_qlora']:.4f}  "
                  f"probe {row['auroc_probe']:.4f}  delta {row['delta']:+.4f}  "
                  f"| train AUROC qlora {row['auroc_qlora_train']:.4f} "
                  f"probe {row['auroc_probe_train']:.4f}  ({row['secs']:.0f}s)\n",
                  flush=True)

    folds_df = pd.DataFrame(fold_rows)
    for k_ in TUNABLE:                     # every fold row carries its own config
        folds_df[k_] = vars(args)[k_]
    folds_df.to_csv(f"{args.out_prefix}_folds.csv", index=False)

    if args.dry_run:
        print(f"\n[dry-run] probe mean per-fold AUROC "
              f"{folds_df.auroc_probe.mean():.4f} +/- {folds_df.auroc_probe.std():.4f}")
        print(f"[dry-run] splits, cohort, probe and IO all validated. "
              f"saved -> {args.out_prefix}_folds.csv")
        return 0

    np.save(f"{args.out_prefix}_oof.npy",
            np.stack([oof["qlora"], oof["probe"]]))   # [model, repeat, N], (qlora, probe)

    # ---- summary -----------------------------------------------------------
    # Estimator, stated once: AUROC is POOLED out-of-fold within a repeat, then
    # averaged over repeats, and acc/sens/spec apply the MEAN of that repeat's
    # fold training thresholds to the same pooled vector. That is exactly what
    # evaluate.py:summarize does, which is what makes these rows comparable to
    # results/main_L12_mean.csv and to the 0.9135 reference. The mean of
    # per-fold AUROCs is kept as a separate column because it is a different
    # estimator (~+0.003 on this cohort) and must never be quoted against a
    # number computed the other way.
    from scipy import stats

    def sd(a):
        """ddof=1 everywhere, and NaN rather than a fake 0.0 for a single value."""
        return round(float(np.std(a, ddof=1)), 4) if len(a) > 1 else float("nan")

    rows = []
    label = {"qlora": "Phi-3 QLoRA 4-bit fine-tune", "probe": "Phi-3 L12 frozen + logreg"}
    thr_mean = {k: np.array([float(np.mean(thr_folds[k][r])) for r in range(args.repeats)])
                for k in ("qlora", "probe")}
    ci = {}
    for key in ("qlora", "probe"):
        per_repeat = [metrics_at(y, oof[key][r], thr_mean[key][r])
                      for r in range(args.repeats)]
        pooled = np.array([m["auroc"] for m in per_repeat])
        per_fold = folds_df[f"auroc_{key}"].to_numpy()
        ci[key] = subject_bootstrap_ci(df, oof[key], thr_mean[key], n_boot=args.n_boot)
        rows.append({
            "model": label[key],
            "auroc_pooled": round(float(pooled.mean()), 4),      # <-- headline
            "auroc_pooled_sd": sd(pooled),
            "auroc_ci_lo": round(ci[key]["auroc_ci"][0], 4),
            "auroc_ci_hi": round(ci[key]["auroc_ci"][1], 4),
            "auroc_fold_mean": round(float(per_fold.mean()), 4),  # different estimator
            "auroc_fold_sd": sd(per_fold),
            "auroc_train": round(float(folds_df[f"auroc_{key}_train"].mean()), 4),
            "acc": round(float(np.mean([m["acc"] for m in per_repeat])), 4),
            "sens": round(float(np.mean([m["sens"] for m in per_repeat])), 4),
            "spec": round(float(np.mean([m["spec"] for m in per_repeat])), 4),
        })

    # delta: pooled per repeat (the headline estimator) and per fold (for the t-test)
    d_pooled = np.array([roc_auc_score(y, oof["qlora"][r]) - roc_auc_score(y, oof["probe"][r])
                         for r in range(args.repeats)])
    d = folds_df["delta"].to_numpy()
    n_f = len(d)
    if n_f > 1:
        t_stat, p_val = stats.ttest_rel(folds_df.auroc_qlora, folds_df.auroc_probe)
    else:
        t_stat, p_val = float("nan"), float("nan")
    boot = paired_delta_bootstrap(df, oof["qlora"], oof["probe"], n_boot=args.n_boot)
    rows.append({
        "model": "delta (qlora - frozen), paired",
        "auroc_pooled": round(float(d_pooled.mean()), 4),
        "auroc_pooled_sd": (round(float(np.std(d_pooled, ddof=1)), 4)
                            if args.repeats > 1 else float("nan")),
        "auroc_ci_lo": round(boot["delta_lo"], 4),
        "auroc_ci_hi": round(boot["delta_hi"], 4),
        "auroc_fold_mean": round(float(d.mean()), 4),
        "auroc_fold_sd": round(float(d.std(ddof=1)), 4) if n_f > 1 else float("nan"),
        "auroc_train": "", "acc": "", "sens": "", "spec": "",
    })

    # minimum detectable effect for the paired-by-fold t-test, so a null can be
    # phrased as "not better by more than X" instead of "no difference"
    mde = (float(stats.t.ppf(0.975, n_f - 1)) * float(d.std(ddof=1)) / math.sqrt(n_f)
           if n_f > 1 else float("nan"))

    print("\n" + "=" * 78)
    print(f"Control vs ProbableAD, subject-disjoint {args.folds}-fold x {args.repeats} repeat(s)")
    print("AUROC below is POOLED out-of-fold per repeat (evaluate.py estimator).\n")
    for row in rows[:2]:
        print(f"  {row['model']:28s} AUROC {row['auroc_pooled']:.4f} "
              f"+/- {row['auroc_pooled_sd']:.4f} "
              f"[{row['auroc_ci_lo']:.4f}, {row['auroc_ci_hi']:.4f}]  "
              f"acc {row['acc']:.3f}  sens {row['sens']:.3f}  spec {row['spec']:.3f}")
        print(f"  {'':28s} (mean-of-per-fold AUROC {row['auroc_fold_mean']:.4f} "
              f"+/- {row['auroc_fold_sd']:.4f} -- NOT the same estimator; "
              f"train AUROC {row['auroc_train']:.4f})")
    print(f"\n  paired delta (pooled, per repeat): {d_pooled.mean():+.4f}")
    print(f"  paired delta (per fold): {d.mean():+.4f} "
          f"+/- {d.std(ddof=1) if n_f > 1 else float('nan'):.4f}  "
          f"(qlora wins {int((d > 0).sum())}/{n_f} folds)")
    print(f"  subject bootstrap on the delta: {boot['delta_mean']:+.4f} "
          f"[{boot['delta_lo']:+.4f}, {boot['delta_hi']:+.4f}], "
          f"p_boot={boot['p_boot']:.4f} ({boot['n_draws']} draws)")
    print(f"  paired t-test over folds: t={t_stat:+.3f}, p={p_val:.4f}")
    print(f"\n  READING THIS: the {n_f} folds share training data (and repeats re-partition")
    print("  the SAME 477 files), so the t-test is anti-conservative -- treat it as a")
    print("  sign check, not proof. The subject bootstrap is the interval to quote; it")
    print("  covers test-set sampling noise only, NOT fine-tuning seed variance.")
    print(f"  Minimum detectable |delta| is ~{mde:.3f} AUROC and that figure itself")
    print("  assumes independent folds, so it is a floor. A null here means 'not")
    print(f"  better by more than ~{mde:.3f}', never 'no difference'.")
    tr_q = float(folds_df["auroc_qlora_train"].mean())
    if tr_q < 0.95:
        print(f"\n  !! train-fold AUROC for QLoRA is only {tr_q:.4f}. The fine-tune may")
        print("  !! have failed (dead adapters / LR too low / overflow). Check the")
        print("  !! skipped_steps column before reporting this as a null result.")
    if int(folds_df["skipped_steps"].sum()) > 0:
        print(f"\n  !! {int(folds_df['skipped_steps'].sum())} optimiser steps were skipped "
              "on inf/nan across folds.")
    print(f"\n  reference (5x5, pooled OOF, same cohort): frozen L12 logreg 0.9135, "
          f"late fusion 0.9246")
    print("=" * 78)

    pd.DataFrame(rows).to_csv(f"{args.out_prefix}.csv", index=False)
    print(f"\nsaved -> {args.out_prefix}.csv")
    print(f"saved -> {args.out_prefix}_folds.csv")
    print(f"saved -> {args.out_prefix}_config.json")
    print(f"saved -> {args.out_prefix}_oof.npy   [2, {args.repeats}, {N}] = (qlora, probe)")
    print("  NOTE: row 0 (qlora) is a SOFTMAX PROBABILITY in [0, 1]; row 1 (probe) is")
    print("  an unbounded decision_function. They are NOT on a common scale -- any")
    print("  fusion consumer must z-score each per training fold before combining.")
    print("\n  acc/sens/spec for the QLoRA row rest on a Youden threshold fit to the")
    print("  fine-tuned model's OWN training scores, which are near-separable "
          f"(train AUROC {tr_q:.4f}).")
    print("  That threshold is train-fold-only and therefore protocol-legal, but it")
    print("  sits in an arbitrary gap. Compare the two arms on AUROC, not on the")
    print("  operating point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
