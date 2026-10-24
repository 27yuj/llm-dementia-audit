"""
Per-utterance Phi-3 hidden states, extracted IN NARRATIVE CONTEXT.

Mean pooling a whole transcript collapses the narrative: a speaker who starts
coherently and then derails ends up in the same place as one who was mildly
vague throughout. To model the TRAJECTORY we need one vector per utterance --
but an utterance encoded in isolation loses exactly the context that makes
derailment visible.

So: ONE forward pass over the full transcript (identical input to the one
src/extract_activations.py already used), then mean-pool the token positions
belonging to each *PAR utterance. Every utterance vector is therefore
conditioned on everything the speaker said before it, and the whole extraction
costs the same as the existing per-transcript pass.

Token->utterance assignment uses the fast tokenizer's character offset mapping
against the utterance character spans in the joined text. That is exact rather
than heuristic: the transcript fed to the model is literally ' '.join(utterances),
and we assert that it equals build_manifest.extract_par_text() for every file,
so these embeddings describe the same string the cached activations do.

Output: artifacts/utterance_embeddings.npz
  emb        float32 [total_utterances, hidden]   layer-`--layer` mean over the
                                                  utterance's token positions
  counts     int32   [n_files]                    utterances retained per file
  utt_tokens int32   [total_utterances]           tokens pooled per utterance
  file_ids   str     [n_files]                    row order of `counts`
Rows of `emb` are ordered file-by-file in manifest order; the span for file i is
emb[counts[:i].sum() : counts[:i+1].sum()].

A token-count-weighted mean of a file's utterance vectors reproduces the cached
artifacts/L12_mean.npy row to cosine ~0.999 under --quant cpu (same dtype, same
input). Not to 1.0: the cached row averages over EVERY token position including
BOS, while tokens that belong to no utterance -- BOS among them -- are assigned
-1 here and pooled into nothing. The script prints that cosine as a live
fidelity check, because the mean-pool control arm in src/temporal_probe.py has
to reproduce the established 0.9135 baseline and 4-bit quantisation perturbs
hidden states enough to put that at risk. Hence --quant cpu is the DEFAULT: it
is bf16 on CPU, the exact configuration behind artifacts/activations.pt, it
costs ~40 min of wall clock, and it leaves the (shared, single) GPU free.
"""

import argparse
import os
import sys
import time

# This machine's antivirus intercepts TLS with its own root CA, which lives in the
# Windows trust store but not in certifi's bundle. Use the OS store so HuggingFace
# downloads work with certificate verification still enabled.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pylangacq  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from build_manifest import (  # noqa: E402
    DROP_TOKENS,
    UNINTELLIGIBLE,
    extract_par_text,
)


# ----------------------------------------------------------------------------
# utterance segmentation -- the same cleaning rules build_manifest.py applied
# ----------------------------------------------------------------------------
def par_utterances(cha_path: Path, keep_fillers: bool) -> list[str]:
    """Split the participant's speech into cleaned utterances.

    Token filtering is character-for-character the rule set in
    build_manifest.extract_par_text; the only difference is that we stop at each
    utterance boundary instead of concatenating everything. main() asserts that
    ' '.join(result) == extract_par_text(...), which is what makes that claim
    checkable rather than aspirational.
    """
    chat = pylangacq.read_chat(str(cha_path))
    utterances = []
    for utt in chat.utterances():
        if utt.participant != "PAR":
            continue
        words = []
        for token in utt.tokens:
            w = (token.word or "").strip()
            if not w or w in DROP_TOKENS:
                continue
            if w.lower() in UNINTELLIGIBLE:
                continue
            # &-um / &=laughs style codes: keep the filled pause, drop the event code.
            if w.startswith("&"):
                if not keep_fillers:
                    continue
                w = w.lstrip("&-=+")
                if not w:
                    continue
            # fallin(g) -> falling ; the parens mark an unrealised sound.
            w = w.replace("(", "").replace(")", "")
            if w:
                words.append(w)
        if words:
            utterances.append(" ".join(words))
    return utterances


def utterance_char_spans(utterances: list[str]) -> tuple[str, np.ndarray, np.ndarray]:
    """Joined transcript plus the [start, end) character span of each utterance."""
    starts, ends, pos = [], [], 0
    for i, u in enumerate(utterances):
        if i:
            pos += 1                      # the single joining space
        starts.append(pos)
        pos += len(u)
        ends.append(pos)
    return " ".join(utterances), np.array(starts), np.array(ends)


def assign_tokens(offsets, utt_ends) -> np.ndarray:
    """Map each token to an utterance index via its character start offset.

    Utterances are separated by exactly one space, and the tokenizer never emits
    a token spanning that space plus text on both sides, so the token's start
    offset alone identifies the utterance. searchsorted on the utterance end
    offsets sends a token that begins on the separator to the FOLLOWING
    utterance, which is where its text actually lives (SentencePiece attaches
    the leading space to the next word).

    Returns -1 for special tokens, which carry an empty (0, 0) offset.
    """
    starts = np.array([o[0] for o in offsets])
    lengths = np.array([o[1] - o[0] for o in offsets])
    owner = np.searchsorted(utt_ends, starts, side="right")
    owner = np.clip(owner, 0, len(utt_ends) - 1)
    owner[lengths <= 0] = -1
    return owner


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def load_model(model_id, quant):
    cuda = torch.cuda.is_available()
    kwargs = {"low_cpu_mem_usage": True, "attn_implementation": "eager"}
    if quant == "4bit" and cuda:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = {"": 0}
    elif quant == "fp16" and cuda:
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = {"": 0}
    else:
        # bfloat16 on cpu == the exact configuration that produced
        # artifacts/activations.pt, so utterance means are bit-comparable to it
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "cpu"

    print(f"loading {model_id} ({quant}) ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifest.csv")
    ap.add_argument("--out", default="artifacts/utterance_embeddings.npz")
    ap.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--layer", type=int, default=12,
                    help="hidden_states index; 0 = embeddings, matches activations.pt")
    ap.add_argument("--quant", choices=["fp16", "4bit", "cpu"], default="cpu",
                    help="cpu (default) is bf16 and matches the configuration "
                         "behind artifacts/activations.pt, so the mean-pool "
                         "control still reproduces 0.9135; 4bit fits an 8GB "
                         "card but perturbs the hidden states; fp16 needs ~7.6GB")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--keep-fillers", action="store_true", default=True)
    ap.add_argument("--drop-fillers", dest="keep_fillers", action="store_false")
    ap.add_argument("--reference", default="artifacts/L12_mean.npy",
                    help="cached transcript-level means for the fidelity check")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N transcripts")
    ap.add_argument("--dry-run", action="store_true",
                    help="tokenise and assign spans only, no model load")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} | cuda={cuda}", flush=True)
    if cuda and not args.dry_run:
        print(f"device: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} transcripts | layer {args.layer}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if not tok.is_fast:
        raise SystemExit("need a fast tokenizer for character offset mapping")

    # artifacts/L12_mean.npy is the layer-12 token-mean of the cached bf16 CPU pass,
    # row-aligned with L12_mean_order.csv (identical to manifest order).
    reference = None
    order_path = Path(args.reference).with_name(Path(args.reference).stem + "_order.csv")
    if args.layer == 12 and Path(args.reference).exists():
        ref_order = (pd.read_csv(order_path).file_id if order_path.exists()
                     else pd.read_csv(args.manifest).file_id)
        ref_mat = np.load(args.reference)
        if len(ref_order) == len(ref_mat):
            reference = dict(zip(ref_order.astype(str).tolist(), ref_mat))

    model = None if args.dry_run else load_model(args.model_id, args.quant)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.npz")

    # Resume support: a cpu run takes ~40 min and shouldn't be lost to a crash.
    emb_chunks, counts, utt_tokens, file_ids = [], [], [], []
    if partial_path.exists() and not args.dry_run:
        ck = np.load(partial_path, allow_pickle=False)
        emb_chunks = [ck["emb"]]
        counts = list(ck["counts"])
        utt_tokens = list(ck["utt_tokens"])
        file_ids = [str(f) for f in ck["file_ids"]]
        print(f"resuming from checkpoint: {len(file_ids)} files already encoded",
              flush=True)

    done = set(file_ids)
    remaining = df[~df.file_id.astype(str).isin(done)]
    n_seg_mismatch, n_truncated, cos_checks = 0, 0, []
    t_start = time.time()

    for i, row in enumerate(remaining.itertuples(), start=1):
        cha_path = Path(row.path)
        utterances = par_utterances(cha_path, args.keep_fillers)
        text, u_starts, u_ends = utterance_char_spans(utterances)

        # the segmentation must rebuild exactly the string the manifest cached
        if text != extract_par_text(cha_path, args.keep_fillers):
            n_seg_mismatch += 1
            print(f"  [warn] {row.file_id}: segmentation != manifest text", flush=True)

        enc = tok(text, return_offsets_mapping=True, truncation=True,
                  max_length=args.max_length, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        owner = assign_tokens(offsets, u_ends)
        n_tok = len(offsets)
        if n_tok >= args.max_length:
            n_truncated += 1

        n_kept = 0
        if args.dry_run:
            kept = [int((owner == u).sum()) for u in range(len(utterances))]
            kept = [k for k in kept if k > 0]
            n_kept = len(kept)
            counts.append(n_kept)
            utt_tokens.extend(kept)
            file_ids.append(str(row.file_id))
        else:
            inputs = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            h = out.hidden_states[args.layer][0].float().cpu().numpy()  # [seq, H]
            del out

            vecs, toks = [], []
            for u in range(len(utterances)):
                idx = np.flatnonzero(owner == u)
                if idx.size == 0:
                    continue          # utterance fell outside the truncation window
                vecs.append(h[idx].mean(0))
                toks.append(int(idx.size))
            if not vecs:
                print(f"  [warn] {row.file_id}: no utterances retained, skipping",
                      flush=True)
                continue

            V = np.stack(vecs).astype(np.float32)
            n_kept = len(vecs)
            emb_chunks.append(V)
            counts.append(n_kept)
            utt_tokens.extend(toks)
            file_ids.append(str(row.file_id))

            if reference is not None and str(row.file_id) in reference:
                w = np.array(toks, dtype=np.float64)
                pooled = (V.astype(np.float64) * w[:, None]).sum(0) / w.sum()
                ref = reference[str(row.file_id)].astype(np.float64)
                cos_checks.append(float(
                    pooled @ ref / (np.linalg.norm(pooled) * np.linalg.norm(ref) + 1e-12)))

        if i % 25 == 0 or i == len(remaining):
            rate = (time.time() - t_start) / i
            print(f"  [{i}/{len(remaining)}] {row.file_id} "
                  f"{n_kept} utts / {n_tok} tok  {rate:.2f}s/file  "
                  f"ETA {rate * (len(remaining) - i) / 60:.1f}m", flush=True)

        if not args.dry_run and i % args.checkpoint_every == 0:
            np.savez(partial_path, emb=np.concatenate(emb_chunks),
                     counts=np.array(counts, dtype=np.int32),
                     utt_tokens=np.array(utt_tokens, dtype=np.int32),
                     file_ids=np.array(file_ids))

    counts_a = np.array(counts, dtype=np.int32)
    print(f"\nutterances per transcript: median {np.median(counts_a):.0f}  "
          f"mean {counts_a.mean():.1f}  max {counts_a.max()}  "
          f"total {counts_a.sum()}")
    if n_truncated:
        print(f"{n_truncated} transcripts hit --max-length {args.max_length}")
    print(f"segmentation mismatches vs manifest text: {n_seg_mismatch}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    if cos_checks:
        c = np.array(cos_checks)
        print(f"\nfidelity vs {args.reference} (token-weighted utterance mean): "
              f"cosine median {np.median(c):.5f}  min {c.min():.5f}")
        if np.median(c) < 0.99:
            print("[warn] hidden states drifted from the cached bf16 activations; "
                  "rerun with --quant cpu if the mean-pool control arm has to "
                  "reproduce the published 0.9135 baseline exactly")

    np.savez(out_path, emb=np.concatenate(emb_chunks),
             counts=counts_a, utt_tokens=np.array(utt_tokens, dtype=np.int32),
             file_ids=np.array(file_ids), layer=np.int32(args.layer),
             max_length=np.int32(args.max_length),
             model_id=np.array(args.model_id), quant=np.array(args.quant))
    partial_path.unlink(missing_ok=True)

    total = int(counts_a.sum())
    print(f"\nsaved [{total}, {emb_chunks[0].shape[1]}] utterance vectors "
          f"for {len(file_ids)} files -> {out_path}")
    print(f"wall time {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
