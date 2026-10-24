"""
Extract frozen Phi-3 hidden states for every transcript in the manifest.

Runs on CPU in bfloat16 (no CUDA required). One forward pass per transcript gives
us every layer at once, so we cache ALL layers under two pooling schemes:

    last  -- hidden state of the final token   (the original Colab choice)
    mean  -- attention-mask-weighted mean over tokens

Caching both costs nothing extra at extraction time and lets the layer/pooling
sweep run offline in seconds. Output tensors are float32, shaped
[n_files, n_layers, hidden_dim], aligned row-for-row with the manifest.
"""

import argparse
import os
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
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifest.csv")
    ap.add_argument("--out", default="artifacts/activations.pt")
    ap.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N transcripts")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__} | threads={torch.get_num_threads()}")

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} transcripts to encode")

    print(f"loading {args.model_id} (bfloat16, cpu) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.pt")

    # Resume support: CPU passes are slow enough that a crash shouldn't cost the run.
    done_last, done_mean, done_ids, done_tok = [], [], [], []
    if partial_path.exists():
        ck = torch.load(partial_path)
        done_last, done_mean = list(ck["last"]), list(ck["mean"])
        done_ids, done_tok = list(ck["file_ids"]), list(ck["n_tokens"])
        print(f"resuming from checkpoint: {len(done_ids)} already encoded")

    remaining = df[~df.file_id.astype(str).isin(set(map(str, done_ids)))]
    t_start = time.time()

    for i, row in enumerate(remaining.itertuples(), start=1):
        inputs = tokenizer(
            row.text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # tuple of [1, seq, hidden] -- one per layer, embeddings first
        stack = torch.stack(out.hidden_states, dim=0)[:, 0].float()  # [L, seq, H]
        mask = inputs["attention_mask"][0].float()                   # [seq]

        done_last.append(stack[:, -1, :].clone())
        done_mean.append(((stack * mask[None, :, None]).sum(1) / mask.sum()).clone())
        done_ids.append(str(row.file_id))
        done_tok.append(int(mask.sum().item()))

        if i % 5 == 0 or i == len(remaining):
            rate = (time.time() - t_start) / i
            eta = rate * (len(remaining) - i) / 60
            print(f"  [{i}/{len(remaining)}] {row.file_id} "
                  f"({done_tok[-1]} tok) {rate:.1f}s/file  ETA {eta:.0f}m", flush=True)

        if i % args.checkpoint_every == 0:
            torch.save(
                {"last": torch.stack(done_last), "mean": torch.stack(done_mean),
                 "file_ids": done_ids, "n_tokens": done_tok},
                partial_path,
            )

    payload = {
        "last": torch.stack(done_last),   # [N, L, H]
        "mean": torch.stack(done_mean),
        "file_ids": done_ids,
        "n_tokens": done_tok,
        "model_id": args.model_id,
        "dtype": "bfloat16",
        "max_length": args.max_length,
    }
    torch.save(payload, out_path)
    partial_path.unlink(missing_ok=True)

    print(f"\nsaved {payload['last'].shape} (last) and {payload['mean'].shape} (mean) "
          f"-> {out_path}")
    print(f"total wall time: {(time.time() - t_start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
