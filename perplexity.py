"""
Per-token surprisal of each transcript under frozen Phi-3.

Hypothesis under test: AD speech is less predictable (broken syntax -> higher
perplexity). The opposite is also plausible and worth stating up front --
impaired narratives are shorter, more repetitive, and lean on high-frequency
deictic words ("this", "here", "it"), all of which a language model finds MORE
predictable. Repetition in particular drives perplexity down, not up.

Records the full surprisal distribution, not just the mean, so a shift in
variance or in the tail can be detected even if the mean does not move.
"""

import argparse
import os
import time
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifest.csv")
    ap.add_argument("--out", default="artifacts/perplexity.csv")
    ap.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--quant", choices=["4bit", "fp16", "cpu"], default="4bit",
                    help="4bit fits an 8GB card comfortably; fp16 needs ~7.6GB")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} | cuda={cuda}")
    if cuda:
        print(f"device: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    kwargs = {"low_cpu_mem_usage": True, "attn_implementation": "eager"}
    if args.quant == "4bit" and cuda:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = {"": 0}
    elif args.quant == "fp16" and cuda:
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "cpu"

    print(f"loading {args.model_id} ({args.quant}) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s")

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)

    rows, t_start = [], time.time()
    for i, r in enumerate(df.itertuples(), start=1):
        enc = tok(r.text, return_tensors="pt", truncation=True,
                  max_length=args.max_length)
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()

        # surprisal of each token given its prefix
        nll = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")
        nll = nll.cpu()

        rows.append({
            "file_id": str(r.file_id),
            "n_tokens": int(ids.shape[1]),
            "nll_mean": float(nll.mean()),                 # = log perplexity
            "perplexity": float(torch.exp(nll.mean())),
            "nll_median": float(nll.median()),
            "nll_std": float(nll.std()),
            "nll_p90": float(torch.quantile(nll, 0.90)),
            "frac_surprising": float((nll > 5.0).float().mean()),
        })

        if i % 25 == 0 or i == len(df):
            rate = (time.time() - t_start) / i
            print(f"  [{i}/{len(df)}] {rate:.2f}s/file  "
                  f"ETA {rate * (len(df) - i) / 60:.1f}m", flush=True)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nsaved {len(out)} rows -> {args.out}")
    print(f"wall time {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    raise SystemExit(main())
