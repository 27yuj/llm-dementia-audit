# llm-dementia-audit

Probing frozen large language model representations for signs of cognitive
impairment in spontaneous speech, evaluated on the DementiaBank Pitt corpus.

## What the models are

Transcripts are encoded once by a **frozen** `Phi-3-mini-4k-instruct` (3.8B) —
its weights are never updated; the pipeline only reads hidden states out of it.
A single forward pass caches all 33 layers under two pooling schemes
(last-token and mean over tokens), so any layer/pooling combination can be
evaluated offline.

Classifiers placed on those representations:

| Model | File |
|---|---|
| Contrastive PCA (unsupervised direction finder) | `src/models.py` |
| Logistic regression on standardised features | `src/evaluate.py` |
| Low-rank non-linear probe (16-unit GELU bottleneck) | `src/models.py` |
| 1D-CNN / GRU / Transformer over per-utterance vectors | `src/temporal_probe.py` |
| Stacking ensembles (averaging, logistic, LightGBM, XGBoost, RF) | `src/stacking_gbm.py` |
| 4-bit QLoRA fine-tuning with a classification head | `src/qlora_finetune.py` |

Non-neural reference models — TF-IDF, disfluency rates, demographics — are in
`src/confounds.py` and `src/disfluency.py`.

## Evaluation

The corpus is longitudinal: the same patients were recorded repeatedly, so
transcripts are **not** independent. Every split therefore groups by
`subject_id` (`StratifiedGroupKFold`, 5 folds × 5 repeats) so no speaker
appears in both training and testing. Scalers, PCA bases, TF-IDF vocabularies,
decision thresholds and hyperparameters are fit inside the training fold only.
Confidence intervals resample subjects rather than transcripts.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Frozen extraction runs on CPU in bfloat16 (~34 min, no GPU). QLoRA and
surprisal scoring use 4-bit quantization and fit in 8 GB of VRAM. For CUDA,
install `torch` from the `cu128` index — `pip install --upgrade torch` will not
switch builds, since the CPU and CUDA wheels share a version number.

## Running

```bash
python src/build_manifest.py          # parse transcripts -> artifacts/manifest.csv
python src/extract_activations.py     # frozen hidden states, all 33 layers
python src/evaluate.py --mode sweep   # layer/pooling comparison
python src/evaluate.py --mode main    # cPCA vs logistic vs probe
python src/temporal_probe.py          # sequence models over utterances
python src/stacking_gbm.py            # fusion ensembles
python src/qlora_finetune.py          # fine-tuning vs frozen, same folds
```

Most scripts take `--cohort`, `--layer`, `--pooling` and `--folds`; run any with
`--help`.

## Data

**No data is included in this repository, and none may be added.** The
DementiaBank Pitt corpus is distributed by TalkBank under a membership
agreement that prohibits redistribution. Transcripts, audio, and anything
derived from them (activations, per-visit records) are excluded by
`.gitignore` and must stay local.

To run the pipeline, request access at
[dementia.talkbank.org](https://dementia.talkbank.org/) and place `Pitt.cha/`
at the repository root.

Before committing anything new, check it carries no transcript text or
participant identifiers:

```bash
git diff --cached --name-only | xargs -I{} sh -c 'head -1 "{}" | grep -qiE "text|subject_id|mmse" && echo "REVIEW: {}"'
```

## License

MIT (see `LICENSE`). The corpus is not covered by this license.
