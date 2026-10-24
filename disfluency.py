"""
Count CHAT disfluency annotations per transcript.

Written to a SEPARATE csv (not the manifest) so it can run while activation
extraction is still reading manifest.csv.

Note on provenance: these markers differ in how recoverable they are from raw
audio. Filled pauses, repetitions and unintelligible stretches are within reach
of an ASR front-end; retracing/reformulation codes ([//], [///]) are human
annotator judgements. A deployed screening tool would not have the latter, so
the two families are reported separately.
"""

import re
from pathlib import Path

import pandas as pd

PAR_LINE = re.compile(r"^\*PAR:\t(.*)$", re.MULTILINE)

# ASR-recoverable
ASR_MARKERS = {
    "filled_pause": re.compile(r"&-[a-z]+"),
    "repetition": re.compile(r"\[/\]"),
    "pause_short": re.compile(r"\(\.\)"),
    "pause_med": re.compile(r"\(\.\.\)"),
    "pause_long": re.compile(r"\(\.\.\.\)"),
    "trailing_off": re.compile(r"\+\.\.\."),
    "unintelligible": re.compile(r"\bxxx\b"),
    "paralinguistic": re.compile(r"&=[a-z:+]+"),
}
# annotator-derived
ANNOT_MARKERS = {
    "retracing": re.compile(r"\[//\]"),
    "reformulation": re.compile(r"\[///\]"),
    "error_code": re.compile(r"\[\*[^\]]*\]"),
}
ALL_MARKERS = {**ASR_MARKERS, **ANNOT_MARKERS}


def count_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    par = "\n".join(PAR_LINE.findall(text))
    # crude word count on the PAR side, for rate normalisation
    stripped = re.sub(r"[\d_]+\s*$", "", par, flags=re.MULTILINE)
    n_words = max(len(stripped.split()), 1)
    counts = {name: len(rx.findall(par)) for name, rx in ALL_MARKERS.items()}
    rates = {f"{k}_per100": 100.0 * v / n_words for k, v in counts.items()}
    return {**counts, **rates, "par_tokens": n_words}


def main():
    manifest = pd.read_csv("artifacts/manifest.csv")
    rows = []
    for r in manifest.itertuples():
        rows.append({"file_id": str(r.file_id), **count_file(Path(r.path))})
    out = pd.DataFrame(rows)
    out.to_csv("artifacts/disfluency.csv", index=False)
    print(f"wrote {len(out)} rows -> artifacts/disfluency.csv")

    merged = manifest.merge(out, on="file_id")
    merged = merged[(merged.group == "Control") | (merged.dx == "ProbableAD")]
    merged["label"] = (merged.dx == "ProbableAD").astype(int)
    cols = [f"{k}_per100" for k in ALL_MARKERS]
    print("\nrate per 100 words, Control vs ProbableAD:")
    summary = merged.groupby("label")[cols].mean().T
    summary.columns = ["Control", "ProbableAD"]
    summary["ratio"] = (summary["ProbableAD"] / summary["Control"].clip(lower=1e-9)).round(2)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
