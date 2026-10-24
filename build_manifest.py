"""
Build a subject-level manifest of the DementiaBank Pitt "Cookie Theft" transcripts.

One row per .cha file, carrying:
  - subject_id / visit_id parsed from the filename (e.g. NNN-V.cha -> subject NNN, visit V)
  - the folder-level group label (Control / Dementia)
  - the per-visit clinical label from the @ID header (Control, MCI, ProbableAD, ...)
  - age, sex, MMSE from the @ID header
  - the cleaned PAR-only transcript

The subject_id column is what makes speaker-disjoint splitting possible: the Pitt
corpus is longitudinal, so a random file-level split leaks the same speaker across
train and test.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import pylangacq

# @ID: eng|Pitt|PAR|72;|female|Control||Participant|27||
#      1    2    3   4    5      6      7      8       9
# Field 9 is documented as "education" in the CHAT spec, but DementiaBank Pitt
# stores MMSE there (Control 24-30, Dementia 1-30 -- verified against the corpus).
ID_RE = re.compile(r"^@ID:\s*(.*)$", re.MULTILINE)

# CHAT annotation noise we never want in the model input.
DROP_TOKENS = {"+", "-", ".", "?", "!", ",", "+...", "+/.", "+//.", "+<", "++"}
UNINTELLIGIBLE = {"xxx", "yyy", "www"}


def parse_id_header(cha_path: Path) -> dict:
    """Pull age / sex / diagnosis / MMSE off the PAR @ID line."""
    text = cha_path.read_text(encoding="utf-8", errors="replace")
    for raw in ID_RE.findall(text):
        fields = raw.split("|")
        if len(fields) < 9 or fields[2] != "PAR":
            continue

        age_raw = fields[3].strip()
        age = None
        if age_raw:
            m = re.match(r"(\d+)", age_raw)
            if m:
                age = int(m.group(1))

        mmse_raw = fields[8].strip()
        mmse = int(mmse_raw) if mmse_raw.isdigit() else None

        return {
            "age": age,
            "sex": fields[4].strip() or None,
            "dx": fields[5].strip() or None,
            "mmse": mmse,
        }
    return {"age": None, "sex": None, "dx": None, "mmse": None}


def extract_par_text(cha_path: Path, keep_fillers: bool) -> str:
    """Concatenate the participant's words, dropping CHAT markup."""
    chat = pylangacq.read_chat(str(cha_path))
    words = []
    for utt in chat.utterances():
        if utt.participant != "PAR":
            continue
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
    return " ".join(words)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cha-root", default="Pitt.cha")
    ap.add_argument("--task", default="cookie", help="subfolder task name")
    ap.add_argument("--out", default="artifacts/manifest.csv")
    ap.add_argument(
        "--keep-fillers",
        action="store_true",
        default=True,
        help="retain filled pauses (um/uh); they are clinically informative",
    )
    ap.add_argument("--drop-fillers", dest="keep_fillers", action="store_false")
    ap.add_argument("--min-words", type=int, default=3)
    args = ap.parse_args()

    cha_root = Path(args.cha_root)
    rows, skipped = [], []

    for group in ("Control", "Dementia"):
        task_dir = cha_root / group / args.task
        if not task_dir.is_dir():
            print(f"[warn] missing {task_dir}", file=sys.stderr)
            continue

        for cha_path in sorted(task_dir.glob("*.cha")):
            stem = cha_path.stem
            if "-" in stem:
                subject_id, visit_id = stem.split("-", 1)
            else:
                subject_id, visit_id = stem, "0"

            meta = parse_id_header(cha_path)
            try:
                text = extract_par_text(cha_path, args.keep_fillers)
            except Exception as exc:  # noqa: BLE001 - report and continue
                skipped.append((stem, f"parse error: {exc}"))
                continue

            n_words = len(text.split())
            if n_words < args.min_words:
                skipped.append((stem, f"only {n_words} words"))
                continue

            rows.append(
                {
                    "file_id": stem,
                    "subject_id": subject_id,
                    "visit_id": visit_id,
                    "group": group,
                    "label": 1 if group == "Dementia" else 0,
                    "dx": meta["dx"],
                    "age": meta["age"],
                    "sex": meta["sex"],
                    "mmse": meta["mmse"],
                    "n_words": n_words,
                    "path": str(cha_path).replace("\\", "/"),
                    "text": text,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out_path}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for stem, why in skipped[:20]:
            print(f"  {stem}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
