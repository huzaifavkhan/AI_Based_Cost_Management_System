"""
Generate TSV bounding-box files for numbered invoice images (2–23).

Design: imports the same preprocessing and label-alignment helpers from
finetune_layoutlm_kaggle.py so that the OCR pipeline and label logic are
identical to training — guaranteeing train/inference parity per Option A.

Output format (matches FACTU*.tsv used by the training script):
    left,top,width,height,text,label

Run on Kaggle (Tesseract pre-installed) or locally with Tesseract installed:
    python scripts/generate_box_files.py

Box files are optional; the training script falls back to heuristic label
alignment when no box file is found.
"""

import sys
import csv
import json
import importlib.util
from pathlib import Path

# ── Locate the main training script so we can reuse its helpers ───────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_MAIN_SCRIPT = _SCRIPT_DIR / "finetune_layoutlm_kaggle.py"

if not _MAIN_SCRIPT.exists():
    _MAIN_SCRIPT = Path("/kaggle/working/finetune_layoutlm_kaggle.py")

if not _MAIN_SCRIPT.exists():
    print(f"ERROR: cannot find finetune_layoutlm_kaggle.py")
    print("  Place generate_box_files.py in the same directory as the training script.")
    sys.exit(1)

spec   = importlib.util.spec_from_file_location("layoutlm_main", _MAIN_SCRIPT)
lm     = importlib.util.module_from_spec(spec)
# Suppress the train() call at module level by monkey-patching before exec
import builtins
_real_print = builtins.print
spec.loader.exec_module(lm)

# ── Re-use helpers from the training script ───────────────────────────────────
_get_words_and_boxes  = lm._get_words_and_boxes
_align_labels         = lm._align_labels
_disambiguate_labels  = lm._disambiguate_labels
_parse_txt            = lm._parse_txt

# ── Paths ─────────────────────────────────────────────────────────────────────
_TRAIN_DIR   = lm.TRAIN_DIR
IMG_DIR      = lm.IMG_DIR
ENTITY_DIR   = lm.ENTITY_DIR
BOX_DIR      = lm.BOX_DIR

_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# ── Label map: model field → TSV label ────────────────────────────────────────
# Mirrors _TSV_LABEL_MAP in reverse so we output labels the training script
# understands when it reads back the generated TSV.
_FIELD_TO_TSV = {
    "VENDOR":     "SUPPLIER",
    "DATE":       "INVOICE_DATE",
    "TOTAL":      "TOTAL_AMOUNT",
    "SUBTOTAL":   "TOTAL_UNTAXED",
    "INVOICE_NO": "NUMBER",
    "PO_NO":      "PO_NUMBER",
    "CURRENCY":   "O",   # no direct TSV label; mark O so it still maps to O
    "O":          "O",
}


def _bio_to_tsv_label(bio_label: str) -> str:
    import re
    field = re.sub(r"^[BI]-", "", bio_label)
    return _FIELD_TO_TSV.get(field, "O")


def generate_box_file(img_path: Path, entity_path: Path, out_tsv: Path):
    from PIL import Image
    img      = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    words, boxes = _get_words_and_boxes(img, lang="eng")
    img.close()

    if not words:
        print(f"  [skip] no OCR words — {img_path.name}")
        return

    entities   = _parse_txt(entity_path)
    bio_labels = _align_labels(words, entities)
    bio_labels = _disambiguate_labels(words, bio_labels, boxes)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["left", "top", "width", "height", "text", "label"])
        for word, box, bio in zip(words, boxes, bio_labels):
            # Convert normalised [0-1000] coordinates back to pixel space
            left   = int(box[0] / 1000 * img_w)
            top    = int(box[1] / 1000 * img_h)
            right  = int(box[2] / 1000 * img_w)
            bottom = int(box[3] / 1000 * img_h)
            width  = max(right - left, 1)
            height = max(bottom - top, 1)
            tsv_label = _bio_to_tsv_label(bio)
            writer.writerow([left, top, width, height, word, tsv_label])

    n_labeled = sum(1 for b in bio_labels if b != "O")
    print(f"  [ok] {out_tsv.name}  ({len(words)} words, {n_labeled} labeled)")


def main():
    stems = [str(i) for i in range(2, 24)]   # "2" … "23"

    print(f"Box-file generator")
    print(f"  img/     → {IMG_DIR}")
    print(f"  entities/→ {ENTITY_DIR}")
    print(f"  box/     → {BOX_DIR}\n")

    ok = skipped = 0
    for stem in stems:
        entity_path = ENTITY_DIR / f"{stem}.txt"
        if not entity_path.exists():
            print(f"  [skip] no entity file for {stem}")
            skipped += 1
            continue

        img_path = next(
            (IMG_DIR / (stem + ext) for ext in _IMG_EXTENSIONS
             if (IMG_DIR / (stem + ext)).exists()),
            None,
        )
        if img_path is None:
            print(f"  [skip] no image found for {stem}")
            skipped += 1
            continue

        out_tsv = BOX_DIR / f"{stem}.tsv"
        if out_tsv.exists():
            print(f"  [exists] {out_tsv.name} — skipping (delete to regenerate)")
            ok += 1
            continue

        try:
            generate_box_file(img_path, entity_path, out_tsv)
            ok += 1
        except Exception as e:
            print(f"  [error] {stem}: {type(e).__name__}: {e}")
            skipped += 1

    print(f"\nDone. Generated={ok}  Skipped/failed={skipped}")


if __name__ == "__main__":
    main()
