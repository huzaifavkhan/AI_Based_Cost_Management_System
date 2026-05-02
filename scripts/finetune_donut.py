"""
Fine-tune Donut on all available labeled invoice/receipt data:

  1. SROIE train + test  (data/real_invoices/train/ + test/)   — 973 receipts
  2. invoice_dataset_model_1..9  (TSV annotations)             — 900 invoices
  3. Groq extraction cache  (data/extraction_cache/)           — our own invoices

Total: ~1,873 labeled samples. Run on Kaggle GPU for best results (~15 min).
On CPU with MAX_SAMPLES=200 it takes ~20 min.

Usage:
    python scripts/finetune_donut.py
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import DonutProcessor, VisionEncoderDecoderModel

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE         = Path(__file__).resolve().parent.parent
CACHE_DIR     = _BASE / "data" / "extraction_cache"
REAL_DIR      = _BASE / "data" / "real_invoices"
OUTPUT_DIR    = _BASE / "data" / "donut_invoice"

_LOCAL_BASE   = _BASE / "data" / "donut_invoice_base"
BASE_MODEL    = str(_LOCAL_BASE) if _LOCAL_BASE.exists() else "naver-clova-ix/donut-base"

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS        = 3
LR            = 5e-5
MAX_LENGTH    = 128
IMG_SIZE      = (560, 420)    # height × width — reduced for 8GB CPU RAM
MAX_SAMPLES   = 200           # set to None on Kaggle GPU to use all 1,873

TASK_START    = "<s_invoice>"
TASK_END      = "</s_invoice>"


# ── Target sequence builder ───────────────────────────────────────────────────

def fields_to_target(fields: dict) -> str:
    def _tag(name, val):
        # Omit tag entirely if None — don't teach model to output empty tags
        if val is None:
            return ""
        v = str(val).strip()
        return f"<s_{name}>{v}</s_{name}>" if v else ""
    return (
        TASK_START
        + _tag("vendor",         fields.get("vendor"))
        + _tag("invoice_number", fields.get("invoice_number"))
        + _tag("date",           fields.get("date"))
        + _tag("po_number",      fields.get("po_number"))
        + _tag("total",          fields.get("total_amount"))
        + _tag("currency",       fields.get("currency"))
        + TASK_END
    )


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_sroie_samples():
    """SROIE format: entities/*.txt  +  img/*.jpg
    Labels: company→vendor, date→date, total→total_amount
    Loads from both train/ and test/ subdirectories.
    """
    samples = []
    for split in ("train", "test"):
        split_dir = REAL_DIR / split
        img_dir    = split_dir / "img"
        entity_dir = split_dir / "entities"
        if not img_dir.exists():
            continue
        for txt_path in sorted(entity_dir.glob("*.txt")):
            img_path = next(
                (img_dir / (txt_path.stem + ext)
                 for ext in (".jpg", ".jpeg", ".png")
                 if (img_dir / (txt_path.stem + ext)).exists()),
                None,
            )
            if img_path is None:
                continue
            try:
                meta = json.loads(txt_path.read_text(encoding="utf-8"))
                fields = {
                    "vendor":         meta.get("company"),
                    "invoice_number": None,
                    "date":           meta.get("date"),
                    "po_number":      None,
                    "total_amount":   meta.get("total"),
                }
                samples.append((Image.open(img_path).convert("RGB"), fields_to_target(fields)))
            except Exception as e:
                pass  # skip corrupted files silently
    print(f"  [SROIE train+test] Loaded {len(samples)} samples")
    return samples


def load_model_dataset_samples():
    """invoice_dataset_model_1..9 format: annotations/*.tsv  +  images/*.jpg
    TSV columns: left,top,width,height,text,label
    Key labels: SUPPLIER, NUMBER, INVOICE_DATE, PO_NUMBER, TOTAL_AMOUNT
    Multi-word fields are split across rows — we join them with spaces.
    """
    TARGET_LABELS = {
        "SUPPLIER":     "vendor",
        "NUMBER":       "invoice_number",
        "INVOICE_DATE": "date",
        "PO_NUMBER":    "po_number",
        "TOTAL_AMOUNT": "total_amount",
    }
    samples = []
    for i in range(1, 10):
        dataset_dir = REAL_DIR / f"invoice_dataset_model_{i}"
        img_dir     = dataset_dir / "images"
        ann_dir     = dataset_dir / "annotations"
        if not ann_dir.exists():
            continue
        for tsv_path in sorted(ann_dir.glob("*.tsv")):
            img_path = next(
                (img_dir / (tsv_path.stem + ext)
                 for ext in (".jpg", ".jpeg", ".png")
                 if (img_dir / (tsv_path.stem + ext)).exists()),
                None,
            )
            if img_path is None:
                continue
            try:
                rows = list(csv.DictReader(tsv_path.read_text(encoding="utf-8").splitlines()))
                # Accumulate tokens per field label
                field_tokens: dict[str, list[str]] = {v: [] for v in TARGET_LABELS.values()}
                for row in rows:
                    mapped = TARGET_LABELS.get(row.get("label", ""))
                    if mapped:
                        field_tokens[mapped].append(row["text"])
                fields = {k: " ".join(v) if v else None for k, v in field_tokens.items()}
                samples.append((Image.open(img_path).convert("RGB"), fields_to_target(fields)))
            except Exception:
                pass
    print(f"  [invoice_dataset_model_1..9] Loaded {len(samples)} samples")
    return samples


def load_cache_samples():
    """Groq-labeled invoices from data/extraction_cache/"""
    samples = []
    for json_path in CACHE_DIR.glob("*.json"):
        img_path = json_path.with_suffix(".jpg")
        if not img_path.exists():
            continue
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            # Cache stores fields directly (not nested under "fields" key)
            if isinstance(raw, dict) and "vendor" in raw:
                fields = raw
            else:
                fields = raw.get("fields", raw)
            samples.append((Image.open(img_path).convert("RGB"), fields_to_target(fields)))
        except Exception:
            pass
    print(f"  [groq cache] Loaded {len(samples)} samples")
    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print(f"\n[finetune_donut] Base model: {BASE_MODEL}")

    processor = DonutProcessor.from_pretrained(BASE_MODEL, local_files_only=_LOCAL_BASE.exists())
    model     = VisionEncoderDecoderModel.from_pretrained(
        BASE_MODEL, local_files_only=_LOCAL_BASE.exists()
    )

    new_tokens = [
        "<s_invoice>", "</s_invoice>",
        "<s_vendor>",  "</s_vendor>",
        "<s_invoice_number>", "</s_invoice_number>",
        "<s_date>",    "</s_date>",
        "<s_po_number>", "</s_po_number>",
        "<s_total>",   "</s_total>",
    ]
    added = processor.tokenizer.add_tokens(new_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_invoice>"])[0]
    processor.image_processor.size      = {"height": IMG_SIZE[0], "width": IMG_SIZE[1]}
    print(f"  Added {added} task tokens\n")

    # Load all datasets
    samples = (
        load_sroie_samples()
        + load_model_dataset_samples()
        + load_cache_samples()
    )
    if not samples:
        print("ERROR: No samples found.")
        sys.exit(1)

    random.shuffle(samples)
    if MAX_SAMPLES and len(samples) > MAX_SAMPLES:
        print(f"  Capping {len(samples)} → {MAX_SAMPLES} samples (set MAX_SAMPLES=None on GPU)")
        samples = samples[:MAX_SAMPLES]

    est_min = len(samples) * EPOCHS * 10 // 60
    print(f"\n  Total: {len(samples)} samples | {EPOCHS} epochs | ~{est_min} min on CPU\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model  = model.to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False

    _seen = set()
    trainable = [p for p in model.parameters()
                 if p.requires_grad and id(p) not in _seen and not _seen.add(id(p))]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,} (decoder only)\n")

    model.train()
    optimizer = AdamW(trainable, lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for img, target in samples:
            pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
            labels = processor.tokenizer(
                target,
                add_special_tokens=False,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs     = model(pixel_values=pixel_values, labels=labels)
            loss        = outputs.loss
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        print(f"  Epoch {epoch}/{EPOCHS}  loss={epoch_loss / len(samples):.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved to {OUTPUT_DIR} — Donut is now active as Tier 0 extractor.\n")


if __name__ == "__main__":
    train()
