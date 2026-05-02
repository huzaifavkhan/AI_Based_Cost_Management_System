"""
Fine-tune Donut on combined train/ folder:
  - .xml  (900)  → French FACTURE invoices  (SUPPLIER, invoice_number, currency_id, total)
  - .txt  (973)  → SROIE Malaysian/US receipts (company, date, total with/without symbol)
Total: 1873 labeled samples

Kaggle setup:
  1. Upload data/real_invoices/train/ as a Kaggle dataset  (keep the train/ folder name)
  2. Optionally upload data/donut_invoice/ as a second dataset (saves re-downloading base model)
  3. New notebook → GPU T4 x1 → Internet ON → paste this script → Run All
  4. Download /kaggle/working/donut_invoice/ → replace cost-management-system/data/donut_invoice/
"""

import os, json, re, random
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Install / upgrade dependencies ────────────────────────────────────────────
os.system("pip install -q transformers sentencepiece pillow torch --upgrade")

import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import DonutProcessor, VisionEncoderDecoderModel

# ── Accelerator ───────────────────────────────────────────────────────────────
USE_TPU = False
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device(); USE_TPU = True; print("Accelerator: TPU")
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerator: {device}")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Auto-detect train/ folder — works regardless of your Kaggle dataset name
_INPUT    = Path("/kaggle/input")
_img_hits = list(_INPUT.rglob("img"))
if _img_hits:
    TRAIN_DIR = _img_hits[0].parent
    print(f"[setup] Dataset found at: {TRAIN_DIR}")
else:
    # Fallback: try the hardcoded path you used before
    TRAIN_DIR = Path("/kaggle/input/datasets/huzaifaahmedkhann/fyp-data/train/train")
    print(f"[setup] WARNING: img/ folder not found — trying {TRAIN_DIR}")

IMG_DIR    = TRAIN_DIR / "img"
ENTITY_DIR = TRAIN_DIR / "entities"
OUTPUT_DIR = Path("/kaggle/working/donut_invoice")

# Optional: pre-uploaded base model avoids re-downloading on every run
_LOCAL_BASE = Path("/kaggle/input/datasets/huzaifaahmedkhann/fyp-data/donut_invoice_base/donut_invoice_base")
BASE_MODEL  = str(_LOCAL_BASE) if _LOCAL_BASE.exists() else "naver-clova-ix/donut-base"
print(f"[setup] Base model: {BASE_MODEL}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS      = 5
LR          = 5e-5
MAX_LENGTH  = 128
IMG_SIZE    = (560, 420)
MAX_SAMPLES = None          # None = use all 1873 on GPU

TASK_START  = "<s_invoice>"
TASK_END    = "</s_invoice>"


# ── Currency helpers ──────────────────────────────────────────────────────────

_SYMBOL_MAP = [
    (r"\bRM\b",         "MYR"),
    (r"₹",              "INR"),
    (r"₨|Rs\.?\s*\d",  "PKR"),
    (r"£",              "GBP"),
    (r"€",              "EUR"),
    (r"د\.إ",           "AED"),
    (r"\$",             "USD"),
]

def _currency_from_string(text: str) -> str | None:
    """Detect currency code from a string containing a symbol."""
    for pattern, code in _SYMBOL_MAP:
        if re.search(pattern, text):
            return code
    return None

def _currency_from_address(address: str) -> str:
    """Infer currency from address text. Defaults to MYR (SROIE is 90% Malaysian)."""
    a = address.lower()
    if any(k in a for k in ("malaysia","johor","kuala lumpur","penang","selangor","sabah","sarawak","kl ")):
        return "MYR"
    if any(k in a for k in ("pakistan","karachi","lahore","islamabad","rawalpindi")):
        return "PKR"
    if any(k in a for k in ("india","mumbai","delhi","bangalore","chennai","hyderabad")):
        return "INR"
    if any(k in a for k in ("united states","usa"," ca "," ny "," tx "," fl "," wa ")):
        return "USD"
    if any(k in a for k in ("uk","london","england","britain")):
        return "GBP"
    if any(k in a for k in ("france","paris","lyon","marseille","états unis")):
        return "EUR"
    if any(k in a for k in ("dubai","uae","abu dhabi","sharjah")):
        return "AED"
    return "MYR"  # safe default — SROIE is predominantly Malaysian

def _currency_from_xml_text(xml_text: str) -> str | None:
    """Scan raw XML content for currency symbols when currency_id tag is absent."""
    found = _currency_from_string(xml_text)
    if found:
        return found
    text_lower = xml_text.lower()
    if any(k in text_lower for k in ("états unis","united states","usa"," ca "," ny ")):
        return "USD"
    if any(k in text_lower for k in ("france","paris","lyon")):
        return "EUR"
    if any(k in text_lower for k in ("malaysia","johor","kuala lumpur")):
        return "MYR"
    if any(k in text_lower for k in ("pakistan","karachi","lahore")):
        return "PKR"
    if any(k in text_lower for k in ("dubai","uae","abu dhabi")):
        return "AED"
    if any(k in text_lower for k in ("uk","london","england")):
        return "GBP"
    return None

def _strip_currency_to_float(raw: str) -> float | None:
    """
    Parse amount string regardless of currency symbol or locale format.
      'RM1.38'       → 1.38
      '$8.20'        → 8.20
      '€ 348 786,00' → 348786.0   (European: space-thousands, comma-decimal)
      '9.00'         → 9.0
    """
    s = re.sub(r"[€£¥₹₩₺₽﷼฿৳]|د\.إ|R\$|\bRM\b|\bRp\b|\bRs\.?\b|\bFr\.?\b|\bkr\b|\$",
               "", raw, flags=re.IGNORECASE).strip()
    if re.search(r",\d{1,2}$", s):                  # European decimal comma
        s = s.replace(".", "").replace(" ", "").replace(",", ".")
    else:
        s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


# ── Target sequence builder ───────────────────────────────────────────────────

def fields_to_target(fields: dict) -> str:
    def _tag(name, val):
        v = str(val).strip() if val is not None else ""
        return f"<s_{name}>{v}</s_{name}>"
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


# ── Field parsers ─────────────────────────────────────────────────────────────

def _parse_xml(label_path: Path) -> dict:
    """Parse French FACTURE XML annotation."""
    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    root     = ET.parse(label_path).getroot()

    def _x(tag):
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None

    # Fix total: XML stores tax in total_amount — compute real total
    untaxed = _x("total_untaxed")
    tax     = _x("tax_amount")
    total   = None
    try:
        if untaxed and tax:
            total = str(round(float(untaxed) + float(tax), 2))
        elif untaxed:
            total = untaxed
        else:
            total = _x("total_amount")
    except (ValueError, TypeError):
        total = _x("total_amount")

    # Currency: structured field first, then scan XML text
    currency = _x("currency_id") or _currency_from_xml_text(raw_text)

    return {
        "vendor":         _x("supplier"),
        "invoice_number": _x("invoice_number"),
        "date":           _x("invoice_date"),
        "po_number":      _x("po_number"),
        "total_amount":   total,
        "currency":       currency,
    }


def _parse_txt(label_path: Path) -> dict:
    """Parse SROIE .txt JSON annotation."""
    meta      = json.loads(label_path.read_text(encoding="utf-8"))
    raw_total = str(meta.get("total", "") or "")
    address   = str(meta.get("address", "") or "")

    # Currency: symbol in total string first, then infer from address
    currency     = _currency_from_string(raw_total) or _currency_from_address(address)
    total_amount = _strip_currency_to_float(raw_total) if raw_total else None

    return {
        "vendor":         meta.get("company"),
        "invoice_number": None,          # SROIE has no invoice number
        "date":           meta.get("date"),
        "po_number":      None,
        "total_amount":   str(total_amount) if total_amount is not None else None,
        "currency":       currency,
    }


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_all_samples():
    samples   = []
    xml_count = txt_count = skipped = 0

    for label_path in sorted(ENTITY_DIR.glob("*")):
        if label_path.suffix not in (".xml", ".txt"):
            continue

        img_path = next(
            (IMG_DIR / (label_path.stem + ext)
             for ext in (".jpg", ".jpeg", ".png")
             if (IMG_DIR / (label_path.stem + ext)).exists()),
            None,
        )
        if not img_path:
            skipped += 1
            continue

        try:
            if label_path.suffix == ".xml":
                fields = _parse_xml(label_path)
                xml_count += 1
            else:
                fields = _parse_txt(label_path)
                txt_count += 1
            samples.append((img_path, fields_to_target(fields)))
        except Exception as e:
            skipped += 1

    print(f"  Loaded {len(samples)} samples  ({xml_count} XML French + {txt_count} TXT SROIE)")
    print(f"  Skipped: {skipped}")
    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print(f"\nBase model: {BASE_MODEL}\n")

    _local = _LOCAL_BASE.exists()
    processor = DonutProcessor.from_pretrained(BASE_MODEL, local_files_only=_local)
    model     = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL, local_files_only=_local)

    new_tokens = [
        "<s_invoice>",        "</s_invoice>",
        "<s_vendor>",         "</s_vendor>",
        "<s_invoice_number>", "</s_invoice_number>",
        "<s_date>",           "</s_date>",
        "<s_po_number>",      "</s_po_number>",
        "<s_total>",          "</s_total>",
        "<s_currency>",       "</s_currency>",
    ]
    added = processor.tokenizer.add_tokens(new_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_invoice>"])[0]
    processor.image_processor.size      = {"height": IMG_SIZE[0], "width": IMG_SIZE[1]}
    print(f"Added {added} task tokens\n")

    samples = load_all_samples()
    if not samples:
        print("ERROR: No samples found."); return

    random.shuffle(samples)
    if MAX_SAMPLES and len(samples) > MAX_SAMPLES:
        print(f"Capping {len(samples)} → {MAX_SAMPLES}")
        samples = samples[:MAX_SAMPLES]

    print(f"\nTraining: {len(samples)} samples × {EPOCHS} epochs on {device}\n")

    model = model.to(device)

    # Freeze encoder — only fine-tune decoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    _seen = set()
    trainable = [p for p in model.parameters()
                 if p.requires_grad and id(p) not in _seen and not _seen.add(id(p))]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,} (decoder only)\n")

    model.train()
    optimizer = AdamW(trainable, lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for img_path, target in samples:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            pv = processor(img, return_tensors="pt").pixel_values.to(device)
            img.close()

            labels = processor.tokenizer(
                target,
                add_special_tokens=False,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            labels[labels == processor.tokenizer.pad_token_id] = -100

            loss = model(pixel_values=pv, labels=labels).loss
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            if USE_TPU:
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch}/{EPOCHS}  loss={epoch_loss / len(samples):.4f}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved → {OUTPUT_DIR}")
    print("Download the donut_invoice/ folder from Kaggle output and replace data/donut_invoice/ locally.")


train()
