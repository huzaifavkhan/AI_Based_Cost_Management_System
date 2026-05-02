# Non-IID data partitioning for Federated LayoutLMv3 training.
#
# Splits the 1873 training images into 3 client datasets that differ in
# document format, language, and label distribution — matching the realistic
# scenario where different companies hold invoices from different suppliers.
#
# Partition design:
#   Client A (EuroSupplier, 900 images):
#       All 900 FACTU*.jpg French invoices (XML annotations).
#       Rich label set: VENDOR, DATE, TOTAL, INVOICE_NO, PO_NO, CURRENCY.
#       Language: French (translated to English before LayoutLMv3).
#       Non-IID signal: structured European invoice layout, EUR currency.
#
#   Client B (AsiaRetail_A, ~487 images):
#       Random 50% of the 973 SROIE X0*.jpg receipts (TXT JSON annotations).
#       Sparse label set: VENDOR, DATE, TOTAL only.
#       Language: Malaysian English (no translation).
#       Non-IID signal: point-of-sale receipt layout, RM currency.
#
#   Client C (AsiaRetail_C, ~486 images):
#       Remaining 50% of SROIE receipts (complementary set to Client B).
#       Same format as Client B — quantity non-IID (different specific receipts).
#
# Label alignment helpers are copied verbatim from
# scripts/finetune_layoutlm_kaggle.py to ensure identical preprocessing.
from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

_BASE      = Path(__file__).resolve().parent.parent.parent
_TRAIN_DIR = _BASE / "data" / "real_invoices" / "train"
_IMG_DIR   = _TRAIN_DIR / "img"
_ENTITY_DIR = _TRAIN_DIR / "entities"

LABEL2ID = {
    "O":              0,
    "B-VENDOR":       1,  "I-VENDOR":       2,
    "B-DATE":         3,  "I-DATE":         4,
    "B-TOTAL":        5,  "I-TOTAL":        6,
    "B-SUBTOTAL":     7,  "I-SUBTOTAL":     8,
    "B-INVOICE_NO":   9,  "I-INVOICE_NO":  10,
    "B-PO_NO":       11,  "I-PO_NO":       12,
    "B-CURRENCY":    13,  "I-CURRENCY":    14,
}


@dataclass
class ImageSample:
    """A single annotated invoice image ready for LayoutLMv3 training."""
    img_path:    Path
    label_path:  Path
    source:      str   # "xml" (French FACTU) | "txt" (SROIE)
    stem:        str   # filename without extension (used for box-file lookup)


# ── Date normalisation (verbatim from finetune_layoutlm_kaggle.py) ─────────────

def _normalise_date(raw: str, source: str = "txt") -> str:
    raw = raw.strip()
    if not raw:
        return raw
    if source == "xml":
        m = re.match(r'^(\d{4})(\d{2})(\d{2})$', raw)
        if m:
            return f"{m.group(2)}/{m.group(3)}/{m.group(1)}"
    s = re.sub(r'[-.]', '/', raw)
    m = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', s)
    if m:
        return f"{int(m.group(2)):02d}/{int(m.group(3)):02d}/{m.group(1)}"
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if a > 12:
            return f"{b:02d}/{a:02d}/{y}"
        return f"{a:02d}/{b:02d}/{y}"
    return raw


def _date_digit_variants(raw: str, source: str) -> set[str]:
    normalised = _normalise_date(raw, source)
    m = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', normalised)
    if m:
        mo, d, y = m.group(1), m.group(2), m.group(3)
        return {f"{y}{mo}{d}", f"{d}{mo}{y}", f"{mo}{d}{y}"}
    return {re.sub(r'[^0-9]', '', raw)}


# ── Currency detection (verbatim from finetune_layoutlm_kaggle.py) ─────────────

_SYMBOL_PATTERNS = [
    (r"(?<![A-Za-z])RM(?![A-Za-z])", "RM"),
    (r"R\$",                          "R$"),
    (r"Rs\.?\s*\d",                  "Rs."),
    (r"₹",                            "₹"),
    (r"£",                            "£"),
    (r"€",                            "€"),
    (r"د\.إ",                         "د.إ"),
    (r"\$",                           "$"),
]


def _currency_symbol_from_text(text: str) -> str | None:
    for pattern, symbol in _SYMBOL_PATTERNS:
        if re.search(pattern, text):
            return symbol
    return None


def _currency_symbol_from_address(address: str) -> str | None:
    a = address.lower()
    if any(k in a for k in ("malaysia", "johor", "kuala lumpur", "penang",
                            "selangor", "sabah", "sarawak", "kl ")):
        return "RM"
    if any(k in a for k in ("pakistan", "karachi", "lahore",
                            "islamabad", "rawalpindi")):
        return "Rs."
    if any(k in a for k in ("india", "mumbai", "delhi", "bangalore",
                            "chennai", "hyderabad")):
        return "₹"
    if any(k in a for k in ("united states", "usa", " ca ", " ny ",
                            " tx ", " fl ", " wa ", "états unis")):
        return "$"
    if any(k in a for k in ("uk", "london", "england", "britain")):
        return "£"
    if any(k in a for k in ("france", "paris", "lyon", "marseille")):
        return "€"
    if any(k in a for k in ("dubai", "uae", "abu dhabi", "sharjah")):
        return "د.إ"
    return None


def _detect_currency_symbol(total_str: str = "",
                             address: str = "",
                             full_text: str = "") -> str | None:
    if total_str:
        sym = _currency_symbol_from_text(total_str)
        if sym:
            return sym
    if address:
        sym = _currency_symbol_from_address(address)
        if sym:
            return sym
    if full_text:
        sym = _currency_symbol_from_text(full_text)
        if sym:
            return sym
    return None


# ── Amount normalisation (verbatim from finetune_layoutlm_kaggle.py) ───────────

def _normalise_amount(raw: str) -> str:
    s = re.sub(
        r"[€£¥₹₩₺₽﷼฿৳]|د\.إ|R\$"
        r"|(?<![A-Za-z])RM(?![A-Za-z])"
        r"|(?<![A-Za-z])Rp(?![A-Za-z])"
        r"|Rs\.?|Fr\.?|\bkr\b|\$",
        "", raw, flags=re.IGNORECASE
    ).strip()
    if re.search(r",\d{1,2}$", s):
        s = s.replace(".", "").replace(" ", "").replace(",", ".")
    else:
        s = s.replace(",", "").replace(" ", "")
    return s


# ── Translation (verbatim from finetune_layoutlm_kaggle.py) ───────────────────

def _translate_words(words: list[str]) -> list[str]:
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="fr", target="en")
        translated = translator.translate_batch(words)
        return [
            t if (t and isinstance(t, str)) else w
            for t, w in zip(translated, words)
        ]
    except Exception:
        return words


# ── Entity parsers (verbatim from finetune_layoutlm_kaggle.py) ─────────────────

def _parse_xml(label_path: Path) -> dict:
    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    root     = ET.parse(label_path).getroot()

    def _x(tag):
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None

    untaxed   = _x("total_untaxed")
    tax       = _x("tax_amount")
    sub_total = _x("sub_total")
    try:
        if untaxed and tax:
            total = str(round(float(untaxed) + float(tax), 2))
        elif untaxed:
            total = untaxed
        elif sub_total:
            total = sub_total
        else:
            total = _x("total_amount")
    except (ValueError, TypeError):
        total = _x("sub_total") or _x("total_amount")

    raw_date = _x("invoice_date") or ""
    address  = _x("address") or ""
    currency = _detect_currency_symbol(
        total_str="",
        address=address,
        full_text=raw_text,
    )

    return {
        "vendor":           _x("supplier"),
        "date":             _normalise_date(raw_date, source="xml"),
        "date_raw":         raw_date,
        "total_amount":     total,
        "subtotal_amount":  _x("total_untaxed") or _x("sub_total"),
        "invoice_number":   _x("invoice_number"),
        "po_number":        _x("po_number"),
        "currency":         currency,
        "source":           "xml",
    }


def _parse_txt(label_path: Path) -> dict:
    meta      = json.loads(label_path.read_text(encoding="utf-8"))
    raw_date  = str(meta.get("date", "") or "")
    raw_total = str(meta.get("total", "") or "")
    address   = str(meta.get("address", "") or "")
    currency  = _detect_currency_symbol(total_str=raw_total, address=address)

    return {
        "vendor":         meta.get("company"),
        "date":           _normalise_date(raw_date, source="txt"),
        "date_raw":       raw_date,
        "total_amount":   raw_total,
        "invoice_number": None,
        "po_number":      None,
        "currency":       currency,
        "source":         "txt",
    }


# ── Tesseract word extraction (verbatim from finetune_layoutlm_kaggle.py) ──────

def _get_words_and_boxes(pil_image) -> tuple[list[str], list[list[int]]]:
    """Run Tesseract → (words, normalised [0,1000] boxes)."""
    import os
    import pytesseract
    for _p in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]:
        if os.path.isfile(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break
    w, h   = pil_image.size
    config = "--oem 3 --psm 6"
    data   = pytesseract.image_to_data(
        pil_image, config=config, output_type=pytesseract.Output.DICT
    )
    words, boxes = [], []
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        if not text or int(data["conf"][i]) < 30:
            continue
        x  = data["left"][i];  y  = data["top"][i]
        bw = data["width"][i]; bh = data["height"][i]
        boxes.append([
            min(max(int(x      / w * 1000), 0), 1000),
            min(max(int(y      / h * 1000), 0), 1000),
            min(max(int((x+bw) / w * 1000), 0), 1000),
            min(max(int((y+bh) / h * 1000), 0), 1000),
        ])
        words.append(text)
    return words, boxes


# ── BIO label alignment (verbatim from finetune_layoutlm_kaggle.py) ────────────

def _align_labels(words: list[str], entities: dict) -> list[str]:
    """Align entity values to OCR words → BIO label list."""
    labels     = ["O"] * len(words)
    words_norm = [re.sub(r"[^a-z0-9./,]", "", w.lower()) for w in words]
    source     = entities.get("source", "txt")

    def _mark(start: int, length: int, field: str):
        labels[start] = f"B-{field}"
        for k in range(1, length):
            if start + k < len(labels):
                labels[start + k] = f"I-{field}"

    # ── VENDOR ────────────────────────────────────────────────────────────────
    vendor = entities.get("vendor")
    if vendor:
        vtokens = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in vendor.split() if t]
        vtokens = [t for t in vtokens if t]
        n = len(vtokens)
        if n:
            for i in range(len(words) - n + 1):
                if all(words_norm[i+j] == vtokens[j] for j in range(n)):
                    _mark(i, n, "VENDOR"); break
            else:
                for i in range(len(words)):
                    if words_norm[i] == vtokens[0]:
                        run = 1
                        while run < n and (i+run) < len(words) and words_norm[i+run] == vtokens[run]:
                            run += 1
                        if run >= max(1, n // 2):
                            _mark(i, run, "VENDOR"); break

    # ── DATE ──────────────────────────────────────────────────────────────────
    date_raw = entities.get("date_raw", "") or entities.get("date", "")
    if date_raw:
        variants = _date_digit_variants(date_raw, source)
        for i, w in enumerate(words):
            w_digits = re.sub(r"[^0-9]", "", w)
            if len(w_digits) == 8 and w_digits in variants:
                _mark(i, 1, "DATE"); break
        else:
            date_norm = re.sub(r"[^a-z0-9]", "", date_raw.lower())
            for i, wn in enumerate(words_norm):
                if date_norm and len(date_norm) >= 4 and (date_norm in wn or wn in date_norm):
                    _mark(i, 1, "DATE"); break

    # ── TOTAL ─────────────────────────────────────────────────────────────────
    total_raw = entities.get("total_amount")
    if total_raw:
        total_norm = _normalise_amount(total_raw)
        if total_norm:
            for i, w in enumerate(words):
                if _normalise_amount(w) == total_norm:
                    _mark(i, 1, "TOTAL"); break

    # ── SUBTOTAL ──────────────────────────────────────────────────────────────
    subtotal_raw = entities.get("subtotal_amount")
    if subtotal_raw:
        sub_norm = _normalise_amount(str(subtotal_raw))
        if sub_norm:
            for i, w in enumerate(words):
                if labels[i] == "O" and _normalise_amount(w) == sub_norm:
                    _mark(i, 1, "SUBTOTAL"); break

    # ── INVOICE_NO (XML only) ─────────────────────────────────────────────────
    inv_no = entities.get("invoice_number")
    if inv_no:
        inv_norm = re.sub(r"[^a-z0-9/]", "", inv_no.lower())
        inv_toks = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in inv_no.split() if t]
        inv_toks = [t for t in inv_toks if t]
        n = len(inv_toks)
        for i in range(len(words) - n + 1):
            if all(words_norm[i+j] == inv_toks[j] for j in range(n)):
                _mark(i, n, "INVOICE_NO"); break
        else:
            for i, wn in enumerate(words_norm):
                wn_clean = re.sub(r"[^a-z0-9/]", "", wn)
                if inv_norm and inv_norm == wn_clean:
                    _mark(i, 1, "INVOICE_NO"); break

    # ── PO_NO (XML only) ──────────────────────────────────────────────────────
    po_no = entities.get("po_number")
    if po_no:
        po_norm = re.sub(r"[^a-z0-9]", "", po_no.lower())
        for i, wn in enumerate(words_norm):
            wn_clean = re.sub(r"[^a-z0-9]", "", wn)
            if po_norm and po_norm == wn_clean:
                _mark(i, 1, "PO_NO"); break

    # ── CURRENCY ──────────────────────────────────────────────────────────────
    currency_sym = entities.get("currency")
    if currency_sym:
        sym_esc = re.escape(currency_sym)
        for i, w in enumerate(words):
            if labels[i] == "O" and re.fullmatch(sym_esc, w.strip(), re.IGNORECASE):
                _mark(i, 1, "CURRENCY"); break
        else:
            for i, w in enumerate(words):
                if labels[i] == "O" and re.match(rf'^{sym_esc}', w, re.IGNORECASE):
                    _mark(i, 1, "CURRENCY"); break

    return labels


# ── Main partition function ────────────────────────────────────────────────────

def partition_layoutlm_images(
    train_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, list[ImageSample]]:
    """
    Discover all annotated images in *train_dir* and split them into three
    non-IID client lists.

    Returns
    -------
    {
        "client_a": [ImageSample, ...],   # 900 French FACTU XML invoices
        "client_b": [ImageSample, ...],   # ~487 SROIE TXT receipts
        "client_c": [ImageSample, ...],   # ~486 SROIE TXT receipts
    }
    """
    img_dir    = Path(train_dir) / "img"    if train_dir else _IMG_DIR
    entity_dir = Path(train_dir) / "entities" if train_dir else _ENTITY_DIR

    xml_samples: list[ImageSample] = []
    txt_samples: list[ImageSample] = []

    for label_path in sorted(entity_dir.glob("*")):
        if label_path.suffix not in (".xml", ".txt"):
            continue
        img_path = next(
            (img_dir / (label_path.stem + ext)
             for ext in (".jpg", ".jpeg", ".png")
             if (img_dir / (label_path.stem + ext)).exists()),
            None,
        )
        if img_path is None:
            continue

        sample = ImageSample(
            img_path   = img_path,
            label_path = label_path,
            source     = "xml" if label_path.suffix == ".xml" else "txt",
            stem       = label_path.stem,
        )
        if label_path.suffix == ".xml":
            xml_samples.append(sample)
        else:
            txt_samples.append(sample)

    # Shuffle SROIE list for reproducible 50/50 split
    rng = random.Random(seed)
    rng.shuffle(txt_samples)
    mid = len(txt_samples) // 2

    return {
        "client_a": xml_samples,          # all French FACTU XML
        "client_b": txt_samples[:mid],    # first half of SROIE
        "client_c": txt_samples[mid:],    # second half of SROIE
    }


# ── Sample loading (words + boxes + BIO labels for one ImageSample) ───────────

def load_sample_words_boxes_labels(
    sample: ImageSample,
) -> tuple[list[str], list[list[int]], list[str]] | None:
    """
    Load one ImageSample, run OCR (or read pre-computed boxes if available),
    align BIO labels from the annotation, and return (words, boxes, bio_labels).

    Returns None if the sample fails to load or produces no words.

    For XML (French) samples, words are kept in French for label alignment
    (because entity strings are also in French).  The caller is responsible
    for translating words to English before feeding to LayoutLMv3.
    """
    try:
        from PIL import Image

        img = Image.open(sample.img_path).convert("RGB")
        words, boxes = _get_words_and_boxes(img)
        img.close()

        if not words:
            return None

        if sample.source == "xml":
            entities = _parse_xml(sample.label_path)
        else:
            entities = _parse_txt(sample.label_path)

        bio_labels = _align_labels(words, entities)

        # Translate French words to English AFTER alignment
        if sample.source == "xml":
            words = _translate_words(words)

        return words, boxes, bio_labels

    except Exception as e:
        print(f"[data_partitioner] Skipping {sample.stem}: {e}")
        return None


def get_client_stats(client_id: str, samples: list[ImageSample]) -> dict:
    """Return a summary dict for logging and dashboard display."""
    xml_count = sum(1 for s in samples if s.source == "xml")
    txt_count = sum(1 for s in samples if s.source == "txt")
    return {
        "client_id":    client_id,
        "n_samples":    len(samples),
        "xml_invoices": xml_count,   # French FACTU (rich labels)
        "txt_receipts": txt_count,   # SROIE (sparse labels)
    }
