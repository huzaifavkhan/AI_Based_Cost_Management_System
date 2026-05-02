# Two-stage invoice field extractor.
# Stage 1: Tesseract (ocr_engine.py) extracts words + bounding boxes.
# Stage 2: LayoutLMv3 classifies each word into a field label (this file).
#
# Replaces Donut as Tier 0. Does NOT hallucinate — only labels text that
# Tesseract actually found in the image.
#
# French (and other non-English) invoices are translated to English before
# being fed to LayoutLMv3, matching how the model was trained.
#
# Model location: data/layoutlm_invoice/  (output of finetune_layoutlm_kaggle.py)
from __future__ import annotations
import re
from pathlib import Path
from PIL import Image

from app.extraction.ocr_engine import _best_word_data, extract_text_from_pil
from app.extraction.field_extractor import extract_fields as regex_extract
from app.extraction.currency_detector import detect_currency

_BASE      = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _BASE / "data" / "layoutlm_invoice"

# Lazy-loaded — only imported when actually needed
_processor = None
_model     = None

# Maps model label → our fields dict key
_LABEL2FIELD: dict[str, str] = {
    "B-VENDOR":     "vendor",           "I-VENDOR":     "vendor",
    "B-DATE":       "date",             "I-DATE":       "date",
    "B-TOTAL":      "total_amount",     "I-TOTAL":      "total_amount",
    "B-SUBTOTAL":   "subtotal_amount",  "I-SUBTOTAL":   "subtotal_amount",
    "B-INVOICE_NO": "invoice_number",   "I-INVOICE_NO": "invoice_number",
    "B-PO_NO":      "po_number",        "I-PO_NO":      "po_number",
    "B-CURRENCY":   "currency",         "I-CURRENCY":   "currency",
}


# ── Date normalisation ────────────────────────────────────────────────────────

_MONTH_NAMES_EXT = {
    "january": 1, "february": 2, "march": 3,   "april": 4,
    "may": 5,     "june": 6,     "july": 7,    "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _normalise_date(raw: str) -> str:
    """
    Normalise a date string extracted by LayoutLMv3 to MM/DD/YYYY.
    Handles YYYYMMDD, YYYY-MM-DD, DD/MM/YYYY (when day > 12), MM/DD/YYYY,
    and month-name formats: "21-March-2016", "Mar 21, 2016", etc.
    """
    raw = raw.strip()
    if not raw or len(raw) < 6:
        return raw

    # YYYYMMDD compact
    m = re.match(r'^(\d{4})(\d{2})(\d{2})$', raw)
    if m:
        return f"{m.group(2)}/{m.group(3)}/{m.group(1)}"

    # DD.MM.YYYY explicit guard (before generic [-.]→/ substitution)
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', raw)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if a > 12:
            return f"{b:02d}/{a:02d}/{y}"
        return f"{a:02d}/{b:02d}/{y}"

    s = re.sub(r'[-.]', '/', raw)

    # YYYY/MM/DD
    m = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', s)
    if m:
        return f"{int(m.group(2)):02d}/{int(m.group(3)):02d}/{m.group(1)}"

    # DD/MM/YYYY or MM/DD/YYYY
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if a > 12:                       # unambiguous DD/MM/YYYY
            return f"{b:02d}/{a:02d}/{y}"
        return f"{a:02d}/{b:02d}/{y}"    # assume MM/DD/YYYY

    # MM/DD/YY or DD/MM/YY (2-digit year — common on thermal receipts)
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2})$', s)
    if m:
        a, b, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = 2000 + y2 if y2 < 50 else 1900 + y2
        if a > 12:
            return f"{b:02d}/{a:02d}/{y}"
        return f"{a:02d}/{b:02d}/{y}"

    # Pattern A: DD[-\s]Month[-\s,]YYYY  e.g. "21-March-2016", "21 Mar 16"
    m = re.match(r'^(\d{1,2})[\s\-]([A-Za-z]+)[\s\-,]?\s*(\d{2,4})$', raw.strip())
    if m:
        mo = _MONTH_NAMES_EXT.get(m.group(2).lower())
        if mo:
            y = int(m.group(3))
            y = (2000 + y if y < 50 else 1900 + y) if y < 100 else y
            return f"{mo:02d}/{int(m.group(1)):02d}/{y}"

    # Pattern B: Month DD, YYYY  e.g. "March 21, 2016", "Mar 21, 2016"
    m = re.match(r'^([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{2,4})$', raw.strip())
    if m:
        mo = _MONTH_NAMES_EXT.get(m.group(1).lower())
        if mo:
            y = int(m.group(3))
            y = (2000 + y if y < 50 else 1900 + y) if y < 100 else y
            return f"{mo:02d}/{int(m.group(2)):02d}/{y}"

    return raw


def _is_valid_date(s: str) -> bool:
    """Return True only if s looks like a real date (not a stray digit/flag)."""
    if not s or len(s) < 6:
        return False
    # Must contain at least one separator or be all-digits of length 8
    return bool(re.search(r'[\-/]', s) or re.fullmatch(r'\d{8}', s))


# ── Translation ───────────────────────────────────────────────────────────────

def _translate_to_english(words: list[str]) -> list[str]:
    """
    Detect language and translate non-English words to English.
    Uses langdetect + deep-translator (optional deps — skips gracefully if absent).
    Bounding boxes are NOT changed; only the word strings are replaced.
    """
    try:
        from langdetect import detect, LangDetectException
        text = " ".join(words)
        try:
            lang = detect(text)
        except Exception:
            lang = "en"

        if lang == "en":
            return words

        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=lang, target="en").translate_batch(words)
        return [
            t if (t and isinstance(t, str)) else w
            for t, w in zip(translated, words)
        ]
    except ImportError:
        return words   # langdetect / deep-translator not installed — skip silently
    except Exception:
        return words   # any network/API error — keep original


# ── Model loading ──────────────────────────────────────────────────────────────

def _load() -> bool:
    """Load fine-tuned LayoutLMv3 model if available. Returns True on success."""
    global _processor, _model
    if _model is not None:
        return True
    if not _MODEL_DIR.exists():
        return False
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        _processor = LayoutLMv3Processor.from_pretrained(str(_MODEL_DIR))
        _model     = LayoutLMv3ForTokenClassification.from_pretrained(str(_MODEL_DIR))
        _model.eval()
        print("[LayoutLMv3] Model loaded from", _MODEL_DIR)
        return True
    except Exception as e:
        print(f"[LayoutLMv3] Failed to load model: {e}")
        return False


# ── Bounding box helpers ───────────────────────────────────────────────────────

def _normalize_box(x0: float, top: float, x1: float, bottom: float,
                   img_w: int, img_h: int) -> list[int]:
    """
    Convert pixel-space bbox to LayoutLMv3's expected [0, 1000] integer range.
    Clamps to [0, 1000] to guard against off-image Tesseract boxes.
    """
    return [
        min(max(int(x0   / img_w * 1000), 0), 1000),
        min(max(int(top  / img_h * 1000), 0), 1000),
        min(max(int(x1   / img_w * 1000), 0), 1000),
        min(max(int(bottom / img_h * 1000), 0), 1000),
    ]


# ── Amount parsing (no dependency on donut_extractor) ─────────────────────────

def _parse_amount(raw: str) -> float | None:
    """
    Parse amount strings across locales:
      '€ 348 786,00'  → 348786.0   (European: space-thousands, comma-decimal)
      'RM1.38'        → 1.38
      '$1,234.56'     → 1234.56
    """
    s = re.sub(
        r"[€£¥₹₩₺₽﷼฿৳]|د\.إ|R\$|\bRM\b|\bRp\b|\bRs\.?\b|\bFr\.?\b|\bkr\b",
        "", raw, flags=re.IGNORECASE
    ).strip()
    if re.search(r",\d{1,2}$", s):
        # European format: comma is decimal separator
        s = s.replace(".", "").replace(" ", "").replace(",", ".")
    else:
        s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


# ── Post-processing validation ────────────────────────────────────────────────

def _postprocess_fields(fields: dict, words: list[str]) -> dict:
    """
    Sanitize LayoutLMv3 output with post-extraction rules.

    Rule 1: Reject percentage tokens labeled as TOTAL (e.g. "7%" → not TOTAL).
    Rule 2: Flag integer totals as suspect (no decimal point).
    Rule 3: Decimal repair — if total is integer, search words for ".XX" fragment.
    Rule 4: SUBTOTAL > TOTAL sanity flag.
    """
    total = fields.get("total_amount")

    # Rule 1: Reject % values as TOTAL
    if total is not None:
        total_str = str(total).strip()
        if re.search(r'\d+\.?\d*\s*%$', total_str):
            del fields["total_amount"]
            total = None

    # Rule 2: Flag integer totals (may be truncated decimal)
    if total is not None:
        try:
            total_f = float(total)
            total_str = str(total_f)
            if "." not in total_str or total_str.endswith(".0"):
                fields["_total_is_integer"] = True
        except (ValueError, TypeError):
            pass

    # Rule 3: Decimal repair — search for a ".XX" token near the integer total
    if total is not None and fields.get("_total_is_integer"):
        for word in words:
            if re.match(r'^\.\d{2}$', word.strip()):
                try:
                    repaired = float(str(total) + word.strip())
                    fields["total_amount"] = repaired
                    fields.pop("_total_is_integer", None)
                    total = repaired
                    break
                except ValueError:
                    pass

    # Rule 4: SUBTOTAL > TOTAL sanity flag (informational, does not discard)
    subtotal = fields.get("subtotal_amount")
    if total is not None and subtotal is not None:
        try:
            if float(subtotal) > float(total):
                fields["_subtotal_gt_total_warning"] = True
        except (ValueError, TypeError):
            pass

    return fields


# ── Core inference ─────────────────────────────────────────────────────────────

def _run_inference(pil_image: Image.Image) -> dict | None:
    """
    Run LayoutLMv3 token classification on a PIL image.
    Returns a partial fields dict (vendor/date/total_amount) or None.
    """
    import torch

    try:
        img_w, img_h = pil_image.size
        words_data   = _best_word_data(pil_image.convert("RGB"))
        if not words_data:
            return None

        words = [w["text"] for w in words_data]
        boxes = [
            _normalize_box(w["x0"], w["top"], w["x1"], w["bottom"], img_w, img_h)
            for w in words_data
        ]

        # Detect currency from ORIGINAL words before translation can destroy
        # currency symbols (Google Translate converts € amounts to $ amounts).
        original_text   = " ".join(words)
        detected_currency = detect_currency(text=original_text)

        # Translate non-English invoices (e.g. French) to English
        words = _translate_to_english(words)

        encoding = _processor(
            pil_image.convert("RGB"),
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = _model(**encoding).logits   # (1, seq_len, num_labels)

        preds  = logits.argmax(-1).squeeze().tolist()
        if isinstance(preds, int):
            preds = [preds]

        id2label = _model.config.id2label
        word_ids = encoding.word_ids(batch_index=0)

        # First-subtoken rule: use the label of the first subtoken for each word
        seen: set[int] = set()
        word_labels: dict[int, str] = {}
        for tok_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen:
                continue
            seen.add(word_idx)
            label = id2label.get(preds[tok_idx], "O")
            if label != "O":
                word_labels[word_idx] = label

        if not word_labels:
            return None

        # Group consecutive words that belong to the same field.
        # Only the FIRST B-span per field is kept — this prevents footer
        # mentions (e.g. "Wal-Mart exclusive Eagles CD") from being merged
        # with the header vendor name.
        field_words: dict[str, list[str]] = {}
        active_field: str | None = None
        for word_idx, label in sorted(word_labels.items()):
            field    = _LABEL2FIELD.get(label)
            is_begin = label.startswith("B-")
            if field:
                if is_begin:
                    if field not in field_words:   # only take the first B- span
                        field_words[field] = [words[word_idx]]
                        active_field = field
                    # subsequent B- spans for the same field are ignored
                elif field == active_field:
                    field_words[field].append(words[word_idx])

        if not field_words:
            return None

        fields: dict = {}
        for field, tokens in field_words.items():
            fields[field] = " ".join(tokens)

        if "total_amount" in fields:
            fields["total_amount"] = _parse_amount(fields["total_amount"])
        if "subtotal_amount" in fields:
            fields["subtotal_amount"] = _parse_amount(fields["subtotal_amount"])
        if "date" in fields:
            normalised = _normalise_date(fields["date"])
            if _is_valid_date(normalised):
                fields["date"] = normalised
            else:
                del fields["date"]   # reject stray tokens labeled as DATE

        # Currency priority:
        #   1. LayoutLMv3 B-CURRENCY token — spatially located in the image
        #   2. Pre-translation symbol scan  — fallback when no token was labeled
        #      (Google Translate converts € → $ so we must use the original text)
        if not fields.get("currency") and detected_currency != "UNKNOWN":
            fields["currency"] = detected_currency

        # Apply post-processing validation rules
        fields = _postprocess_fields(fields, words)

        return fields

    except Exception as e:
        print(f"[LayoutLMv3] Inference failed: {e}")
        return None


# ── Merge helper ───────────────────────────────────────────────────────────────

def _merge_with_regex(layoutlm_fields: dict | None, pil_image: Image.Image) -> dict:
    """
    Combine LayoutLMv3 results (vendor/date/total) with regex extraction
    (invoice_number/po_number) derived from Tesseract plain-text output.
    LayoutLMv3 takes priority for the fields it recognises.
    """
    ocr_text     = extract_text_from_pil(pil_image)
    regex_fields = regex_extract(ocr_text)

    merged: dict = {
        "vendor": None, "invoice_number": None,
        "date": None, "po_number": None, "total_amount": None,
        "subtotal_amount": None, "currency": None,
    }

    # Apply regex first, then overwrite with LayoutLMv3 where available
    for key in merged:
        if regex_fields.get(key) is not None:
            merged[key] = regex_fields[key]
    # Normalise the regex-extracted date (2-digit years, European format, etc.)
    if merged.get("date"):
        normalised = _normalise_date(merged["date"])
        merged["date"] = normalised if _is_valid_date(normalised) else None
    if layoutlm_fields:
        for key, val in layoutlm_fields.items():
            if val is not None:
                merged[key] = val

    # For total_amount: regex is keyword-anchored ("TOTAL …") so it's more
    # reliable than LayoutLMv3, which sometimes labels item prices as TOTAL.
    # Use LayoutLMv3 only when the regex found nothing.
    lm_total = layoutlm_fields.get("total_amount") if layoutlm_fields else None
    rx_total = regex_fields.get("total_amount")
    if rx_total is not None:
        merged["total_amount"] = rx_total       # regex wins (keyword-anchored)
    elif lm_total is not None:
        merged["total_amount"] = lm_total       # LM as fallback only

    # Currency: prefer pre-translation detection from _run_inference (captures
    # original € / £ symbols before Google Translate converts them to $).
    # Fall back to scanning the raw OCR text if LayoutLMv3 didn't provide one.
    if layoutlm_fields and layoutlm_fields.get("currency"):
        merged["currency"] = layoutlm_fields["currency"]
    else:
        currency = detect_currency(text=ocr_text)
        if currency != "UNKNOWN":
            merged["currency"] = currency

    # Clean up internal sentinel keys from _postprocess_fields
    merged.pop("_total_is_integer", None)
    merged.pop("_subtotal_gt_total_warning", None)

    # Add ISO-formatted date as a separate key for downstream use
    if merged.get("date"):
        m = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', merged["date"])
        if m:
            merged["date_iso"] = f"{m.group(3)}-{m.group(1)}-{m.group(2)}"

    return merged


# ── Public API ────────────────────────────────────────────────────────────────

def extract_fields_with_layoutlm(image_path: str) -> dict | None:
    """
    Extract invoice fields from an image file using the two-stage pipeline.
    Returns a fields dict or None (pipeline falls through to Groq).
    """
    if not _load():
        return None
    try:
        img           = Image.open(image_path).convert("RGB")
        layoutlm_out  = _run_inference(img)
        merged        = _merge_with_regex(layoutlm_out, img)

        if any(v is not None for v in merged.values()):
            return merged
        return None
    except Exception as e:
        print(f"[LayoutLMv3] Image load failed: {e}")
        return None


def extract_fields_from_pdf_with_layoutlm(pdf_path: str) -> dict | None:
    """
    Extract invoice fields from a PDF using the two-stage pipeline.
    Renders up to 2 pages to images, runs LayoutLMv3 on each, merges results.
    Returns a fields dict or None.
    """
    if not _load():
        return None
    try:
        import pdfplumber

        merged: dict = {
            "vendor": None, "invoice_number": None,
            "date": None, "po_number": None, "total_amount": None,
        }

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:2]:
                img  = page.to_image(resolution=200).original  # match training DPI
                lm   = _run_inference(img)
                mix  = _merge_with_regex(lm, img)
                for key in merged:
                    if merged[key] is None and mix.get(key) is not None:
                        merged[key] = mix[key]

        if any(v is not None for v in merged.values()):
            return merged
        return None
    except Exception as e:
        print(f"[LayoutLMv3] PDF extraction failed: {e}")
        return None


def is_available() -> bool:
    """True if the fine-tuned LayoutLMv3 model is present and loadable."""
    return _MODEL_DIR.exists()
