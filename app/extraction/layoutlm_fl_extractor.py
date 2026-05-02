# Drop-in extractor that uses the FL-updated (LoRA-merged) LayoutLMv3 model.
#
# After federated training completes, layoutlm_fl_coordinator.py merges the
# LoRA adapter weights back into the base LayoutLMv3 and saves the result to
# data/layoutlm_lora_merged/.  This file exposes an interface identical to
# layoutlm_extractor.py but loads from that merged directory.
#
# If no merged model exists, falls back silently (returns None) — the caller
# (main.py) handles the fallback to the base layoutlm_extractor or Groq OCR.
#
# _run_inference() is copied verbatim from layoutlm_extractor.py so the two
# paths remain in sync — any fix to one should be applied to the other.
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from app.extraction.ocr_engine import _best_word_data, extract_text_from_pil
from app.extraction.field_extractor import extract_fields as regex_extract
from app.extraction.currency_detector import detect_currency

_BASE            = Path(__file__).resolve().parent.parent.parent
_FL_MERGED_DIR   = _BASE / "data" / "layoutlm_lora_merged"
_FL_META_PATH    = _BASE / "data" / "layoutlm_lora" / "fl_lora_metadata.json"

# Flag file that main.py / dashboard checks to enable/disable the FL model
_FL_ACTIVE_FLAG  = _BASE / "data" / "fl_model" / "use_fl_layoutlm.flag"

# Lazy-loaded — only imported when actually needed
_processor = None
_model     = None

_LABEL2FIELD: dict[str, str] = {
    "B-VENDOR":     "vendor",         "I-VENDOR":     "vendor",
    "B-DATE":       "date",           "I-DATE":       "date",
    "B-TOTAL":      "total_amount",   "I-TOTAL":      "total_amount",
    "B-INVOICE_NO": "invoice_number", "I-INVOICE_NO": "invoice_number",
    "B-PO_NO":      "po_number",      "I-PO_NO":      "po_number",
    "B-CURRENCY":   "currency",       "I-CURRENCY":   "currency",
}


# ── Status helpers ─────────────────────────────────────────────────────────────

def is_fl_model_available() -> bool:
    """True if a merged FL-LoRA model exists and can be loaded."""
    return (
        _FL_MERGED_DIR.exists()
        and (_FL_MERGED_DIR / "config.json").exists()
        and any(_FL_MERGED_DIR.glob("model.safetensors"))
    )


def is_fl_model_active() -> bool:
    """True if the dashboard has toggled the FL model ON."""
    return _FL_ACTIVE_FLAG.exists()


def activate_fl_model() -> None:
    """Create the flag file — routes extraction through FL-LoRA model."""
    _FL_ACTIVE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    _FL_ACTIVE_FLAG.touch()


def deactivate_fl_model() -> None:
    """Remove the flag file — routes extraction through base LayoutLMv3."""
    _FL_ACTIVE_FLAG.unlink(missing_ok=True)


def get_fl_model_metadata() -> dict | None:
    """
    Read fl_lora_metadata.json — returns run_id, n_rounds, final_macro_f1,
    per_class_f1_final, epsilon, and full privacy_report.  Returns None if
    the metadata file does not exist.
    """
    if not _FL_META_PATH.exists():
        return None
    try:
        return json.loads(_FL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Model loading ──────────────────────────────────────────────────────────────

def _load() -> bool:
    """Load the merged FL-LoRA LayoutLMv3 model if available. Returns True on success."""
    global _processor, _model
    if _model is not None:
        return True
    if not is_fl_model_available():
        return False
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        _processor = LayoutLMv3Processor.from_pretrained(str(_FL_MERGED_DIR))
        _model     = LayoutLMv3ForTokenClassification.from_pretrained(str(_FL_MERGED_DIR))
        _model.eval()
        print("[FL-LayoutLMv3] Merged FL model loaded from", _FL_MERGED_DIR)
        return True
    except Exception as e:
        print(f"[FL-LayoutLMv3] Failed to load model: {e}")
        return False


def reload_fl_model() -> bool:
    """Force a model reload — call after training completes and merge is done."""
    global _processor, _model
    _processor = None
    _model     = None
    return _load()


# ── Bounding box / date / amount helpers (identical to layoutlm_extractor.py) ─

def _normalize_box(x0: float, top: float, x1: float, bottom: float,
                   img_w: int, img_h: int) -> list[int]:
    return [
        min(max(int(x0     / img_w * 1000), 0), 1000),
        min(max(int(top    / img_h * 1000), 0), 1000),
        min(max(int(x1     / img_w * 1000), 0), 1000),
        min(max(int(bottom / img_h * 1000), 0), 1000),
    ]


def _normalise_date(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw
    s = re.sub(r'[-.]', '/', raw)
    m = re.match(r'^(\d{4})(\d{2})(\d{2})$', raw)
    if m:
        return f"{m.group(2)}/{m.group(3)}/{m.group(1)}"
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


def _parse_amount(raw: str) -> float | None:
    s = re.sub(
        r"[€£¥₹₩₺₽﷼฿৳]|د\.إ|R\$|\bRM\b|\bRp\b|\bRs\.?\b|\bFr\.?\b|\bkr\b",
        "", raw, flags=re.IGNORECASE
    ).strip()
    if re.search(r",\d{1,2}$", s):
        s = s.replace(".", "").replace(" ", "").replace(",", ".")
    else:
        s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def _translate_to_english(words: list[str]) -> list[str]:
    try:
        from langdetect import detect
        text = " ".join(words)
        try:
            lang = detect(text)
        except Exception:
            lang = "en"
        if lang == "en":
            return words
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=lang, target="en").translate_batch(words)
        return [t if (t and isinstance(t, str)) else w for t, w in zip(translated, words)]
    except ImportError:
        return words
    except Exception:
        return words


# ── Core inference (verbatim from layoutlm_extractor.py) ──────────────────────

def _run_inference(pil_image: Image.Image) -> dict | None:
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

        original_text     = " ".join(words)
        detected_currency = detect_currency(text=original_text)

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
            logits = _model(**encoding).logits

        preds    = logits.argmax(-1).squeeze().tolist()
        if isinstance(preds, int):
            preds = [preds]

        id2label = _model.config.id2label
        word_ids = encoding.word_ids(batch_index=0)

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

        field_words: dict[str, list[str]] = {}
        for word_idx, label in sorted(word_labels.items()):
            field = _LABEL2FIELD.get(label)
            if field:
                field_words.setdefault(field, []).append(words[word_idx])

        if not field_words:
            return None

        fields: dict = {}
        for field, tokens in field_words.items():
            fields[field] = " ".join(tokens)

        if "total_amount" in fields:
            fields["total_amount"] = _parse_amount(fields["total_amount"])
        if "date" in fields:
            fields["date"] = _normalise_date(fields["date"])

        if not fields.get("currency") and detected_currency != "UNKNOWN":
            fields["currency"] = detected_currency

        # Tag the result so callers know which model produced it
        fields["_extraction_method"] = "layoutlm-fl-lora"
        return fields

    except Exception as e:
        print(f"[FL-LayoutLMv3] Inference failed: {e}")
        return None


def _merge_with_regex(layoutlm_fields: dict | None, pil_image: Image.Image) -> dict:
    ocr_text     = extract_text_from_pil(pil_image)
    regex_fields = regex_extract(ocr_text)

    merged: dict = {
        "vendor": None, "invoice_number": None,
        "date": None, "po_number": None, "total_amount": None,
        "currency": None,
    }

    for key in merged:
        if regex_fields.get(key) is not None:
            merged[key] = regex_fields[key]
    if layoutlm_fields:
        for key, val in layoutlm_fields.items():
            if val is not None and not key.startswith("_"):
                merged[key] = val

    if layoutlm_fields and layoutlm_fields.get("currency"):
        merged["currency"] = layoutlm_fields["currency"]
    else:
        currency = detect_currency(text=ocr_text)
        if currency != "UNKNOWN":
            merged["currency"] = currency

    merged["extraction_method"] = "layoutlm-fl-lora"
    return merged


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_fields_with_fl_layoutlm(image_path: str) -> dict | None:
    """
    Extract invoice fields using the FL-updated LoRA-merged LayoutLMv3 model.
    Interface identical to extract_fields_with_layoutlm() in layoutlm_extractor.py.
    Returns None if the FL model is unavailable (caller falls through to base model).
    """
    if not _load():
        return None
    try:
        img          = Image.open(image_path).convert("RGB")
        layoutlm_out = _run_inference(img)
        merged       = _merge_with_regex(layoutlm_out, img)

        if any(v is not None for k, v in merged.items() if not k.startswith("extraction")):
            return merged
        return None
    except Exception as e:
        print(f"[FL-LayoutLMv3] Image load failed: {e}")
        return None


def extract_fields_from_pil_with_fl_layoutlm(pil_image: Image.Image) -> dict | None:
    """Variant that accepts a PIL image directly."""
    if not _load():
        return None
    try:
        layoutlm_out = _run_inference(pil_image)
        merged       = _merge_with_regex(layoutlm_out, pil_image)

        if any(v is not None for k, v in merged.items() if not k.startswith("extraction")):
            return merged
        return None
    except Exception as e:
        print(f"[FL-LayoutLMv3] Inference failed: {e}")
        return None


def extract_fields_from_pdf_with_fl_layoutlm(pdf_path: str) -> dict | None:
    """Extract from a PDF — renders up to 2 pages, runs inference on each."""
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
                img = page.to_image(resolution=150).original
                lm  = _run_inference(img)
                mix = _merge_with_regex(lm, img)
                for key in merged:
                    if merged[key] is None and mix.get(key) is not None:
                        merged[key] = mix[key]

        if any(v is not None for v in merged.values()):
            merged["extraction_method"] = "layoutlm-fl-lora"
            return merged
        return None
    except Exception as e:
        print(f"[FL-LayoutLMv3] PDF extraction failed: {e}")
        return None
