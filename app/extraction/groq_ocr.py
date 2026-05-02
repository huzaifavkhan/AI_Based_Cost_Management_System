# Groq Vision OCR — Tier 0 extraction for images and PDFs.
#
# Sends document pages to Groq's hosted LLaMA-4 vision model, which understands
# invoice layout, context, and field relationships — far superior to Tesseract.
#
# API key is read from the XAI_API_KEY environment variable.
# Falls back silently if the API is unavailable or returns incomplete data.

from __future__ import annotations
import os
import base64
import hashlib
import json
import re
import tempfile
from pathlib import Path

import requests
from PIL import Image

# Load .env file if present (so XAI_API_KEY can be set there instead of the shell)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed — fall back to shell env var

# ── Config ────────────────────────────────────────────────────────────────────
_API_KEY   = os.environ.get("XAI_API_KEY", "")
_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"   # fast, vision-capable
_TIMEOUT   = 30   # seconds

# ── Label cache — every successful Groq extraction is saved as a Donut training sample ──
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "extraction_cache"


def _cache_label(image_b64: str, fields: dict) -> None:
    """
    Save (image, fields) pair to disk for Donut fine-tuning.
    Key is an MD5 of the first 200 chars of the base64 string — unique per image,
    cheap to compute, collision-proof at this scale.
    """
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key      = hashlib.md5(image_b64.encode()).hexdigest()
        img_path = _CACHE_DIR / f"{key}.jpg"
        lbl_path = _CACHE_DIR / f"{key}.json"
        if lbl_path.exists():
            return   # already cached — skip
        img_bytes = base64.b64decode(image_b64)
        img_path.write_bytes(img_bytes)
        lbl_path.write_text(json.dumps({"fields": fields}, indent=2))
    except Exception:
        pass   # cache failure must never break the extraction pipeline

# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert invoice and receipt data extraction system.
Extract the following fields from the document image and return ONLY valid JSON.
No explanation, no markdown, no code blocks — just the raw JSON object.

Fields to extract:
{
  "invoice_number": "the unique document transaction identifier. Priority order: (1) Transaction ID, (2) Invoice Number / Invoice No / Invoice ID / Inv No, (3) Order ID / Order Number, (4) Receipt Number / Receipt No / Receipt #, (5) GST Invoice No / GST ID / Tax Invoice Number, (6) Reference Number / Ref No, (7) Confirmation Number, (8) Voucher Number, (9) Booking ID, (10) Serial Number / S/N, (11) C01 / Register No or any alphanumeric code printed as a document identifier. NEVER use a card number or account number — reject any value that contains three or more consecutive X characters (e.g. XXXXXXXXXXXX0041 is a masked card number, not a transaction ID). If none of the above labels exist, use null (string or null)",
  "vendor": "the seller, company, or store name (string or null)",
  "date": "the invoice/receipt/bill date in its original format (string or null)",
  "po_number": "purchase order number if present (string or null)",
  "total_amount": "the final total amount as a number without currency symbol (float or null)",
  "total_raw": "the total amount exactly as it appears on the document including currency symbol (e.g. '€ 348,786.00', 'RM1.38', 'PKR 5,000', '$120.00') — null if not found",
  "currency_id": "ISO 4217 currency code (e.g. 'USD', 'PKR', 'EUR', 'GBP', 'AED', 'MYR', 'INR', 'SAR'). Look for: explicit currency field, symbol next to amount, document header, or infer from country/language of the document. ALWAYS try to identify — only return null if truly impossible"
}

Rules:
- For total_amount, use the FINAL total (after tax), not subtotal
- For total_raw, copy the exact text as printed including currency symbol
- For currency_id: French/European docs → EUR, Malaysian docs with RM → MYR, Pakistani docs → PKR, Arabic/UAE → AED
- For vendor, use the SELLER name, not the buyer/bill-to name
- If a field is not present, use null
- Return ONLY the JSON object, nothing else"""


# ── Image encoding ────────────────────────────────────────────────────────────

def _encode_image_to_base64(image_path: str) -> str | None:
    """Read image file and encode as base64 string."""
    try:
        # Resize if too large (Groq has token limits for images)
        img = Image.open(image_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Cap at 1568px on longest side (Groq vision optimal)
        max_side = 1568
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Save to bytes
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def _encode_pil_to_base64(pil_image: Image.Image) -> str | None:
    """Encode a PIL image to base64 JPEG string."""
    try:
        import io
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")

        max_side = 1568
        w, h = pil_image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


# ── API call ──────────────────────────────────────────────────────────────────

def _call_groq_vision(image_b64: str) -> dict | None:
    """
    Send image to Groq vision API and parse the JSON response.
    Returns a dict with invoice fields, or None on failure.
    """
    if not _API_KEY:
        return None

    payload = {
        "model": _MODEL,
        "messages": [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract all invoice fields from this document.",
                    },
                ],
            },
        ],
        "temperature": 0.0,   # deterministic output
        "max_tokens": 512,
    }

    try:
        resp = requests.post(
            _API_URL,
            headers={
                "Authorization": f"Bearer {_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        result  = _parse_json_response(content)
        if result:
            _cache_label(image_b64, result)   # save as Donut training sample
        return result
    except Exception as e:
        print(f"[Groq OCR] API call failed: {e}")
        return None


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from model response, handling minor formatting issues."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find the JSON object
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None

    try:
        from app.extraction.currency_detector import detect_currency
        data = json.loads(m.group(0))

        # Normalize total_amount to float
        if data.get("total_amount") is not None:
            try:
                data["total_amount"] = float(str(data["total_amount"]).replace(",", ""))
            except (ValueError, TypeError):
                data["total_amount"] = None

        # Reject masked card numbers (3+ consecutive X's) as invoice_number
        inv = data.get("invoice_number")
        if inv and re.search(r'X{3,}', str(inv), re.IGNORECASE):
            data["invoice_number"] = None

        # Resolve currency using all available signals:
        # 1. currency_id (ISO code Groq extracted)
        # 2. total_raw   (raw amount string with symbol e.g. "€ 348,786.00", "RM1.38")
        # 3. vendor/address text for context inference
        currency_id = data.pop("currency_id", None)
        total_raw   = data.pop("total_raw", None)     # remove helper field from output
        context_text = " ".join(filter(None, [
            total_raw,
            str(data.get("vendor") or ""),
            str(data.get("invoice_number") or ""),
        ]))
        symbol = detect_currency(
            text=context_text,
            currency_id=currency_id or "",
            amount_context=total_raw or "",
        )
        data["currency"] = symbol if symbol != "UNKNOWN" else None

        return data
    except json.JSONDecodeError:
        return None


# ── Validation ────────────────────────────────────────────────────────────────

def _is_useful(fields: dict | None) -> bool:
    """
    Returns True if the Groq response contains at least one non-null field.
    Prevents passing empty dicts downstream.
    """
    if not fields:
        return False
    return any(v is not None for v in fields.values())


# ── Public API ────────────────────────────────────────────────────────────────

def extract_fields_with_groq(image_path: str) -> dict | None:
    """
    Run Groq vision OCR on an image file.
    Returns a fields dict {invoice_number, vendor, date, po_number, total_amount}
    or None if extraction failed / returned no useful data.
    """
    b64 = _encode_image_to_base64(image_path)
    if not b64:
        return None
    result = _call_groq_vision(b64)
    return result if _is_useful(result) else None


def extract_fields_from_pil_with_groq(pil_image: Image.Image) -> dict | None:
    """
    Run Groq vision OCR on a PIL image (used for scanned PDF pages).
    """
    b64 = _encode_pil_to_base64(pil_image)
    if not b64:
        return None
    result = _call_groq_vision(b64)
    return result if _is_useful(result) else None


def extract_fields_from_pdf_with_groq(pdf_path: str) -> dict | None:
    """
    Extract fields from a PDF by rendering page 1 as an image and sending to Groq.
    For multi-page PDFs, merges fields from pages 1-2 (first page has header,
    second may have totals).
    Returns merged fields dict or None.
    """
    try:
        import pdfplumber
        merged: dict = {
            "invoice_number": None,
            "vendor": None,
            "date": None,
            "po_number": None,
            "total_amount": None,
        }
        found_any = False

        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = pdf.pages[:2]   # first 2 pages only
            for page in pages_to_check:
                img = page.to_image(resolution=150).original
                fields = extract_fields_from_pil_with_groq(img)
                if fields:
                    found_any = True
                    # Merge: prefer first non-null value for each field
                    for key in merged:
                        if merged[key] is None and fields.get(key) is not None:
                            merged[key] = fields[key]

        return merged if found_any else None
    except Exception as e:
        print(f"[Groq OCR] PDF extraction failed: {e}")
        return None
