"""
Run all invoices in data/real_invoices/ through Groq and cache
(image, fields) pairs to data/extraction_cache/ for Donut fine-tuning.

Usage:
    python scripts/label_invoices.py
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pathlib import Path
from PIL import Image

# Must set API key before import — set XAI_API_KEY in your environment or .env file
API_KEY = os.environ.get("XAI_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("XAI_API_KEY is not set. Add it to your .env file or environment.")
os.environ["XAI_API_KEY"] = API_KEY

from app.extraction.groq_ocr import (
    extract_fields_with_groq,
    extract_fields_from_pdf_with_groq,
    _CACHE_DIR,
)

INVOICE_DIR = Path(__file__).resolve().parent.parent / "data" / "real_invoices"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def run():
    files = [f for f in INVOICE_DIR.iterdir() if f.is_file()]
    print(f"Found {len(files)} invoices in {INVOICE_DIR}\n")

    ok = 0
    for f in files:
        ext = f.suffix.lower()
        print(f"  Processing {f.name} ...", end=" ", flush=True)
        try:
            if ext == ".pdf":
                fields = extract_fields_from_pdf_with_groq(str(f))
            elif ext in IMAGE_EXTS:
                fields = extract_fields_with_groq(str(f))
            else:
                print("skipped (unsupported type)")
                continue

            if fields and any(v is not None for v in fields.values()):
                print(f"OK -> vendor={fields.get('vendor')!r}  "
                      f"total={fields.get('total_amount')}  "
                      f"inv#={fields.get('invoice_number')!r}")
                ok += 1
            else:
                print("no fields extracted")
        except Exception as e:
            print(f"ERROR: {e}")

    cached = list(_CACHE_DIR.glob("*.json"))
    print(f"\nOK {ok}/{len(files)} invoices labeled")
    print(f"OK {len(cached)} samples now in {_CACHE_DIR}")

if __name__ == "__main__":
    run()
