# GOT-OCR2.0 — Vision-Language Model OCR (Tier 0 for image uploads)
#
# GOT-OCR2.0 (580M params) is a VLM that understands document *context*,
# unlike Tesseract which just pattern-matches pixel shapes. It:
#   - Reads tables, two-column layouts, and curved/skewed text correctly
#   - Outputs clean structured text (no background noise garbage)
#   - Runs on CPU (slow ~10-30s per image, but accurate)
#
# Model is downloaded from HuggingFace on first use (~1.5GB, cached locally).
# Falls back silently if model unavailable or inference fails.
#
# Reference: https://huggingface.co/ucaslcl/GOT-OCR2_0

from __future__ import annotations
import os
import tempfile
from pathlib import Path

# Lazy imports — only load the heavy model when first needed
_model = None
_tokenizer = None
_load_error: str | None = None   # set if model failed to load, avoids retry


def _load_model():
    """Load GOT-OCR2.0 model and tokenizer (once, then cached in module globals)."""
    global _model, _tokenizer, _load_error

    if _model is not None:
        return True
    if _load_error is not None:
        return False

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        model_name = "ucaslcl/GOT-OCR2_0"
        print(f"[GOT-OCR] Loading model {model_name} (first run — may take a minute)...")

        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        _model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            pad_token_id=151643,
        )
        _model = _model.eval()

        # Use GPU if available, otherwise CPU
        if torch.cuda.is_available():
            _model = _model.cuda()
            print("[GOT-OCR] Running on GPU")
        else:
            print("[GOT-OCR] Running on CPU (expect ~15-30s per image)")

        return True

    except Exception as e:
        _load_error = str(e)
        print(f"[GOT-OCR] Model load failed: {e}")
        return False


def extract_text_with_got_ocr(image_path: str) -> str | None:
    """
    Run GOT-OCR2.0 on an image file.

    Uses 'ocr' mode — plain text output preserving spatial layout.
    Returns None if:
      - model failed to load
      - inference failed
      - output is < 30 chars (likely a blank/noise image)

    Falls back silently so the pipeline continues to Tesseract.
    """
    if not _load_model():
        return None

    try:
        import torch

        with torch.no_grad():
            # ocr_type='ocr'     → plain text, layout-aware
            # ocr_type='format'  → markdown with tables (heavier, slower)
            result = _model.chat(
                _tokenizer,
                image_path,
                ocr_type="ocr",
            )

        if not result or not isinstance(result, str):
            return None

        text = result.strip()
        return text if len(text) >= 30 else None

    except Exception as e:
        print(f"[GOT-OCR] Inference failed: {e}")
        return None


def extract_text_with_got_ocr_pil(pil_image) -> str | None:
    """
    Run GOT-OCR2.0 on a PIL image (used for scanned PDF pages).
    Saves to a temp file first since GOT-OCR expects a file path.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        pil_image.save(tmp_path, format="PNG")
        result = extract_text_with_got_ocr(tmp_path)
        return result
    except Exception:
        return None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def is_got_ocr_available() -> bool:
    """Returns True if GOT-OCR model loaded successfully."""
    return _load_model()
