# Tesseract OCR wrapper (Tier 2 — fallback for scanned docs and images)
# Mirrors LiteParse's TesseractEngine: word-level bbox extraction, confidence
# filtering, and spatial grid projection of OCR results.
import os
import statistics
import re
import pytesseract
from PIL import Image
from .preprocessor import preprocess_image

# ── Auto-detect Tesseract on Windows ─────────────────────────────────────────
_WINDOWS_PATHS = [
    r"D:\Extras\Tesseract\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]
for _p in _WINDOWS_PATHS:
    if os.path.isfile(_p):
        pytesseract.pytesseract.tesseract_cmd = _p
        break

# PSM modes to try in order:
#   6 = single uniform block (good for structured docs)
#   4 = single column (good for receipts)
#   3 = fully automatic (Tesseract default)
_PSM_MODES = [6, 4, 3]

# LiteParse confidence threshold — discard words below 30% confidence
_MIN_CONFIDENCE = 30   # Tesseract returns 0–100


# ── Word-level data extraction (mirrors LiteParse TesseractEngine.recognize) ──

def _get_word_data(img: Image.Image, psm: int) -> list[dict]:
    """
    Run Tesseract with image_to_data and return a list of word dicts:
      { text, x0, x1, top, bottom, confidence }
    Filters out empty text and confidence < _MIN_CONFIDENCE.
    """
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        img, config=config,
        output_type=pytesseract.Output.DICT
    )

    words = []
    n = len(data["text"])
    for i in range(n):
        text = str(data["text"][i]).strip()
        conf = int(data["conf"][i])
        if not text or conf < _MIN_CONFIDENCE:
            continue
        x    = data["left"][i]
        y    = data["top"][i]
        w    = data["width"][i]
        h    = data["height"][i]
        words.append({
            "text":  text,
            "x0":    float(x),
            "x1":    float(x + w),
            "top":   float(y),
            "bottom": float(y + h),
            "conf":  conf,
        })
    return words


def _best_word_data(img: Image.Image) -> list[dict]:
    """
    Try PSM 6, 4, 3 and return the word list with the most words.
    Mirrors LiteParse's multi-PSM approach.
    """
    best_words: list[dict] = []
    for psm in _PSM_MODES:
        try:
            words = _get_word_data(img, psm)
            if len(words) > len(best_words):
                best_words = words
        except Exception:
            continue
    return best_words


# ── Spatial reconstruction from OCR words ─────────────────────────────────────
# Mirrors LiteParse's grid projection applied to OCR results:
# group words into lines by Y-proximity, then join with gap-proportional spacing.

def _ocr_words_to_text(words: list[dict]) -> str:
    """
    Convert word-level OCR results into spatially reconstructed text.
    Uses the same line-grouping + gap-based joining as spatial_pdf_parser.py.
    """
    if not words:
        return ""

    # Sort top-to-bottom, left-to-right
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    # Estimate median char width
    char_widths = [(w["x1"] - w["x0"]) / max(len(w["text"]), 1) for w in words]
    char_w = max(statistics.median(char_widths), 3.0)

    # Estimate median line height for grouping threshold
    heights = [w["bottom"] - w["top"] for w in words]
    median_h  = statistics.median(heights)
    threshold = median_h * 0.6

    # Group into lines
    lines: list[list[dict]] = []
    current   = [words[0]]
    cur_top   = words[0]["top"]
    cur_bot   = words[0]["bottom"]

    for word in words[1:]:
        overlap = min(cur_bot, word["bottom"]) - max(cur_top, word["top"])
        if overlap >= threshold or abs(word["top"] - cur_top) < threshold:
            current.append(word)
            cur_top = min(cur_top, word["top"])
            cur_bot = max(cur_bot, word["bottom"])
        else:
            lines.append(sorted(current, key=lambda w: w["x0"]))
            current = [word]
            cur_top = word["top"]
            cur_bot = word["bottom"]
    if current:
        lines.append(sorted(current, key=lambda w: w["x0"]))

    # Join each line using gap-proportional spacing (same as spatial_pdf_parser)
    text_lines = []
    for line in lines:
        parts = [line[0]["text"]]
        for i in range(1, len(line)):
            gap = line[i]["x0"] - line[i-1]["x1"]
            if gap <= 0:
                spaces = ""
            elif gap < char_w:
                spaces = " "
            elif gap < char_w * 3:
                spaces = " "
            else:
                spaces = " " * min(int(gap / char_w), 8)
            parts.append(spaces + line[i]["text"])
        text_lines.append("".join(parts))

    # Clean up (mirrors cleanText.ts)
    cleaned = []
    for line in text_lines:
        line = re.sub(r' {5,}', '  ', line).rstrip().replace('\x00', ' ')
        cleaned.append(line)

    # Trim leading/trailing blank lines
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return "\n".join(cleaned)


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_text_from_image(image_path: str, preprocess: bool = True) -> str:
    """
    Run Tesseract OCR on an image file.
    Uses word-level extraction + spatial reconstruction (LiteParse-equivalent).
    Falls back to plain string output if word-level extraction fails.
    """
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if preprocess:
        img = preprocess_image(img)

    words = _best_word_data(img)
    if words:
        return _ocr_words_to_text(words)

    # Fallback: plain string (original behaviour)
    return _best_ocr_text_plain(img)


def extract_text_from_pil(pil_image: Image.Image, preprocess: bool = False) -> str:
    """
    Run Tesseract on an already-loaded PIL image (used for scanned PDFs
    converted page-by-page to images).
    """
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    if preprocess:
        pil_image = preprocess_image(pil_image)

    words = _best_word_data(pil_image)
    if words:
        return _ocr_words_to_text(words)

    return _best_ocr_text_plain(pil_image)


def extract_with_confidence(image_path: str, min_confidence: int = 50) -> str:
    """
    Extract text with per-word confidence filtering (higher threshold version).
    """
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img = preprocess_image(img)

    words = _get_word_data(img, psm=6)
    words = [w for w in words if w["conf"] >= min_confidence]
    return _ocr_words_to_text(words) if words else ""


# ── Plain-string fallback (original _best_ocr_text) ──────────────────────────

def _run_tesseract_plain(img: Image.Image, psm: int) -> str:
    config = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(img, config=config).strip()


def _best_ocr_text_plain(img: Image.Image) -> str:
    best_text = ""
    best_count = 0
    for psm in _PSM_MODES:
        try:
            text = _run_tesseract_plain(img, psm)
            count = len(text.split())
            if count > best_count:
                best_count = count
                best_text = text
        except Exception:
            continue
    return best_text
