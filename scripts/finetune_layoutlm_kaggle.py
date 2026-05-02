"""
Fine-tune LayoutLMv3 on SROIE + French FACTURE dataset.

KEY DESIGN DECISIONS (for generalization to unseen invoices):

  Option A — Train/inference input parity:
    Always use Tesseract for words & boxes.
    TSV / SROIE box files are used ONLY as label oracles via IoU overlap.
    This means training and inference see the same noisy Tesseract output,
    so the model is never surprised by OCR tokenization at inference time.

  Option B — OCR noise augmentation:
    During training, random perturbations simulate Tesseract errors:
      • Character substitution (0→O, 1→l, rn→m)
      • Bounding box coordinate jitter
      • Token splitting ("46,00" → ["46,", "00"])
      • Random O-token drops
    This makes the model robust to OCR variance on genuinely unseen invoices.

Dataset layout:
  train/
    img/       — FACTU*.jpg + X000*.jpg + *.pdf
    entities/  — FACTU*.xml + X000*.txt
    box/       — FACTU*.tsv  (French ground-truth word-level labels)
    box/       — X000*.txt   (SROIE pre-annotated bounding boxes)

Label schema:
  O=0,
  B-VENDOR=1,     I-VENDOR=2,
  B-DATE=3,       I-DATE=4,
  B-TOTAL=5,      I-TOTAL=6,
  B-SUBTOTAL=7,   I-SUBTOTAL=8,
  B-INVOICE_NO=9, I-INVOICE_NO=10,
  B-PO_NO=11,     I-PO_NO=12,
  B-CURRENCY=13,  I-CURRENCY=14

Kaggle setup:
  GPU T4 x1 → Internet ON → paste this script → Run All
  Best checkpoint → /kaggle/working/layoutlm_invoice_best/
  Final model    → /kaggle/working/layoutlm_invoice/

Reproducibility:
  Set RANDOM_SEED below to any integer for deterministic splits and
  augmentation.  Default is 42.
"""

import os, json, re, random, hashlib, csv, threading
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict as _defaultdict
from pathlib import Path

from seqeval.metrics import classification_report, f1_score as seq_f1

# ── Install dependencies ───────────────────────────────────────────────────────
ret = os.system("apt-get install -q -y tesseract-ocr tesseract-ocr-fra poppler-utils")
print(f"[setup] apt-get exit code: {ret}")
ret = os.system("pip install -q pytesseract seqeval deep-translator pdf2image opencv-python-headless")
print(f"[setup] pip exit code: {ret}")

import shutil as _shutil
if _shutil.which("pdftoppm") is None:
    print("WARNING: pdftoppm not found — PDF conversion will fail.")
else:
    print(f"[setup] poppler OK: {_shutil.which('pdftoppm')}")

import torch
import numpy as np
import cv2
import pytesseract
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    get_cosine_schedule_with_warmup,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
_INPUT    = Path("/kaggle/input")
_img_hits = list(_INPUT.rglob("img"))
if _img_hits:
    TRAIN_DIR = _img_hits[0].parent
    print(f"[setup] Dataset found at: {TRAIN_DIR}")
else:
    TRAIN_DIR = Path("/kaggle/input/datasets/alisaeed11/fypdata/train")
    print(f"[setup] WARNING: img/ not found — falling back to {TRAIN_DIR}")

IMG_DIR                = TRAIN_DIR / "img"
ENTITY_DIR             = TRAIN_DIR / "entities"
TSV_DIR                = TRAIN_DIR / "box"
BOX_DIR                = TRAIN_DIR / "box"
OUTPUT_DIR             = Path("/kaggle/working/layoutlm_invoice")
BEST_DIR               = Path("/kaggle/working/layoutlm_invoice_best")
PDF_RENDER_DIR         = Path("/kaggle/working/pdf_rendered")
TRANSLATION_CACHE_PATH = Path("/kaggle/working/translation_cache.json")

PDF_RENDER_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "microsoft/layoutlmv3-base"

# ── Reproducibility ────────────────────────────────────────────────────────────
# FIX 1: Default to 42 instead of None — deterministic by default.
# Change to None only if you explicitly want a fresh random seed each run.
RANDOM_SEED: int | None = 42

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"[setup] Random seed fixed to {RANDOM_SEED}")
else:
    print("[setup] NOTE: RANDOM_SEED=None — runs will differ.")

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS          = 20
LR              = 1e-5
BATCH_SIZE      = 4
MAX_LENGTH      = 512
PDF_DPI         = 200
EARLY_STOP_PAT  = 4
MAX_IMG_HEIGHT  = 2000
# FIX 3: Gradient accumulation — effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 16
GRAD_ACCUM_STEPS = 4
# FIX 2: DataLoader workers — set to 2 so GPU stays fed while CPU loads next batch
DATALOADER_WORKERS = 2

# ── Translation config ─────────────────────────────────────────────────────────
USE_TRANSLATION        = False
TRANSLATE_CHUNK_SIZE   = 20
TRANSLATE_SLEEP_S      = 1.0
TRANSLATE_MAX_RETRIES  = 3

if USE_TRANSLATION:
    print(
        "[WARNING] USE_TRANSLATION=True.  With large French datasets this "
        "will stall training for hours.  Set USE_TRANSLATION=False unless "
        "you have <200 samples and a stable connection."
    )

# ── Label schema ───────────────────────────────────────────────────────────────
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
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

_TOTAL_KEYWORDS    = {"total", "totaal", "montant", "ttc", "amount", "due",
                      "payable", "balance", "grand", "net", "owing"}
_SUBTOTAL_KEYWORDS = {"subtotal", "sub", "subtotaal", "untaxed",
                      "pre-tax", "pretax", "excl", "ht", "nettotal"}

_TSV_LABEL_MAP = {
    "NUMBER":           "INVOICE_NO",
    "PO_NUMBER":        "PO_NO",
    "INVOICE_DATE":     "DATE",
    "SUPPLIER":         "VENDOR",
    "TOTAL_AMOUNT":     "TOTAL",
    "TOTAL_UNTAXED":    "SUBTOTAL",
    "ADDRESS":          "O",
    "TAX_AMOUNT":       "O",
    "INVOICE_DUE_DATE": "O",
    "LINE/DESCRIPTION": "O",
    "LINE/QUANTITY":    "O",
    "LINE/PRICE":       "O",
    "LINE/SUB_TOTAL":   "O",
    "LINE/TAX":         "O",
    "LINE/UOM":         "O",
    "O":                "O",
}

_MODEL_FORWARD_KEYS = {"input_ids", "attention_mask", "bbox", "pixel_values"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Image preprocessing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resize_if_large(pil_image: Image.Image) -> Image.Image:
    w, h = pil_image.size
    if h <= MAX_IMG_HEIGHT:
        return pil_image
    scale = MAX_IMG_HEIGHT / h
    return pil_image.resize((int(w * scale), MAX_IMG_HEIGHT), Image.LANCZOS)


def _deskew(pil_image: Image.Image) -> Image.Image:
    gray   = np.array(pil_image.convert("L"))
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 50:
        return pil_image
    try:
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    except Exception:
        return pil_image
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 30:
        return pil_image
    h, w = gray.shape
    M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rot  = cv2.warpAffine(
        np.array(pil_image.convert("RGB")), M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rot)


def _remove_bleedthrough(pil_image: Image.Image) -> Image.Image:
    gray  = np.array(pil_image.convert("L"))
    clean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=15,
    )
    return Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB))


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s        = pts.sum(axis=1)
    rect[0]  = pts[np.argmin(s)]    # top-left     (smallest x+y)
    rect[2]  = pts[np.argmax(s)]    # bottom-right (largest  x+y)
    diff     = np.diff(pts, axis=1)
    rect[1]  = pts[np.argmin(diff)] # top-right    (smallest y-x)
    rect[3]  = pts[np.argmax(diff)] # bottom-left  (largest  y-x)
    return rect


def _correct_perspective(pil_image: Image.Image) -> Image.Image:
    """
    Detect the receipt quadrilateral and warp it to a top-down rectangle.
    Works best on photos where the receipt is on a contrasting background.
    Falls back to the original image if no clear 4-corner contour is found.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    h, w    = img_rgb.shape[:2]

    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 30, 120)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged   = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_image

    # Take the largest contour, apply convex hull, then reduce to 4 corners
    # by trying progressively larger approximation epsilons.
    # Require the contour to cover at least 15% of the image to avoid
    # latching onto small objects in the background.
    candidate = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    if cv2.contourArea(candidate) < 0.15 * h * w:
        return pil_image

    hull = cv2.convexHull(candidate)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break

    if quad is None:
        return pil_image

    rect = _order_points(quad)
    tl, tr, br, bl = rect
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    # Receipts are portrait documents — reject if correction produces a landscape result
    if maxW < 50 or maxH < 50 or maxW > maxH * 1.2:
        return pil_image

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )
    M   = cv2.getPerspectiveTransform(rect, dst)
    out = cv2.warpPerspective(img_rgb, M, (maxW, maxH))
    return Image.fromarray(out)


def _preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    img = _correct_perspective(pil_image)  # straighten warped/photographed receipts
    img = _resize_if_large(img)
    img = _deskew(img)
    img = _remove_bleedthrough(img)
    return img


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tesseract OCR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_words_and_boxes(pil_image: Image.Image, lang: str = "eng"):
    img    = _preprocess_for_ocr(pil_image)
    w, h   = img.size
    config = f"--oem 3 --psm 6 -l {lang}"
    data   = pytesseract.image_to_data(
        img, config=config, output_type=pytesseract.Output.DICT
    )
    words, boxes = [], []
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        if not text or int(data["conf"][i]) < 30:
            continue
        x  = data["left"][i];  y  = data["top"][i]
        bw = data["width"][i]; bh = data["height"][i]
        boxes.append([
            min(max(int(x       / w * 1000), 0), 1000),
            min(max(int(y       / h * 1000), 0), 1000),
            min(max(int((x+bw)  / w * 1000), 0), 1000),
            min(max(int((y+bh)  / h * 1000), 0), 1000),
        ])
        words.append(text)
    return words, boxes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IoU helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((a[2]-a[0]) * (a[3]-a[1]), 1)
    area_b = max((b[2]-b[0]) * (b[3]-b[1]), 1)
    return inter / (area_a + area_b - inter)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Option A — Label oracle: French TSV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _labels_from_tsv(
    tsv_path: Path,
    ocr_boxes: list,
    img_w: int,
    img_h: int,
    iou_threshold: float = 0.30,
) -> list:
    tsv_entries = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "").strip()
            if not text:
                continue
            try:
                left = int(row["left"]); top = int(row["top"])
                w    = int(row["width"]); h   = int(row["height"])
            except (KeyError, ValueError):
                continue
            field = _TSV_LABEL_MAP.get(row.get("label", "O").strip(), "O")
            tsv_entries.append((
                [
                    min(max(int(left     / img_w * 1000), 0), 1000),
                    min(max(int(top      / img_h * 1000), 0), 1000),
                    min(max(int((left+w) / img_w * 1000), 0), 1000),
                    min(max(int((top+h)  / img_h * 1000), 0), 1000),
                ],
                field,
            ))

    raw_fields = []
    for ocr_box in ocr_boxes:
        best_iou, best_field = 0.0, "O"
        for tsv_box, field in tsv_entries:
            score = _iou(ocr_box, tsv_box)
            if score > best_iou:
                best_iou, best_field = score, field
        raw_fields.append(best_field if best_iou >= iou_threshold else "O")

    bio_labels = []
    prev_field  = "O"
    for field in raw_fields:
        if field == "O":
            bio_labels.append("O")
            prev_field = "O"
        elif field != prev_field:
            bio_labels.append(f"B-{field}")
            prev_field = field
        else:
            bio_labels.append(f"I-{field}")
    return bio_labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Option A — Label oracle: SROIE box files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _labels_from_box_file(
    box_path: Path,
    ocr_boxes: list,
    entities: dict,
    img_w: int,
    img_h: int,
    iou_threshold: float = 0.30,
) -> list:
    bf_words, bf_boxes = [], []
    for line in box_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue
        try:
            x1 = int(parts[0]); y1 = int(parts[1])
            x2 = int(parts[4]); y2 = int(parts[5])  # quad format: bottom-right corner
        except ValueError:
            continue
        word = ",".join(parts[8:]).strip()
        if not word:
            continue
        bf_boxes.append([
            min(max(int(x1 / img_w * 1000), 0), 1000),
            min(max(int(y1 / img_h * 1000), 0), 1000),
            min(max(int(x2 / img_w * 1000), 0), 1000),
            min(max(int(y2 / img_h * 1000), 0), 1000),
        ])
        bf_words.append(word)

    if not bf_words:
        return ["O"] * len(ocr_boxes)

    bf_labels = _align_labels(bf_words, entities)
    bf_labels = _disambiguate_labels(bf_words, bf_labels, bf_boxes)
    bf_entries = list(zip(bf_boxes, bf_labels))

    bio_labels = []
    prev_field  = "O"
    for ocr_box in ocr_boxes:
        best_iou, best_bio = 0.0, "O"
        for bf_box, bf_label in bf_entries:
            score = _iou(ocr_box, bf_box)
            if score > best_iou:
                best_iou, best_bio = score, bf_label
        assigned = best_bio if best_iou >= iou_threshold else "O"

        field = re.sub(r"^[BI]-", "", assigned)
        if field == "O":
            bio_labels.append("O")
            prev_field = "O"
        elif assigned.startswith("B-") or field != prev_field:
            # Respect B- from source so disjoint same-field spans stay separate
            bio_labels.append(f"B-{field}")
            prev_field = field
        else:
            bio_labels.append(f"I-{field}")

    return bio_labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Option B — OCR noise augmentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_OCR_SUBSTITUTIONS = [
    ("0", "O"), ("O", "0"),
    ("1", "l"), ("l", "1"),
    ("rn", "m"), ("m", "rn"),
    ("5", "S"), ("S", "5"),
    ("8", "B"), ("B", "8"),
    ("6", "G"), ("G", "6"),
    ("vv", "w"), ("w", "vv"),
]


def _jitter_box(b: list) -> list:
    x1 = min(max(b[0] + random.randint(-20, 20), 0), 1000)
    y1 = min(max(b[1] + random.randint(-20, 20), 0), 1000)
    x2 = min(max(b[2] + random.randint(-20, 20), 0), 1000)
    y2 = min(max(b[3] + random.randint(-20, 20), 0), 1000)
    if x2 <= x1:
        x2 = min(x1 + 1, 1000)
    if y2 <= y1:
        y2 = min(y1 + 1, 1000)
    return [x1, y1, x2, y2]


def _augment_ocr_noise(words: list, boxes: list, labels: list) -> tuple:
    new_words, new_boxes, new_labels = [], [], []
    i = 0
    while i < len(words):
        w, b, l = words[i], list(boxes[i]), labels[i]
        r     = random.random()
        field = re.sub(r"^[BI]-", "", l)

        if field == "O" and r < 0.05:
            i += 1
            continue

        if r < 0.08 and field == "O":
            for src, dst in _OCR_SUBSTITUTIONS:
                if src in w:
                    w = w.replace(src, dst, 1)
                    break

        if random.random() < 0.10:
            b = _jitter_box(b)

        if (random.random() < 0.06
                and "," in w
                and re.search(r"\d,\d", w)
                and len(w) > 2):
            halves       = w.split(",", 1)
            ratio        = (len(halves[0]) + 1) / max(len(w), 1)
            split_x      = int(b[0] + (b[2] - b[0]) * ratio)
            split_x      = min(max(split_x, b[0]), b[2])
            continuation = "I-" + field if field != "O" else "O"
            new_words.append(halves[0] + ","); new_boxes.append([b[0], b[1], split_x, b[3]]); new_labels.append(l)
            new_words.append(halves[1]);        new_boxes.append([split_x, b[1], b[2], b[3]]); new_labels.append(continuation)
            i += 1
            continue

        if (random.random() < 0.04
                and "-" in w
                and re.search(r"\d-\d", w)
                and field == "DATE"):
            parts = w.split("-")
            for k, part in enumerate(parts):
                chunk = part + ("-" if k < len(parts) - 1 else "")
                tag   = l if k == 0 else "I-DATE"
                new_words.append(chunk); new_boxes.append(b); new_labels.append(tag)
            i += 1
            continue

        new_words.append(w); new_boxes.append(b); new_labels.append(l)
        i += 1

    return new_words, new_boxes, new_labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Translation cache  (thread-safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _TranslationCache:
    def __init__(self, path: Path):
        self._path  = path
        self._lock  = threading.Lock()
        self._cache = self._load()
        print(f"[setup] Translation cache: {len(self._cache)} entries")

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def get(self, key: str):
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value):
        with self._lock:
            self._cache[key] = value
            try:
                self._path.write_text(
                    json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                print(f"  [cache] WARNING: {e}")

    def __len__(self):
        with self._lock:
            return len(self._cache)


_TRANSLATION_CACHE = _TranslationCache(TRANSLATION_CACHE_PATH)


def _translate_words(words: list) -> list:
    if not words or not USE_TRANSLATION:
        return words

    import time

    cache_key = hashlib.md5("|".join(words).encode()).hexdigest()
    cached    = _TRANSLATION_CACHE.get(cache_key)
    if cached and len(cached) == len(words):
        return cached

    from deep_translator import GoogleTranslator
    result = []
    for chunk_start in range(0, len(words), TRANSLATE_CHUNK_SIZE):
        chunk            = words[chunk_start : chunk_start + TRANSLATE_CHUNK_SIZE]
        translated_chunk = None
        for attempt in range(TRANSLATE_MAX_RETRIES):
            try:
                time.sleep(TRANSLATE_SLEEP_S)
                translated_chunk = GoogleTranslator(
                    source="fr", target="en"
                ).translate_batch(chunk)
                break
            except Exception as e:
                msg = str(e)
                if "TooManyRequests" in msg or "429" in msg:
                    wait = 5 * (2 ** attempt)
                    print(f"  [translate] rate-limited, waiting {wait}s …")
                    time.sleep(wait)
                else:
                    print(f"  [translate] {type(e).__name__}: {e}")
                    break
        if translated_chunk and len(translated_chunk) == len(chunk):
            result.extend(
                t if (t and isinstance(t, str)) else w
                for t, w in zip(translated_chunk, chunk)
            )
        else:
            result.extend(chunk)

    if len(result) == len(words):
        _TRANSLATION_CACHE.set(cache_key, result)
    return result if len(result) == len(words) else words


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PDF → JPEG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pdf_to_jpeg(pdf_path: Path):
    path_hash = hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()[:8]
    out = PDF_RENDER_DIR / f"{pdf_path.stem}_{path_hash}.jpg"
    if out.exists():
        return out
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(str(pdf_path), dpi=PDF_DPI,
                                  first_page=1, last_page=1)
        if not pages:
            return None
        pages[0].convert("RGB").save(str(out), "JPEG", quality=95)
        return out
    except Exception as e:
        print(f"  [pdf2image] FAILED {pdf_path.name}: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Date normalisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MONTH_NAMES = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12,
}


def _normalise_date(raw: str, source: str = "txt") -> str:
    raw = raw.strip()
    if not raw or len(raw) < 6:
        return raw
    if source == "xml":
        m = re.match(r'^(\d{4})(\d{2})(\d{2})$', raw)
        if m:
            return f"{m.group(2)}/{m.group(3)}/{m.group(1)}"
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', raw)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"{b:02d}/{a:02d}/{y}" if a > 12 else f"{a:02d}/{b:02d}/{y}"
    s = re.sub(r'[-.]', '/', raw)
    m = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', s)
    if m:
        return f"{int(m.group(2)):02d}/{int(m.group(3)):02d}/{m.group(1)}"
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"{b:02d}/{a:02d}/{y}" if a > 12 else f"{a:02d}/{b:02d}/{y}"
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2})$', s)
    if m:
        a, b, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = 2000 + y2 if y2 < 50 else 1900 + y2
        return f"{b:02d}/{a:02d}/{y}" if a > 12 else f"{a:02d}/{b:02d}/{y}"
    m = re.match(r'^(\d{1,2})[\s\-]([A-Za-z]+)[\s\-,]?\s*(\d{2,4})$', raw.strip())
    if m:
        mo = _MONTH_NAMES.get(m.group(2).lower())
        if mo:
            y = int(m.group(3)); y = (2000+y if y<50 else 1900+y) if y<100 else y
            return f"{mo:02d}/{int(m.group(1)):02d}/{y}"
    m = re.match(r'^([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{2,4})$', raw.strip())
    if m:
        mo = _MONTH_NAMES.get(m.group(1).lower())
        if mo:
            y = int(m.group(3)); y = (2000+y if y<50 else 1900+y) if y<100 else y
            return f"{mo:02d}/{int(m.group(2)):02d}/{y}"
    return raw


def _date_digit_variants(raw: str, source: str) -> set:
    n = _normalise_date(raw, source)
    m = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', n)
    if m:
        mo, d, y = m.group(1), m.group(2), m.group(3)
        return {f"{y}{mo}{d}", f"{d}{mo}{y}", f"{mo}{d}{y}"}
    return {re.sub(r'[^0-9]', '', raw)}


def _dates_match(a: str, b: str, source: str) -> bool:
    na, nb = _normalise_date(a, source), _normalise_date(b, source)
    return bool(re.match(r'^\d{2}/\d{2}/\d{4}$', na)) and na == nb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Currency detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYMBOL_PATTERNS = [
    (r"(?<![A-Za-z])RM(?![A-Za-z])", "RM"),
    (r"R\$", "R$"), (r"Rs\.?\s*\d", "Rs."),
    (r"₹","₹"),(r"£","£"),(r"€","€"),(r"د\.إ","د.إ"),
    (r"\$","$"),(r"(?<![A-Za-z])R(?![A-Za-z])","R"),
]
_ISO_TO_SYMBOL = {
    "USD":"$","EUR":"€","GBP":"£","MYR":"RM",
    "IDR":"Rp","ZAR":"R","PKR":"Rs.","INR":"₹","BRL":"R$",
}


def _currency_symbol_from_text(text: str):
    for pattern, symbol in _SYMBOL_PATTERNS:
        if re.search(pattern, text):
            return symbol
    return None


def _currency_symbol_from_address(address: str):
    a = address.lower()
    if any(k in a for k in ("malaysia","johor","kuala lumpur","penang","selangor")):
        return "RM"
    if any(k in a for k in ("pakistan","karachi","lahore","islamabad")):
        return "Rs."
    if any(k in a for k in ("india","mumbai","delhi","bangalore","chennai")):
        return "₹"
    if any(k in a for k in ("united states","usa"," ca "," ny "," tx ",
                             "sunnyvale","san francisco","san clemente")):
        return "$"
    if any(k in a for k in ("uk","london","england","britain")):
        return "£"
    if any(k in a for k in ("france","paris","lyon","marseille")):
        return "€"
    if any(k in a for k in ("dubai","uae","abu dhabi","sharjah")):
        return "د.إ"
    if any(k in a for k in ("south africa","bergville","johannesburg","cape town")):
        return "R"
    if any(k in a for k in ("indonesia","jakarta","bali")):
        return "Rp"
    if any(k in a for k in ("nepal","kathmandu")):
        return "Rs."
    return None


def _detect_currency_symbol(total_str="", address="", full_text="", iso_code=""):
    if iso_code and len(iso_code) == 3:
        sym = _ISO_TO_SYMBOL.get(iso_code.upper())
        if sym: return sym
    if total_str:
        sym = _currency_symbol_from_text(total_str)
        if sym: return sym
    if address:
        sym = _currency_symbol_from_address(address)
        if sym: return sym
    if full_text:
        sym = _currency_symbol_from_text(full_text)
        if sym: return sym
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Amount normalisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalise_amount(raw: str) -> str:
    if raw is None: return ""
    s = re.sub(
        r"[€£¥₹₩₺₽﷼฿৳]|د\.إ|R\$"
        r"|(?<![A-Za-z])RM(?![A-Za-z])"
        r"|(?<![A-Za-z])Rp(?![A-Za-z])"
        r"|Rs\.?|Fr\.?|\bkr\b|\$"
        r"|(?<![A-Za-z])R(?![A-Za-z])",
        "", str(raw).strip(), flags=re.IGNORECASE
    ).strip()
    if re.search(r",\d{1,2}$", s):
        s = s.replace(".", "").replace(" ", "").replace(",", ".")
    else:
        s = s.replace(",", "").replace(" ", "")
    return re.sub(r"\.0+$", "", s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Canonical normalisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalise_to_canonical(raw: dict) -> dict:
    vendor_field = raw.get("vendor")
    if isinstance(vendor_field, dict):
        vendor = vendor_field.get("name")
    else:
        vendor = (vendor_field or raw.get("store_name") or
                  raw.get("merchant") or raw.get("trading_name"))
    invoice_number = (
        raw.get("invoice_number") or raw.get("bill_number") or
        raw.get("tc_number") or raw.get("slip_number") or
        raw.get("receipt_number") or raw.get("reference") or raw.get("order_number")
    )
    date = (raw.get("date") or raw.get("date_of_issue") or
            raw.get("date_paid") or raw.get("date_purchased")) or None
    fin  = raw.get("financials", {}) or {}
    total = (fin.get("total") or fin.get("amount_due") or
             fin.get("amount_paid") or fin.get("net_amount") or raw.get("total"))
    subtotal = fin.get("subtotal") or raw.get("subtotal")
    try:
        tax = float(fin.get("tax_amount") or fin.get("total_tax") or
                    raw.get("tax_amount") or raw.get("total_tax") or 0)
    except (ValueError, TypeError):
        tax = 0.0
    try:
        if total is not None and subtotal is not None and float(total) == float(subtotal) and tax > 0:
            total = round(float(subtotal) + tax, 2)
    except (ValueError, TypeError):
        pass
    if total is None and subtotal is not None:
        total = subtotal
    iso_code = fin.get("currency") or raw.get("currency") or raw.get("currency_id")
    address  = (raw.get("address") or raw.get("location", "") or
                (raw.get("vendor", {}) or {}).get("address", ""))
    currency = _detect_currency_symbol(
        total_str=str(total) if total is not None else "",
        address=str(address),
        iso_code=str(iso_code) if iso_code else "",
    )
    return {
        "vendor":         str(vendor).strip() if vendor else None,
        "invoice_number": str(invoice_number).strip() if invoice_number else None,
        "date":           str(date).strip() if date else None,
        "po_number":      str(raw.get("po_number", "")).strip() or None,
        "total_amount":   str(total) if total is not None else None,
        "subtotal":       str(subtotal) if subtotal is not None else None,
        "currency":       currency,
        "source":         "canonical",
        "date_raw":       str(date).strip() if date else "",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entity parsers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_xml(label_path: Path) -> dict:
    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    root     = ET.parse(label_path).getroot()
    def _x(tag):
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None
    untaxed   = _x("total_untaxed"); tax       = _x("tax_amount")
    sub_total = _x("sub_total");     xml_total = _x("total_amount")
    try:
        if untaxed and tax:
            total = str(round(float(untaxed) + float(tax), 2))
        elif untaxed:
            total = untaxed
        elif xml_total and float(xml_total) > float(tax or 0):
            total = xml_total
        else:
            total = sub_total or xml_total
    except (ValueError, TypeError):
        total = sub_total or xml_total
    raw_date = _x("invoice_date") or ""
    address  = _x("address") or ""
    currency = _detect_currency_symbol(address=address, full_text=raw_text)
    return {
        "vendor":         _x("supplier"),
        "date":           _normalise_date(raw_date, source="xml"),
        "date_raw":       raw_date,
        "total_amount":   total,
        "subtotal":       sub_total or untaxed,
        "invoice_number": _x("invoice_number"),
        "po_number":      _x("po_number"),
        "currency":       currency,
        "source":         "xml",
    }


def _parse_txt(label_path: Path) -> dict:
    raw = json.loads(label_path.read_text(encoding="utf-8", errors="ignore"))
    if "company" in raw and "total" in raw and "address" in raw:
        raw_date  = str(raw.get("date", "") or "")
        raw_total = str(raw.get("total", "") or "")
        address   = str(raw.get("address", "") or "")
        return {
            "vendor":         raw.get("company"),
            "date":           _normalise_date(raw_date, source="txt"),
            "date_raw":       raw_date,
            "total_amount":   raw_total,
            "subtotal":       None,
            "invoice_number": None,
            "po_number":      None,
            "currency":       _detect_currency_symbol(total_str=raw_total, address=address),
            "source":         "txt",
        }
    c = _normalise_to_canonical(raw)
    if c.get("date"):
        c["date"]     = _normalise_date(c["date"], source="txt")
        c["date_raw"] = c.get("date_raw") or c["date"]
    else:
        c["date_raw"] = ""
    return c


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Word index helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_word_index(words_norm: list) -> dict:
    idx = _defaultdict(list)
    for i, w in enumerate(words_norm):
        if w:
            idx[w].append(i)
    return idx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BIO label alignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _align_labels(words: list, entities: dict) -> list:
    labels     = ["O"] * len(words)
    words_norm = [re.sub(r"[^a-z0-9./,]", "", w.lower()) for w in words]
    word_index = _build_word_index(words_norm)
    source     = entities.get("source", "txt")

    def _mark(start, length, field):
        labels[start] = f"B-{field}"
        for k in range(1, length):
            if start + k < len(labels):
                labels[start + k] = f"I-{field}"

    def _kw(w, kset):
        return re.sub(r"[^a-z]", "", w.lower()) in kset

    # VENDOR
    vendor = entities.get("vendor")
    if vendor:
        vt = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in vendor.split() if t]
        vt = [t for t in vt if t]; n = len(vt)
        if n:
            candidates = word_index.get(vt[0], [])
            matched    = False
            for i in candidates:
                if all(i+j < len(words_norm) and words_norm[i+j] == vt[j]
                       for j in range(n)):
                    _mark(i, n, "VENDOR"); matched = True; break
            if not matched:
                for i in candidates:
                    run = 1
                    while run < n and i+run < len(words_norm) and words_norm[i+run] == vt[run]:
                        run += 1
                    if run >= max(1, n // 2):
                        _mark(i, run, "VENDOR"); break

    # DATE
    date_raw = entities.get("date_raw", "") or entities.get("date", "")
    if date_raw:
        variants  = _date_digit_variants(date_raw, source)
        date_norm = re.sub(r"[^a-z0-9]", "", date_raw.lower())
        matched   = False
        for variant in variants:
            for i in word_index.get(variant, []):
                _mark(i, 1, "DATE")
                if i > 0 and re.fullmatch(r'[^\w]+', words[i-1]):
                    labels[i-1] = "O"
                matched = True; break
            if matched: break
        if not matched:
            for i, wn in enumerate(words_norm):
                if date_norm and len(date_norm) >= 4 and (date_norm in wn or wn in date_norm):
                    _mark(i, 1, "DATE")
                    if i > 0 and re.fullmatch(r'[^\w]+', words[i-1]):
                        labels[i-1] = "O"
                    matched = True; break
        if not matched:
            for window in (2, 3, 4):
                for i in range(len(words)-window+1):
                    for cand in (" ".join(words[i:i+window]), "".join(words[i:i+window])):
                        if _dates_match(cand, date_raw, source):
                            _mark(i, window, "DATE"); matched = True; break
                    if matched: break
                if matched: break

    # TOTAL
    total_raw = entities.get("total_amount"); total_idx = None
    if total_raw is not None:
        total_norm = _normalise_amount(str(total_raw))
        if total_norm:
            for i in range(len(words)-1):
                if re.search(r'\d$', words[i]) and re.match(r'^\.\d+$', words[i+1]):
                    if _normalise_amount(words[i]+words[i+1]) == total_norm:
                        _mark(i, 2, "TOTAL"); total_idx = i; break
            if total_idx is None:
                for i in range(len(words)-1, -1, -1):
                    if re.search(r'\d+\.?\d*\s*%', words[i].strip()): continue
                    try:
                        wn = _normalise_amount(words[i])
                        if wn and (wn == total_norm or abs(float(wn)-float(total_norm)) < 0.015):
                            _mark(i, 1, "TOTAL"); total_idx = i; break
                    except (ValueError, TypeError):
                        if _normalise_amount(words[i]) == total_norm:
                            _mark(i, 1, "TOTAL"); total_idx = i; break
            if total_idx is None:
                for window in (2, 3):
                    for i in range(len(words)-window, -1, -1):
                        if any(re.search(r'\d+\.?\d*\s*%', w.strip()) for w in words[i:i+window]): continue
                        if _normalise_amount("".join(words[i:i+window])) == total_norm:
                            _mark(i, window, "TOTAL"); total_idx = i; break
                    if total_idx is not None: break
            if total_idx is not None and total_idx > 0 and _kw(words[total_idx-1], _TOTAL_KEYWORDS):
                labels[total_idx-1] = "B-TOTAL"; labels[total_idx] = "I-TOTAL"

    # SUBTOTAL
    subtotal_raw = entities.get("subtotal")
    if subtotal_raw is not None:
        sn = _normalise_amount(str(subtotal_raw))
        se = total_idx if total_idx is not None else len(words)
        if sn and sn != _normalise_amount(str(total_raw or "")):
            si = None
            for i in range(se-1):
                if re.search(r'\d$', words[i]) and re.match(r'^\.\d+$', words[i+1]):
                    if _normalise_amount(words[i]+words[i+1]) == sn:
                        _mark(i, 2, "SUBTOTAL"); si = i; break
            if si is None:
                for i in range(se-1, -1, -1):
                    try:
                        wn = _normalise_amount(words[i])
                        if wn and (wn == sn or abs(float(wn)-float(sn)) < 0.015):
                            _mark(i, 1, "SUBTOTAL")
                            if i > 0 and _kw(words[i-1], _SUBTOTAL_KEYWORDS):
                                labels[i-1] = "B-SUBTOTAL"; labels[i] = "I-SUBTOTAL"
                            break
                    except (ValueError, TypeError):
                        if _normalise_amount(words[i]) == sn:
                            _mark(i, 1, "SUBTOTAL"); break

    # INVOICE_NO
    inv_no = entities.get("invoice_number")
    if inv_no:
        inv_str  = str(inv_no)
        inv_norm = re.sub(r"[^a-z0-9/]", "", inv_str.lower())
        inv_toks = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in inv_str.split() if t]
        inv_toks = [t for t in inv_toks if t]; n = len(inv_toks); matched = False
        if n > 1:
            candidates = word_index.get(inv_toks[0], [])
            for i in candidates:
                if all(i+j < len(words_norm) and words_norm[i+j] == inv_toks[j]
                       for j in range(n)):
                    _mark(i, n, "INVOICE_NO"); matched = True; break
        if not matched:
            for i, wn in enumerate(words_norm):
                wc = re.sub(r"[^a-z0-9/]", "", wn)
                if inv_norm and (inv_norm == wc or (len(inv_norm) > 4 and inv_norm in wc)):
                    _mark(i, 1, "INVOICE_NO"); matched = True; break
        if not matched:
            inv_digits = re.sub(r"[^0-9]", "", inv_str)
            if len(inv_digits) >= 8:
                for w in range(2, 7):
                    for i in range(len(words)-w+1):
                        if re.sub(r"[^0-9]", "", "".join(words[i:i+w])) == inv_digits:
                            _mark(i, w, "INVOICE_NO"); matched = True; break
                    if matched: break

    # PO_NO
    po_no = entities.get("po_number")
    if po_no:
        pn         = re.sub(r"[^a-z0-9]", "", str(po_no).lower())
        candidates = word_index.get(pn, [])
        if candidates:
            _mark(candidates[0], 1, "PO_NO")
        else:
            for i, wn in enumerate(words_norm):
                if pn and pn == re.sub(r"[^a-z0-9]", "", wn):
                    _mark(i, 1, "PO_NO"); break

    # CURRENCY
    currency_sym = entities.get("currency")
    if currency_sym:
        se      = re.escape(str(currency_sym)); matched = False
        for i, w in enumerate(words):
            if labels[i] == "O" and re.fullmatch(se, w.strip(), re.IGNORECASE):
                _mark(i, 1, "CURRENCY"); matched = True; break
        if not matched:
            for i, w in enumerate(words):
                if labels[i] == "O" and re.match(rf'^{se}', w, re.IGNORECASE):
                    _mark(i, 1, "CURRENCY"); break

    return labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Label disambiguation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _disambiguate_labels(words: list, labels: list, boxes: list) -> list:
    _STRATEGY = {
        "VENDOR":     "largest",
        "DATE":       "topmost",
        "TOTAL":      "bottommost",
        "SUBTOTAL":   "bottommost",
        "INVOICE_NO": "largest",
        "PO_NO":      "largest",
        "CURRENCY":   "largest",
    }

    spans = []
    i = 0
    while i < len(labels):
        lbl = labels[i]
        if lbl.startswith("B-"):
            field = lbl[2:]
            j     = i + 1
            while j < len(labels) and labels[j] == f"I-{field}":
                j += 1
            first_box = boxes[i]
            area      = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
            y_centre  = (first_box[1] + first_box[3]) / 2
            spans.append((field, i, j, area, y_centre))
            i = j
        else:
            i += 1

    winners: dict = {}
    for field, start, end, area, y_centre in spans:
        strategy = _STRATEGY.get(field, "largest")
        current  = winners.get(field)
        if current is None:
            winners[field] = (start, end, area, y_centre)
        else:
            _, _, cur_area, cur_y = current
            if   strategy == "largest"     and area     > cur_area:
                winners[field] = (start, end, area, y_centre)
            elif strategy == "topmost"     and y_centre < cur_y:
                winners[field] = (start, end, area, y_centre)
            elif strategy == "bottommost"  and y_centre > cur_y:
                winners[field] = (start, end, area, y_centre)

    new_labels = ["O"] * len(labels)
    for field, (start, end, _, _) in winners.items():
        new_labels[start] = f"B-{field}"
        for k in range(start + 1, end):
            new_labels[k] = f"I-{field}"

    return new_labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stratified split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _stratified_split(samples: list, val_ratio: float = 0.20) -> tuple:
    field_counts: dict = _defaultdict(int)
    sample_fields      = []
    for s in samples:
        fields = frozenset(
            re.sub(r"^[BI]-", "", ID2LABEL[l])
            for l in s["labels"] if l != 0
        )
        sample_fields.append(fields)
        for f in fields:
            field_counts[f] += 1

    by_field: dict = _defaultdict(list)
    for s, fields in zip(samples, sample_fields):
        rarest = min(fields, key=lambda f: field_counts[f]) if fields else "O"
        by_field[rarest].append(s)

    train_out, val_out = [], []
    for field, bucket in by_field.items():
        random.shuffle(bucket)
        n_val = max(1, int(len(bucket) * val_ratio))
        val_out.extend(bucket[:n_val])
        train_out.extend(bucket[n_val:])

    random.shuffle(train_out)
    random.shuffle(val_out)
    print(f"  Stratified split → train={len(train_out)}  val={len(val_out)}")
    return train_out, val_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InvoiceDataset(Dataset):
    def __init__(self, samples: list, processor: LayoutLMv3Processor,
                 training: bool = True):
        self.samples   = samples
        self.processor = processor
        self.training  = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        image  = Image.open(s["img_path"]).convert("RGB")
        words  = list(s["words"])
        boxes  = [list(b) for b in s["boxes"]]
        labels = list(s["labels"])

        if self.training:
            str_labels = [ID2LABEL[l] for l in labels]
            words, boxes, str_labels = _augment_ocr_noise(words, boxes, str_labels)
            labels = [LABEL2ID.get(l, 0) for l in str_labels]

        encoding = self.processor(
            image, words, boxes=boxes, word_labels=labels,
            padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sample loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_IMG_EXTENSIONS = (
    ".jpg", ".jpeg", ".png",
    ".JPG", ".JPEG", ".PNG",
    ".pdf", ".PDF",
)


def load_all_samples() -> list:
    samples = []
    xml_ok = txt_sroie = txt_other = tsv_ok = 0
    pdf_count = pdf_failed = skipped = 0
    label_via_tsv = label_via_box = label_via_heuristic = 0

    all_labels = [p for p in sorted(ENTITY_DIR.rglob("*"))
                  if p.suffix in (".xml", ".txt")]
    n_xml = sum(1 for p in all_labels if p.suffix == ".xml")
    n_txt = sum(1 for p in all_labels if p.suffix == ".txt")
    print(f"  {len(all_labels)} annotation files found  "
          f"(xml={n_xml}, txt={n_txt})  in {ENTITY_DIR}", flush=True)

    for label_path in all_labels:
        img_path = next(
            (IMG_DIR / (label_path.stem + ext)
             for ext in _IMG_EXTENSIONS
             if (IMG_DIR / (label_path.stem + ext)).exists()),
            None,
        )
        if not img_path:
            skipped += 1; continue

        try:
            if label_path.suffix == ".xml":
                entities = _parse_xml(label_path)
                xml_ok  += 1
            else:
                entities = _parse_txt(label_path)
                if entities.get("source") == "txt":
                    txt_sroie += 1
                else:
                    txt_other += 1

            if img_path.suffix.lower() == ".pdf":
                rendered = _pdf_to_jpeg(img_path)
                if rendered is None:
                    pdf_failed += 1; skipped += 1; continue
                img_path  = rendered
                pdf_count += 1

            img          = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
            is_french    = entities.get("source") == "xml"
            lang         = "fra" if is_french else "eng"
            words, boxes = _get_words_and_boxes(img, lang=lang)
            img.close()

            if not words:
                skipped += 1; continue

            tsv_path = TSV_DIR / (label_path.stem + ".tsv")
            box_file = BOX_DIR / (label_path.stem + ".txt")

            if tsv_path.exists():
                bio_labels = _labels_from_tsv(tsv_path, boxes, img_w, img_h)
                label_via_tsv += 1; tsv_ok += 1
            elif box_file.exists() and not is_french:
                bio_labels = _labels_from_box_file(
                    box_file, boxes, entities, img_w, img_h
                )
                label_via_box += 1
            else:
                bio_labels = _align_labels(words, entities)
                bio_labels = _disambiguate_labels(words, bio_labels, boxes)
                label_via_heuristic += 1

            if is_french or tsv_path.exists():
                words = _translate_words(words)

            int_labels = [LABEL2ID.get(l, 0) for l in bio_labels]

            samples.append({
                "img_path": img_path,
                "words":    words,
                "boxes":    boxes,
                "labels":   int_labels,
            })

        except Exception as e:
            print(f"  [skip] {label_path.name}: {type(e).__name__}: {e}")
            skipped += 1

    print(f"  Loaded {len(samples)}")
    print(f"    {xml_ok} XML French | {tsv_ok} with TSV oracle"
          f" | {txt_sroie} SROIE | {txt_other} other JSON")
    print(f"    {pdf_count} PDFs converted OK | {pdf_failed} PDF failed")
    print(f"  Label oracle: TSV={label_via_tsv}"
          f"  BoxFile={label_via_box}"
          f"  Heuristic={label_via_heuristic}")
    print(f"  Skipped: {skipped}")
    print(f"  Translation cache: {len(_TRANSLATION_CACHE)} entries")
    return samples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 4: _validate — single classification_report call
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _validate(model, val_loader, device, epoch: int, loss_fct) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch        = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch.pop("labels")
            outputs      = model(**batch)

            loss        = loss_fct(outputs.logits.view(-1, NUM_LABELS),
                                   labels_batch.view(-1))
            total_loss += loss.item()

            preds     = outputs.logits.argmax(-1).cpu().numpy()
            label_ids = labels_batch.cpu().numpy()
            for pred_seq, label_seq in zip(preds, label_ids):
                pt, tt = [], []
                for p, l in zip(pred_seq, label_seq):
                    if l == -100: continue
                    pt.append(ID2LABEL[p]); tt.append(ID2LABEL[l])
                all_preds.append(pt); all_true.append(tt)

    avg_loss = total_loss / max(len(val_loader), 1)
    total_f1 = micro_f1 = 0.0

    try:
        # FIX 4: single call — extract both the printable string and the dict
        # from output_dict=True, then format the table ourselves.
        report_dict = classification_report(
            all_true, all_preds,
            digits=4, zero_division=0,
            output_dict=True,
        )
        print(f"\n── Epoch {epoch} validation ──")
        # Print per-class summary from the dict so we avoid a second call.
        header = f"{'':20s} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
        print(header)
        for lbl, vals in sorted(report_dict.items()):
            if isinstance(vals, dict):
                print(f"  {lbl:18s} {vals['precision']:10.4f} {vals['recall']:10.4f}"
                      f" {vals['f1-score']:10.4f} {int(vals.get('support', 0)):10d}")

        micro_f1    = seq_f1(all_true, all_preds, average="micro", zero_division=0)
        total_entry = report_dict.get("TOTAL") or report_dict.get("B-TOTAL")
        if total_entry:
            total_f1 = total_entry.get("f1-score", 0.0)

        print(f"  micro_f1={micro_f1:.4f}  total_f1={total_f1:.4f}")

    except Exception as e:
        print(f"  [seqeval error] {e}")

    return avg_loss, total_f1, micro_f1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training  — FIX 3 (grad accum) + FIX 6 (mixed precision)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # FIX 6: mixed precision scaler — no-ops gracefully on CPU
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[setup] Mixed precision (AMP): {'ON' if use_amp else 'OFF (CPU)'}")
    print(f"[setup] Gradient accumulation steps: {GRAD_ACCUM_STEPS}"
          f"  (effective batch = {BATCH_SIZE * GRAD_ACCUM_STEPS})")

    print(f"\nLoading base model: {BASE_MODEL}")
    processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL, apply_ocr=False)
    model     = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    print("\nLoading + OCR-ing dataset...")
    samples = load_all_samples()
    if not samples:
        print("ERROR: No samples found."); return

    train_samps, val_samps = _stratified_split(samples, val_ratio=0.20)
    print(f"  Train: {len(train_samps)}  Val: {len(val_samps)}\n")

    # FIX 2: num_workers=DATALOADER_WORKERS (2) — CPU prefetch keeps GPU fed.
    # persistent_workers avoids re-spawning between epochs.
    train_loader = DataLoader(
        InvoiceDataset(train_samps, processor, training=True),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=_collate,
        num_workers=DATALOADER_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(DATALOADER_WORKERS > 0),
    )
    val_loader = DataLoader(
        InvoiceDataset(val_samps, processor, training=False),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate,
        num_workers=DATALOADER_WORKERS,
        persistent_workers=(DATALOADER_WORKERS > 0),
    )

    label_weights = torch.ones(NUM_LABELS, device=device)
    label_weights[LABEL2ID["B-TOTAL"]]      =  8.0
    label_weights[LABEL2ID["I-TOTAL"]]      =  8.0
    label_weights[LABEL2ID["B-SUBTOTAL"]]   =  8.0
    label_weights[LABEL2ID["I-SUBTOTAL"]]   =  8.0
    label_weights[LABEL2ID["B-VENDOR"]]     =  5.0
    label_weights[LABEL2ID["I-VENDOR"]]     =  5.0
    label_weights[LABEL2ID["B-DATE"]]       =  5.0
    label_weights[LABEL2ID["I-DATE"]]       =  5.0
    label_weights[LABEL2ID["B-INVOICE_NO"]] =  8.0
    label_weights[LABEL2ID["I-INVOICE_NO"]] =  8.0
    label_weights[LABEL2ID["B-PO_NO"]]      = 10.0
    label_weights[LABEL2ID["I-PO_NO"]]      = 10.0
    label_weights[LABEL2ID["B-CURRENCY"]]   =  4.0
    label_weights[LABEL2ID["I-CURRENCY"]]   =  4.0
    loss_fct = torch.nn.CrossEntropyLoss(weight=label_weights, ignore_index=-100)

    # FIX 3: total_steps accounts for grad accumulation so the LR schedule
    # aligns with actual optimizer steps, not raw batch steps.
    steps_per_epoch = (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    total_steps     = steps_per_epoch * EPOCHS
    optimizer       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    scheduler       = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.06),
        num_training_steps=total_steps,
    )

    best_micro_f1 = -1.0
    best_total_f1 = -1.0
    best_epoch    = 0
    no_improve    = 0

    print(f"Training {EPOCHS} epochs × {steps_per_epoch} optimizer steps "
          f"(grad_accum={GRAD_ACCUM_STEPS}, early_stop={EARLY_STOP_PAT})...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss   = 0.0
        optim_steps  = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            batch        = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch.pop("labels")

            # FIX 6: forward pass under autocast
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss    = loss_fct(
                    outputs.logits.view(-1, NUM_LABELS),
                    labels_batch.view(-1),
                )
                # Scale loss by accumulation steps so gradients are averaged
                loss = loss / GRAD_ACCUM_STEPS

            # FIX 6: scale gradients
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS  # undo division for logging

            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optim_steps += 1

                if optim_steps % 50 == 0:
                    print(f"  Epoch {epoch}/{EPOCHS}  optim_step {optim_steps}/{steps_per_epoch}"
                          f"  loss={epoch_loss/step:.4f}", flush=True)

        val_loss, total_f1, micro_f1 = _validate(
            model, val_loader, device, epoch, loss_fct
        )
        print(f"Epoch {epoch}/{EPOCHS}  "
              f"train_loss={epoch_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"TOTAL_F1={total_f1:.4f}  "
              f"MICRO_F1={micro_f1:.4f}\n", flush=True)

        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1; best_total_f1 = total_f1
            best_epoch    = epoch;    no_improve     = 0
            BEST_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(BEST_DIR))
            processor.save_pretrained(str(BEST_DIR))
            print(f"  ✓ New best  MICRO_F1={micro_f1:.4f}  TOTAL_F1={total_f1:.4f}"
                  f"  — checkpoint → {BEST_DIR}\n")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP_PAT})\n")
            if no_improve >= EARLY_STOP_PAT:
                print(f"  Early stopping at epoch {epoch}.")
                break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\nTraining complete.")
    print(f"  Best MICRO_F1={best_micro_f1:.4f}  TOTAL_F1={best_total_f1:.4f}"
          f"  (epoch {best_epoch})")
    print(f"  Best checkpoint → {BEST_DIR}")
    print(f"  Final model     → {OUTPUT_DIR}")
    print(f"  Translation cache entries: {len(_TRANSLATION_CACHE)}")
    print(f"\nDownload layoutlm_invoice_best/ → place at data/layoutlm_invoice/")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 5: InvoiceExtractor — use the same OCR pipeline as training.
#
# Previously used apply_ocr=True which lets the processor call Tesseract
# internally with default settings — no deskewing, no bleedthrough removal,
# no confidence threshold, and eng-only language.  This differs from the
# training-time _get_words_and_boxes() path and causes a train/inference gap.
#
# Now: apply_ocr=False + explicit _get_words_and_boxes() call, identical to
# how words and boxes are produced during training.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InvoiceExtractor:
    """
    Extract structured fields from any invoice image.

    OCR pipeline is identical to training:
      _preprocess_for_ocr (resize → deskew → bleedthrough removal)
      → pytesseract with --oem 3 --psm 6, conf ≥ 30
      → normalised [0,1000] bounding boxes

    Pass lang="fra" for French invoices.

    Usage:
        extractor = InvoiceExtractor("data/layoutlm_invoice")
        result    = extractor(Image.open("my_invoice.jpg"))
        print(result)
        # {'vendor': 'Biotech Solutions', 'date': '2022-30-01',
        #  'total': '86514.18', 'subtotal': '72190.50',
        #  'invoice_number': 'FA01/2022/045684'}
    """

    def __init__(self, model_path: str = "data/layoutlm_invoice"):
        from transformers import AutoProcessor, AutoModelForTokenClassification
        # FIX 5: apply_ocr=False — we run our own OCR pipeline below.
        self.processor = AutoProcessor.from_pretrained(model_path, apply_ocr=False)
        self.model     = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

    def __call__(self, image: Image.Image, lang: str = "eng") -> dict:
        # FIX 5: run the identical OCR pipeline used at training time.
        words, boxes = _get_words_and_boxes(image, lang=lang)
        if not words:
            return {}

        enc = self.processor(
            image, words, boxes=boxes,
            padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )

        model_inputs = {k: v for k, v in enc.items() if k in _MODEL_FORWARD_KEYS}

        with torch.no_grad():
            logits = self.model(**model_inputs).logits

        raw_tokens = [self.processor.tokenizer.decode([i])
                      for i in enc.input_ids[0]]
        raw_boxes  = enc.bbox[0].tolist()
        raw_labels = [ID2LABEL.get(p.item(), "O")
                      for p in logits.argmax(-1)[0]]

        word_ids = enc.word_ids(batch_index=0)
        words_m, boxes_m, labels_m = self._merge_subwords(
            raw_tokens, raw_boxes, raw_labels, word_ids
        )
        return self._pick_fields(words_m, boxes_m, labels_m)

    @staticmethod
    def _merge_subwords(
        tokens:   list,
        boxes:    list,
        labels:   list,
        word_ids: list,
    ) -> tuple:
        word_buckets: dict = _defaultdict(list)
        for tok_idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            word_buckets[wid].append(tok_idx)

        merged_words, merged_boxes, merged_labels = [], [], []
        for wid in sorted(word_buckets):
            indices = word_buckets[wid]
            word    = "".join(tokens[i] for i in indices).strip()
            box     = boxes[indices[0]]
            count   = Counter(labels[i] for i in indices)

            sorted_labels = sorted(count, key=count.get, reverse=True)
            label = (sorted_labels[1]
                     if sorted_labels[0] == "O" and len(sorted_labels) > 1
                     else sorted_labels[0])

            merged_words.append(word)
            merged_boxes.append(box)
            merged_labels.append(label)

        return merged_words, merged_boxes, merged_labels

    @staticmethod
    def _pick_fields(words: list, boxes: list, labels: list) -> dict:
        FIELDS = {
            "vendor", "date", "total", "subtotal",
            "invoice_number", "po_number", "currency",
        }
        best: dict = {}

        i = 0
        while i < len(labels):
            label = labels[i]
            if label == "O" or label.startswith("I-"):
                i += 1
                continue

            field = label.removeprefix("B-").lower()
            if field not in FIELDS:
                i += 1
                continue

            span_words = [words[i]]
            span_box   = boxes[i]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{label.removeprefix('B-')}":
                span_words.append(words[j])
                j += 1

            y_centre = (span_box[1] + span_box[3]) / 2
            area     = (span_box[2] - span_box[0]) * (span_box[3] - span_box[1])
            text     = " ".join(w for w in span_words if w)

            current = best.get(field)
            if field in ("total", "subtotal"):
                if current is None or y_centre > current[0]:
                    best[field] = (y_centre, text)
            elif field == "date":
                if current is None or y_centre < current[0]:
                    best[field] = (y_centre, text)
            else:
                if current is None or area > current[0]:
                    best[field] = (area, text)

            i = j

        return {f: v[1] for f, v in best.items()}


train()