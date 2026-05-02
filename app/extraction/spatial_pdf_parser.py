# Spatial PDF parser — Python port of LiteParse's core grid-projection logic.
#
# LiteParse reconstructs multi-column invoice layouts using:
#   1. Word-level bounding boxes from the PDF engine
#   2. Line grouping by Y-proximity
#   3. Column gap detection (large horizontal gaps → column separator)
#   4. Gap-based word joining (preserves tokens like "V2VMGJXM-0005" intact)
#
# This module implements those same steps using pdfplumber as the PDF engine.
# It replaces the naive pdfplumber.extract_text() call with spatially-aware output.

from __future__ import annotations
import re
import statistics

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False


# ── Constants (mirrored from LiteParse gridProjection.ts) ─────────────────────
_MERGE_TOLERANCE   = 2    # PDF points — merge nearby anchor X positions
_ANCHOR_MIN_COUNT  = 2    # min occurrences for an X to be a column anchor
_LINE_HEIGHT_RATIO = 0.6  # fraction of median height used for row-grouping threshold
# Number of char-widths gap that indicates a COLUMN boundary (not just a word gap)
_COLUMN_GAP_CHARS  = 3


# ── Public API ────────────────────────────────────────────────────────────────

def extract_text_spatial(pdf_path: str) -> str | None:
    """
    Extract text from a PDF with spatial layout reconstruction.
    Returns None if pdfplumber is unavailable, extraction fails, or < 50 chars.
    """
    if not _PDFPLUMBER_AVAILABLE:
        return None
    try:
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = _process_page(page)
                if page_text:
                    pages_text.append(page_text)
        result = "\n\n".join(pages_text)
        return result if len(result.strip()) >= 50 else None
    except Exception:
        return None


# ── Page processing ───────────────────────────────────────────────────────────

def _process_page(page) -> str:
    """
    Run the full spatial pipeline on one pdfplumber page:
      1. Extract word tokens with bounding boxes
      2. Group into text lines by Y-proximity
      3. Join words within each line using gap-proportional spacing
      4. Clean and return
    """
    raw_words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    )
    if not raw_words:
        return ""

    words = [
        {"text": w["text"], "x0": float(w["x0"]), "x1": float(w["x1"]),
         "top": float(w["top"]), "bottom": float(w["bottom"])}
        for w in raw_words
    ]

    # Step 1: group words into lines
    lines = _group_into_lines(words)
    if not lines:
        return ""

    # Step 2: estimate median character width across the whole page
    char_w = _estimate_char_width(words)

    # Step 3: join each line's words using gap-proportional spacing
    grid_lines = _join_lines(lines, char_w)

    # Step 4: clean
    return _clean_page_text(grid_lines)


# ── Step 1: Line grouping ─────────────────────────────────────────────────────

def _group_into_lines(words: list[dict]) -> list[list[dict]]:
    """
    Group words into text lines based on Y-overlap / proximity.
    Two words belong to the same line if their vertical ranges overlap by
    more than _LINE_HEIGHT_RATIO × median height (same logic as LiteParse).
    """
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    heights = [w["bottom"] - w["top"] for w in sorted_words if w["bottom"] > w["top"]]
    if not heights:
        return []
    median_h  = statistics.median(heights)
    threshold = median_h * _LINE_HEIGHT_RATIO

    lines: list[list[dict]] = []
    current_line   = [sorted_words[0]]
    current_top    = sorted_words[0]["top"]
    current_bottom = sorted_words[0]["bottom"]

    for word in sorted_words[1:]:
        w_top    = word["top"]
        w_bottom = word["bottom"]

        overlap = min(current_bottom, w_bottom) - max(current_top, w_top)
        if overlap >= threshold or abs(w_top - current_top) < threshold:
            current_line.append(word)
            current_top    = min(current_top, w_top)
            current_bottom = max(current_bottom, w_bottom)
        else:
            lines.append(sorted(current_line, key=lambda w: w["x0"]))
            current_line   = [word]
            current_top    = w_top
            current_bottom = w_bottom

    if current_line:
        lines.append(sorted(current_line, key=lambda w: w["x0"]))

    return lines


# ── Step 2: Character width estimation ───────────────────────────────────────

def _estimate_char_width(words: list[dict]) -> float:
    """Estimate median character width across all words on the page."""
    widths = []
    for w in words:
        t = w["text"]
        if t:
            widths.append((w["x1"] - w["x0"]) / max(len(t), 1))
    if not widths:
        return 6.0
    return max(statistics.median(widths), 3.0)


# ── Step 3: Gap-based word joining ───────────────────────────────────────────

def _join_lines(lines: list[list[dict]], char_w: float) -> list[str]:
    """
    Convert each line of word-dicts into a readable string.

    Strategy (mirrors LiteParse's floating-text spacing):
    - For each consecutive pair of words, measure the horizontal gap.
    - gap < 1 char_w  → join directly (no space — rare, handles tight kerning)
    - gap 1–3 char_w  → single space (normal word gap)
    - gap > 3 char_w  → multiple spaces proportional to gap (column separator)
      capped at 8 spaces so the line stays readable for regex extraction.

    pdfplumber already keeps tokens like "V2VMGJXM-0005" intact as one word,
    so hyphens are never lost — unlike the character-grid approach.
    """
    result = []
    for line in lines:
        if not line:
            continue
        parts = [line[0]["text"]]
        for i in range(1, len(line)):
            prev = line[i - 1]
            curr = line[i]
            gap  = curr["x0"] - prev["x1"]

            if gap <= 0:
                # Overlapping or touching — join directly
                spaces = ""
            elif gap < char_w * 1.0:
                # Very tight — single space
                spaces = " "
            elif gap < char_w * _COLUMN_GAP_CHARS:
                # Normal word gap — single space
                spaces = " "
            else:
                # Column-level gap — use proportional spaces (2–8)
                n = min(int(gap / char_w), 8)
                spaces = " " * max(n, 2)

            parts.append(spaces + curr["text"])
        result.append("".join(parts))
    return result


# ── Step 4: Text cleanup (mirrors cleanText.ts) ───────────────────────────────

def _clean_page_text(lines: list[str]) -> str:
    """
    - Collapse runs of 5+ spaces to double-space (column separator, stays readable)
    - Remove null bytes
    - Trim leading/trailing blank lines (margin removal, same as cleanText.ts)
    """
    cleaned = []
    for line in lines:
        line = re.sub(r' {5,}', '  ', line)   # keep 2-space column gap
        line = line.replace('\x00', ' ')
        cleaned.append(line.rstrip())

    # Trim top/bottom blank lines
    start = 0
    while start < len(cleaned) and not cleaned[start].strip():
        start += 1
    end = len(cleaned)
    while end > start and not cleaned[end - 1].strip():
        end -= 1

    return "\n".join(cleaned[start:end])
