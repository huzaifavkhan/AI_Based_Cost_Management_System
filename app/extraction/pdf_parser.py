# PDF text extraction using pdfplumber (Tier 1 — primary path for digital PDFs)
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """
    Extract raw text from a digital PDF using pdfplumber.
    Returns None if the extracted text is too short (< 50 chars),
    which signals the caller to fall back to Tesseract OCR.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[pdf_parser] Error reading {pdf_path}: {e}")
        return None

    text = text.strip()
    if len(text) < 50:
        return None  # Likely a scanned PDF — caller should use OCR
    return text
