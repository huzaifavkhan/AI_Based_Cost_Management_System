# Regex-based structured field extraction from raw OCR / PDF text
import re

# ── Invoice Number ────────────────────────────────────────────────────────────
# Primary labels: invoice / bill / inv
# Handles:
#   "Invoice Number V2VMGJXM-0005"   (word "number" as label)
#   "Invoice #: INV-2024-0001"       (# symbol)
#   "Invoice No. 12345"              (abbreviated)
#   "Invoice: ABC-001"               (plain colon)
#   "Invoice ID: X-123"              (ID qualifier)
#   "Invoice Ref: INV001"            (Ref qualifier)
#   "Inv No: 123", "Bill #: B-001"   (short forms)
# The pattern allows optional whitespace around hyphens/slashes to handle
# cases where pdfplumber splits "V2VMGJXM-0005" across tokens or lines.
# \b…\b ensures "inv" does not match inside "invoice" or other words.
_INVOICE_NUMBER = re.compile(
    r'\b(?:invoice|bill|inv|facture|factu|fatura|rechnung|fattura)\b'
    r'\s*(?:id|number|no\.?|#|num\.?|ref(?:erence)?)?\s*[:\-]?\s*'
    r'([A-Z0-9][A-Z0-9]*(?:\s*[\-/]\s*[A-Z0-9]+)*)',
    re.IGNORECASE
)

# ── PO Number ────────────────────────────────────────────────────────────────
# Word boundary before P.O. prevents matching "po" inside words like "Response"
_PO_NUMBER = re.compile(
    r'(?<!\w)(?:p\.o\.|po|purchase\s+order)\s*(?:number|no\.?|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,30})',
    re.IGNORECASE
)
_PO_BOX_EXCLUDE = re.compile(r'p\.?o\.?\s*box', re.IGNORECASE)

# ── Total Amount ─────────────────────────────────────────────────────────────
# Handles: "Total $20.00", "Grand Total: $1,234.56", "Amount Due 500.00"
#          "Total USD$ 29.01", "Total      USD$  29.01"
#          "TOTAL: 218 628,00 €"  (European space-thousands, comma decimal)
# Takes the LAST match (grand total is usually at the bottom)
_TOTAL_AMOUNT = re.compile(
    r'(?:grand\s+total|total\s+due|amount\s+due|amount\s+paid|total\s+amount|'
    r'net\s+amount|gross\s+amount|tender|total)'
    r'\s*[:\$]?\s*(?:USD\$?|PKR|GBP|EUR|Rs\.?|[€£¥₹])?\s*\$?\s*'
    r'([\d]{1,3}(?:\s\d{3})*(?:,\d{1,2})?|[\d,]+\.?\d*)',
    re.IGNORECASE
)

# ── Vendor / Seller ───────────────────────────────────────────────────────────
# Strategy 1: explicit label (From:, Vendor:, Sold by:, Bill from:)
_VENDOR_LABELED = re.compile(
    r'(?:from|vendor|supplier|bill(?:ed)?\s+(?:from|by)|sold\s+by|seller|pay\s+to)\s*[:\-]\s*(.+)',
    re.IGNORECASE
)
# Strategy 2: first bold/prominent company-like name at top of document
# Matches lines that look like company names (contain Ltd, Inc, Corp, PBC, Co., LLC, etc.)
_VENDOR_COMPANY = re.compile(
    r'^([A-Z][^\n]{2,60}(?:Ltd\.?|Inc\.?|Corp\.?|PBC|LLC|Co\.|GmbH|Pvt\.?|Group|Solutions|Services|Consulting|Technologies)\.?)\s*$',
    re.IGNORECASE | re.MULTILINE
)
# Strategy 3: first non-empty line of document (often the company name on receipts)
_FIRST_LINE = re.compile(r'^\s*([A-Z][^\n]{3,60})\s*$', re.MULTILINE)

# ── Date ─────────────────────────────────────────────────────────────────────
_DATE_PATTERNS = [
    # "March 10, 2026" / "Mar 10, 2026" / "September 4, 2025"
    re.compile(
        r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December'
        r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4})\b',
        re.IGNORECASE
    ),
    # "Fri 04/07/2017", "Mon 12-25-2023" — weekday prefix on receipts
    re.compile(
        r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',
        re.IGNORECASE
    ),
    # DD/MM/YYYY or MM/DD/YYYY (slash or dash)
    re.compile(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b'),
    # DD.MM.YYYY or MM.DD.YYYY (dot-separated, common on European invoices)
    re.compile(r'\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b'),
    # YYYY-MM-DD
    re.compile(r'\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b'),
]

# ── Alternative document-ID labels — same-line value ─────────────────────────
# Maps all common label variants to invoice_number.
# Covers: Transaction ID/No/Number/#, Order ID/No/Number/#,
#         Receipt No/Number/#, Reference No/ID/Number/#, Ref No/ID/#,
#         Document No/Number/#, Doc No/#, Confirmation No/Number/#,
#         Voucher No/Number/#, Booking ID/No, Serial No/Number
_ALT_DOC_NUMBER = re.compile(
    r'(?:'
    r'transaction\s*(?:id|no\.?|number|#)?|'
    r'order\s+(?:id|no\.?|number|#)|'
    r'receipt\s*(?:id|no\.?|number|#)?|'
    r'ref(?:erence)?\s+(?:no\.?|id|number|#)|'
    r'ref\s*[:#]|'
    r'document\s+(?:no\.?|id|number|#)|'
    r'doc\s+(?:no\.?|id|#)|'
    r'confirmation\s+(?:no\.?|id|number|#)|'
    r'voucher\s*(?:no\.?|id|number|#)?|'
    r'booking\s+(?:id|no\.?)|'
    r'serial\s+(?:no\.?|number)'
    r')\s*[:\-#]?\s*'
    r'([A-Z0-9#][A-Z0-9\-#]{2,30})',
    re.IGNORECASE
)

# Label-only variant of the above — used to locate the label in Phase 4
# so that _find_id_in_next_lines can scan the following lines.
_ALT_DOC_LABEL = re.compile(
    r'(?:'
    r'transaction\s*(?:id|no\.?|number|#)?|'
    r'order\s+(?:id|no\.?|number|#)|'
    r'receipt\s*(?:id|no\.?|number|#)?|'
    r'ref(?:erence)?\s+(?:no\.?|id|number|#)|'
    r'ref\s*[:#]|'
    r'document\s+(?:no\.?|id|number|#)|'
    r'doc\s+(?:no\.?|id|#)|'
    r'confirmation\s+(?:no\.?|id|number|#)|'
    r'voucher\s*(?:no\.?|id|number|#)?|'
    r'booking\s+(?:id|no\.?)|'
    r'serial\s+(?:no\.?|number)'
    r')',
    re.IGNORECASE
)

# ── Extractor ────────────────────────────────────────────────────────────────

def extract_fields(raw_text: str) -> dict:
    """
    Extract structured invoice fields from raw text (from OCR or pdfplumber).
    Returns a dict with keys:
        invoice_number, po_number, vendor, date, total_amount
    Missing fields are None.
    """
    return {
        "invoice_number": _extract_invoice_number(raw_text),
        "po_number":      _extract_po_number(raw_text),
        "vendor":         _extract_vendor(raw_text),
        "date":           _extract_date(raw_text),
        "total_amount":   _extract_total_amount(raw_text),
    }


_MASKED_CARD = re.compile(r'X{3,}', re.IGNORECASE)


def _extract_invoice_number(text: str) -> str | None:
    # Words that are label fragments, not valid ID values.
    _BAD = {
        "number", "no", "num", "date", "from", "to", "ref", "invoice",
        "due", "issue", "issued", "paid", "pay", "bill", "id",
        "receipt", "transaction", "order", "document", "reference",
        "confirmation", "voucher", "booking", "serial",
        "facture", "factu", "fatura", "rechnung", "fattura",
    }

    def _ok(val: str) -> bool:
        """True if val is a plausible document ID (not a label word or masked card)."""
        return (
            val.lower() not in _BAD
            and len(val) >= 3
            and not _MASKED_CARD.search(val)
        )

    # ── Phase 1: primary label (invoice / bill / inv) on same line as value ──
    m = _INVOICE_NUMBER.search(text)
    if m:
        val = _clean_id(m.group(1))
        if _ok(val):
            return val

    # ── Phase 2: primary label on a different line (two-column PDF layout) ───
    # Only scan the next 1-3 non-empty lines after the label.
    label_pat = re.compile(
        r'\b(?:invoice|bill|inv|facture|factu|fatura|rechnung|fattura)\b'
        r'\s*(?:id|number|no\.?|#|num\.?|ref(?:erence)?)?',
        re.IGNORECASE
    )
    lm = label_pat.search(text)
    if lm:
        val = _find_id_in_next_lines(text, lm.end(), max_lines=3)
        if val and _ok(val):
            return val

    # ── Phase 3: alternative labels on same line ─────────────────────────────
    # Covers Transaction ID, Order ID/No/Number, Receipt No/Number,
    # Reference No/ID/Number, Ref No/ID/#, Document No/Number,
    # Confirmation No/Number, Voucher No/Number, Booking ID/No, Serial No.
    m = _ALT_DOC_NUMBER.search(text)
    if m:
        val = _clean_id(m.group(1).lstrip('#'))
        if _ok(val):
            return val

    # ── Phase 4: alternative labels on a different line ──────────────────────
    lm = _ALT_DOC_LABEL.search(text)
    if lm:
        val = _find_id_in_next_lines(text, lm.end(), max_lines=3)
        if val and _ok(val):
            return val

    return None


def _clean_id(raw: str) -> str:
    """Remove spaces around separators and collapse whitespace in an ID token."""
    val = re.sub(r'\s*([\-/])\s*', r'\1', raw.strip())
    val = re.sub(r'\s+', '', val)
    return val


# Matches IDs where the FIRST segment contains both letters and digits
# (rules out pure city names like "Karachi" or zip codes like "00000")
_ID_TOKEN = re.compile(r'\b([A-Z0-9]*[A-Z][A-Z0-9]*[0-9][A-Z0-9]*(?:[\-/][A-Z0-9]+)*)\b', re.IGNORECASE)


def _find_id_in_next_lines(text: str, start: int, max_lines: int = 3) -> str | None:
    """
    Scan the next `max_lines` non-empty lines after position `start`.
    Return the first token that looks like a document ID (alphanumeric, contains
    both letters and digits, optionally hyphenated).
    Ignores lines that are clearly addresses (contain words like Street, Box, Ave, etc.)
    """
    _ADDRESS_WORDS = re.compile(
        r'\b(street|avenue|road|lane|box|blvd|suite|floor|karachi|lahore|pasadena|california|pakistan|united)\b',
        re.IGNORECASE
    )

    remaining = text[start:]
    lines_checked = 0

    for line in remaining.splitlines():
        line = line.strip()
        if not line:
            continue
        lines_checked += 1
        if lines_checked > max_lines:
            break

        # Skip lines that look like addresses
        if _ADDRESS_WORDS.search(line):
            continue

        # Look for an ID token in this line
        m = _ID_TOKEN.search(line)
        if m:
            val = _clean_id(m.group(1))
            if len(val) >= 4:
                return val

    return None


def _extract_po_number(text: str) -> str | None:
    # Find all PO-like matches
    for m in _PO_NUMBER.finditer(text):
        start = max(0, m.start() - 10)
        context = text[start : m.start() + 20]
        # Skip if this is a "P.O. Box" address line
        if _PO_BOX_EXCLUDE.search(context):
            continue
        val = m.group(1).strip()
        if val.lower() not in ("box", "number", "no", "ref") and not val.lower().startswith("box"):
            return val
    return None


def _extract_total_amount(text: str) -> float | None:
    # Collect ALL matches and return the last one
    # (grand total appears at the bottom of the invoice)
    matches = list(_TOTAL_AMOUNT.finditer(text))
    if not matches:
        return None
    # Use the last match (most likely the final total, not a subtotal)
    last = matches[-1]
    amount_str = last.group(1).strip()
    try:
        # European space-thousands with comma decimal: "218 628,00"
        if re.search(r',\d{1,2}$', amount_str):
            amount_str = amount_str.replace(' ', '').replace(',', '.')
        else:
            amount_str = amount_str.replace(',', '').replace(' ', '')
        return float(amount_str)
    except ValueError:
        return None


def _find_vendor_in_next_lines(text: str, start: int, max_lines: int = 3) -> str | None:
    """Return the first non-empty, non-address line after position `start`."""
    _ADDR = re.compile(
        r'\b(street|avenue|road|lane|box|blvd|suite|floor|account|no\.|www\.|@)\b',
        re.IGNORECASE
    )
    checked = 0
    for line in text[start:].splitlines():
        line = line.strip()
        if not line:
            continue
        checked += 1
        if checked > max_lines:
            break
        if _ADDR.search(line):
            continue
        if line[0].isdigit():
            continue
        if len(line) < 2:
            continue
        return line
    return None


_VENDOR_LABEL_ONLY = re.compile(
    r'(?:from|vendor|supplier|bill(?:ed)?\s+(?:from|by)|sold\s+by|seller|pay\s+to)\s*[:\-]?\s*$',
    re.IGNORECASE | re.MULTILINE
)


def _extract_vendor(text: str) -> str | None:
    # Strategy 1a: label and value on the same line ("From: Acme Corp")
    m = _VENDOR_LABELED.search(text)
    if m:
        vendor = re.split(r'[\n|]', m.group(1))[0].strip()
        if len(vendor) > 1:
            return vendor

    # Strategy 1b: label alone on one line, value on the next ("PAY TO:\nBorcele Bank")
    lm = _VENDOR_LABEL_ONLY.search(text)
    if lm:
        val = _find_vendor_in_next_lines(text, lm.end())
        if val:
            return val

    # Strategy 2: company-like name pattern (Ltd, Inc, PBC, etc.)
    m = _VENDOR_COMPANY.search(text)
    if m:
        return m.group(1).strip()

    # Strategy 3: top-of-document line scan — receipts always put the vendor name
    # in the first few lines. Scan raw lines (not _FIRST_LINE) so OCR lowercase
    # noise (e.g. "main Street Restaurant") doesn't cause a miss.
    _META_LABEL = re.compile(
        r'^(?:type|card|entry|mode|approval|approved|response|terminal|merchant|'
        r'transaction|auth(?:orization)?|account|batch|ref|sequence|network|'
        r'purchase|sale|credit|debit|cash|change|tax|tip|sub\s*total|grand\s*total|total|'
        r'date|time|phone|tel|fax|email|www\.|http|amount|balance|payment|paid|due)\b',
        re.IGNORECASE
    )
    # "Word(s): Value" — e.g. "Type: CREDIT", "Card Type: DISCOVER"
    _LABEL_VALUE = re.compile(r'^[A-Za-z ]{2,25}:\s*\S')
    # Contains currency/amount → financial line, not vendor
    _CURRENCY_LINE = re.compile(r'[\$£€]|\b(?:USD|PKR|GBP|EUR)\b|\d+\.\d{2}', re.IGNORECASE)
    # Lines that start with thank-you / greeting / footer phrases
    _FOOTER_LINE = re.compile(
        r'^(?:thank|thanks|welcome|please|visit|come\s+again|have\s+a|enjoy|'
        r'follow\s+us|find\s+us|like\s+us|powered\s+by|served\s+by|'
        r'customer\s+copy|merchant\s+copy)',
        re.IGNORECASE
    )

    skip_words = {
        "receipt", "invoice", "bill", "statement", "order", "page", "date",
        "to", "from", "copy", "duplicate", "reprint", "original",
    }

    # Pattern for lines that are clearly OCR garbage from background noise:
    # contain non-printable chars, lone backslashes, or mostly punctuation/symbols
    _GARBAGE_LINE = re.compile(r'[\\|]{1}|^[\W_]+$', re.IGNORECASE)

    non_empty = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in non_empty[:15]:         # only top 15 non-empty lines
        if len(line) < 3 or len(line) > 70:
            continue
        if line.lower() in skip_words:
            continue
        if line[0].isdigit():           # phone numbers, amounts, zip codes
            continue
        if _GARBAGE_LINE.search(line):  # OCR noise / background artifacts
            continue
        if _META_LABEL.match(line):     # payment/transaction keyword at start
            continue
        if _LABEL_VALUE.match(line):    # "Label: Value" metadata pair
            continue
        if _CURRENCY_LINE.search(line): # currency symbol or decimal amount
            continue
        if _FOOTER_LINE.match(line):    # thank-you / footer phrase
            continue
        if re.match(r'^[-=*#~_]+$', line):  # separator lines (---, ===, etc.)
            continue
        if line.isupper() and len(line.split()) == 1:  # single ALL-CAPS word
            continue
        # Skip spaced-out titles like "I N V O I C E" (every "word" is 1-2 chars)
        words = line.split()
        if len(words) >= 3 and all(len(w) <= 2 for w in words):
            continue
        # Skip lines that contain invoice/receipt/tax keywords anywhere
        if re.search(
            r'\b(invoice|receipt|number|subtotal|payment|description|issued|pay\s+to|'
            r'vat|tax|abbreviated|conditions|apply|pan|miti|cashier|counter|terminal|'
            r'particulars|tender|discount|remarks|address|name)\b',
            line, re.IGNORECASE
        ):
            continue
        # Skip lines that are mostly non-alphabetic (OCR garbage like "(NRL LAAN")
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.4:
            continue
        return line

    return None


def _extract_date(text: str) -> str | None:
    # Priority: dates near explicit date labels
    # Matches: "DATE: ...", "Invoice Date:", "Date paid:", "Due Date:", etc.
    labeled = re.search(
        r'(?:^date|date\s+(?:paid|issued|due|of\s+invoice)|invoice\s+date|due\s+date)\s*[:\-]?\s*(.{5,40})',
        text, re.IGNORECASE | re.MULTILINE
    )
    if labeled:
        candidate = labeled.group(1).strip()
        result = _scan_date_patterns(candidate)
        if result:
            return result

    # Fall back to any date found anywhere in the document
    return _scan_date_patterns(text)


def _scan_date_patterns(text: str) -> str | None:
    """Try each date pattern and return the first match (group 1 if present, else group 0)."""
    for pattern in _DATE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        # Patterns with a weekday prefix capture the date in group 1;
        # others capture the whole match in group 1 directly.
        try:
            return m.group(1).strip()
        except IndexError:
            return m.group(0).strip()
    return None
