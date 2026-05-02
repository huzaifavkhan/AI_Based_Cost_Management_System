"""
Currency detection utility — shared by Groq, Donut, and XML extractors.

Priority chain (strict order):
  1. Structured field  — currency_id (ISO code like USD, PKR, EUR)
  2. Explicit code     — "USD 120", "Total: PKR 5000", "120 USD"
  3. Symbol            — $, €, £, ₨, RM, د.إ already in text
  4. Textual mention   — "US Dollar", "Pakistani Rupee", "Dirham"
  5. Context inference — country/city names in surrounding text

Always returns a symbol (e.g. $, ₨, €). Returns "UNKNOWN" if nothing found.
"""
from __future__ import annotations
import re

# ── ISO code → symbol ─────────────────────────────────────────────────────────
_ISO_MAP: dict[str, str] = {
    "USD": "$",   "US":  "$",
    "CAD": "$",   "AUD": "$",   "SGD": "$",   "HKD": "$",   "NZD": "$",
    "PKR": "₨",   "PK":  "₨",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",   "UK":  "£",
    "AED": "د.إ", "UAE": "د.إ",
    "SAR": "﷼",
    "MYR": "RM",
    "IDR": "Rp",
    "THB": "฿",
    "JPY": "¥",   "CNY": "¥",   "CNH": "¥",
    "KRW": "₩",
    "CHF": "Fr",
    "SEK": "kr",  "NOK": "kr",  "DKK": "kr",
    "BDT": "৳",
    "NPR": "₨",   "LKR": "₨",
    "TRY": "₺",
    "BRL": "R$",
    "ZAR": "R",
    "MXN": "$",
    "RUB": "₽",
}

# ── Regex: explicit code near a number ────────────────────────────────────────
# Matches "USD 120", "120 USD", "PKR5,000", "Total: EUR 99.00"
_CODE_PATTERN = re.compile(
    r"(?<![A-Z])("
    + "|".join(re.escape(k) for k in sorted(_ISO_MAP, key=len, reverse=True))
    + r")(?![A-Z])",
    re.IGNORECASE,
)

# ── Symbol patterns (ordered: longest/most specific first) ────────────────────
_SYMBOL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"د\.إ"),                           "د.إ"),
    (re.compile(r"﷼"),                              "﷼"),
    (re.compile(r"₨"),                              "₨"),
    (re.compile(r"₹"),                              "₹"),
    (re.compile(r"₺"),                              "₺"),
    (re.compile(r"₽"),                              "₽"),
    (re.compile(r"₩"),                              "₩"),
    (re.compile(r"฿"),                              "฿"),
    (re.compile(r"৳"),                              "৳"),
    (re.compile(r"R\$"),                            "R$"),
    (re.compile(r"\bRp\.?\s*\d"),                   "Rp"),
    (re.compile(r"\bRM\.?\s*\d"),                   "RM"),
    (re.compile(r"\bRs\.?\s*\d", re.IGNORECASE),   "₨"),
    (re.compile(r"\bFr\.?\s*\d", re.IGNORECASE),   "Fr"),
    (re.compile(r"\bkr\.?\s*\d", re.IGNORECASE),   "kr"),
    (re.compile(r"€"),                              "€"),
    (re.compile(r"£"),                              "£"),
    (re.compile(r"¥"),                              "¥"),
    (re.compile(r"\$"),                             "$"),
]

# ── Textual mentions ──────────────────────────────────────────────────────────
_TEXT_MAP: list[tuple[str, str]] = [
    ("pakistani rupee",    "₨"),
    ("indian rupee",       "₹"),
    ("sri lanka rupee",    "₨"),
    ("nepalese rupee",     "₨"),
    ("us dollar",          "$"),
    ("united states dollar","$"),
    ("canadian dollar",    "$"),
    ("australian dollar",  "$"),
    ("singapore dollar",   "$"),
    ("hong kong dollar",   "$"),
    ("euro",               "€"),
    ("pound sterling",     "£"),
    ("british pound",      "£"),
    ("dirham",             "د.إ"),
    ("saudi riyal",        "﷼"),
    ("malaysian ringgit",  "RM"),
    ("ringgit",            "RM"),
    ("rupiah",             "Rp"),
    ("baht",               "฿"),
    ("yen",                "¥"),
    ("yuan",               "¥"),
    ("renminbi",           "¥"),
    ("won",                "₩"),
    ("franc",              "Fr"),
    ("lira",               "₺"),
    ("ruble",              "₽"),
    ("real",               "R$"),
    ("rand",               "R"),
    ("taka",               "৳"),
    ("rupee",              "₨"),   # generic fallback after specific ones
    ("dollar",             "$"),   # generic fallback
]

# ── Context inference ─────────────────────────────────────────────────────────
_CONTEXT_MAP: list[tuple[str, str]] = [
    # Pakistan
    ("pakistan", "₨"), ("karachi", "₨"), ("lahore", "₨"),
    ("islamabad", "₨"), ("rawalpindi", "₨"), ("faisalabad", "₨"),
    # UAE
    ("uae", "د.إ"), ("dubai", "د.إ"), ("abu dhabi", "د.إ"),
    ("sharjah", "د.إ"), ("united arab", "د.إ"),
    # UK
    ("united kingdom", "£"), ("uk ", "£"), (" uk,", "£"), ("london", "£"),
    ("england", "£"), ("britain", "£"),
    # EU
    ("france", "€"), ("germany", "€"), ("spain", "€"), ("italy", "€"),
    ("netherlands", "€"), ("belgium", "€"), ("paris", "€"),
    # Malaysia
    ("malaysia", "RM"), ("kuala lumpur", "RM"), ("johor", "RM"), ("penang", "RM"),
    # India
    ("india", "₹"), ("mumbai", "₹"), ("delhi", "₹"), ("bangalore", "₹"),
    ("chennai", "₹"), ("hyderabad", "₹"),
    # Saudi
    ("saudi", "﷼"), ("riyadh", "﷼"), ("jeddah", "﷼"),
    # Indonesia
    ("indonesia", "Rp"), ("jakarta", "Rp"),
    # Japan
    ("japan", "¥"), ("tokyo", "¥"),
    # China
    ("china", "¥"), ("beijing", "¥"), ("shanghai", "¥"),
    # Korea
    ("korea", "₩"), ("seoul", "₩"),
    # Switzerland
    ("switzerland", "Fr"), ("zurich", "Fr"), ("geneva", "Fr"),
    # USA (last — broad match)
    ("united states", "$"), ("usa", "$"),
]


def detect_currency(
    text: str = "",
    currency_id: str | None = None,
    amount_context: str = "",
) -> str:
    """
    Detect currency symbol from any combination of inputs.

    Args:
        text:          Full raw document text (OCR output, JSON dump, etc.)
        currency_id:   Explicit ISO code from a structured field (e.g. "PKR", "USD")
        amount_context: Text immediately surrounding the amount field

    Returns:
        Currency symbol string, e.g. "$", "₨", "€", or "UNKNOWN".
    """
    # ── Priority 1: structured currency_id field ──────────────────────────────
    if currency_id:
        code = currency_id.strip().upper()
        if code in _ISO_MAP:
            return _ISO_MAP[code]

    # Search both amount context and full text (context first — more specific)
    search_targets = [amount_context, text]
    combined = f"{amount_context} {text}".lower()

    # ── Priority 2: explicit ISO code in text ─────────────────────────────────
    for target in search_targets:
        m = _CODE_PATTERN.search(target)
        if m:
            code = m.group(1).upper()
            if code in _ISO_MAP:
                return _ISO_MAP[code]

    # ── Priority 3: currency symbol directly in text ──────────────────────────
    for target in search_targets:
        for pattern, symbol in _SYMBOL_PATTERNS:
            if pattern.search(target):
                return symbol

    # ── Priority 4: textual mention ───────────────────────────────────────────
    for phrase, symbol in _TEXT_MAP:
        if phrase in combined:
            return symbol

    # ── Priority 5: contextual inference ─────────────────────────────────────
    for keyword, symbol in _CONTEXT_MAP:
        if keyword in combined:
            return symbol

    return "UNKNOWN"
