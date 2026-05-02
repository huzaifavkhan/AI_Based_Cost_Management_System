# Rule-based anomaly detection — deterministic checks on extracted invoice fields
from __future__ import annotations
from datetime import datetime, date


class RuleChecker:
    """
    Stateful rule checker. Maintains a set of seen invoice numbers across
    all processed invoices in the current session (for duplicate detection).
    """
    MAX_AMOUNT = 50_000.0
    MIN_AMOUNT = 1.0
    ROUND_THRESHOLD = 10_000.0
    MAX_DAYS_AFTER_PO = 90

    def __init__(self):
        self._seen_invoice_numbers: set[str] = set()

    def check(self, fields: dict, po: dict | None = None) -> list[str]:
        """
        Run all rule-based checks. Returns a list of flag strings.
        Empty list means no rule-based anomalies detected.
        """
        flags: list[str] = []
        flags += self._check_duplicate(fields)
        flags += self._check_missing_fields(fields)
        flags += self._check_amount_range(fields)
        flags += self._check_round_number(fields)
        flags += self._check_dates(fields, po)
        return flags

    # ── Rules ────────────────────────────────────────────────────────────────

    def _check_duplicate(self, fields: dict) -> list[str]:
        inv_num = fields.get("invoice_number")
        if not inv_num:
            return []
        if inv_num in self._seen_invoice_numbers:
            return [f"DUPLICATE: Invoice number '{inv_num}' has been submitted before"]
        self._seen_invoice_numbers.add(inv_num)
        return []

    def _check_missing_fields(self, fields: dict) -> list[str]:
        flags = []
        for field in ("vendor", "total_amount", "date"):
            if fields.get(field) is None:
                flags.append(f"MISSING_FIELD: '{field}' could not be extracted from invoice")
        return flags

    def _check_amount_range(self, fields: dict) -> list[str]:
        total = fields.get("total_amount")
        if total is None:
            return []
        flags = []
        if total > self.MAX_AMOUNT:
            flags.append(f"HIGH_AMOUNT: Invoice total ${total:,.2f} exceeds threshold ${self.MAX_AMOUNT:,.0f}")
        elif total < self.MIN_AMOUNT:
            flags.append(f"LOW_AMOUNT: Invoice total ${total:,.2f} is suspiciously low")
        return flags

    def _check_round_number(self, fields: dict) -> list[str]:
        total = fields.get("total_amount")
        if total is None or total <= self.ROUND_THRESHOLD:
            return []
        if total % 1000 == 0:
            return [f"ROUND_AMOUNT: Total ${total:,.0f} is a suspicious round number"]
        return []

    def _check_dates(self, fields: dict, po: dict | None) -> list[str]:
        date_str = fields.get("date")
        if not date_str:
            return []

        inv_date = _parse_date(date_str)
        if inv_date is None:
            return []

        flags = []

        if inv_date.weekday() >= 5:  # Saturday=5, Sunday=6
            flags.append(f"WEEKEND_DATE: Invoice dated on a weekend ({date_str})")

        if po is not None:
            po_date_str = po.get("date_issued") or po.get("po_date") or ""
            po_date = _parse_date(str(po_date_str))
            if po_date:
                if inv_date < po_date:
                    flags.append(
                        f"PREDATES_PO: Invoice date ({date_str}) is before PO issue date ({po_date_str})"
                    )
                elif (inv_date - po_date).days > self.MAX_DAYS_AFTER_PO:
                    flags.append(
                        f"LATE_INVOICE: Invoice is {(inv_date - po_date).days} days after PO issue date"
                    )
        return flags


# ── Helpers ──────────────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
    "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y",
    "%B %d %Y", "%b %d %Y",
]


def _parse_date(s: str) -> date | None:
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None
