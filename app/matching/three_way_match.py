# Three-way match engine: Invoice ↔ Purchase Order ↔ Goods Receipt
from __future__ import annotations
import pandas as pd
from pathlib import Path
from .fuzzy_matcher import best_score, is_match, VENDOR_THRESHOLD, DEFAULT_THRESHOLD

AMOUNT_TOLERANCE = 0.02   # ±2% tolerance on total amount comparison
POINTS_PER_DISCREPANCY = 20


class ThreeWayMatcher:
    def __init__(self, po_path: str, gr_path: str):
        self.pos = pd.read_csv(po_path)
        self.grs = pd.read_csv(gr_path)
        # Normalise column names
        self.pos.columns = [c.strip().lower().replace(" ", "_") for c in self.pos.columns]
        self.grs.columns = [c.strip().lower().replace(" ", "_") for c in self.grs.columns]

    def match_invoice(self, fields: dict) -> dict:
        """
        Run the five-step matching process.

        Returns:
            {
                "status": "MATCHED" | "FLAGGED" | "UNMATCHED",
                "discrepancies": [...],
                "match_score": 0–100,
                "po": {...} | None,
                "gr": {...} | None,
            }
        """
        discrepancies: list[str] = []
        po_record = None
        gr_record = None

        # ── Step 1: PO Lookup ────────────────────────────────────────────────
        invoice_po_ref = fields.get("po_number") or ""
        if not invoice_po_ref:
            discrepancies.append("MISSING_PO: No PO reference found on invoice")
            return self._result("FLAGGED", discrepancies, 0, None, None)

        po_record = self._find_po(invoice_po_ref)
        if po_record is None:
            discrepancies.append(f"NO_MATCHING_PO: '{invoice_po_ref}' not found in PO database")
            return self._result("UNMATCHED", discrepancies, 0, None, None)

        po = po_record.to_dict()

        # ── Step 2: GR Lookup ────────────────────────────────────────────────
        gr_record = self._find_gr(po.get("po_number", ""))
        if gr_record is None:
            discrepancies.append(
                f"NO_GR: No goods receipt for PO {po.get('po_number')} — delivery may be pending"
            )
            gr = None
        else:
            gr = gr_record.to_dict()

        # ── Step 3: Amount Validation ────────────────────────────────────────
        invoice_total = fields.get("total_amount")
        po_total = _to_float(po.get("grand_total"))
        if invoice_total is not None and po_total is not None:
            diff_pct = abs(invoice_total - po_total) / max(po_total, 0.01)
            if diff_pct > AMOUNT_TOLERANCE:
                discrepancies.append(
                    f"AMOUNT_MISMATCH: Invoice ${invoice_total:.2f} vs PO ${po_total:.2f} "
                    f"({diff_pct*100:.1f}% variance)"
                )

        # ── Step 4: Quantity Validation ──────────────────────────────────────
        if gr is not None:
            po_qty = _to_float(po.get("quantity"))
            gr_qty = _to_float(gr.get("quantity_received"))
            if po_qty is not None and gr_qty is not None and po_qty != gr_qty:
                discrepancies.append(
                    f"QTY_DISCREPANCY: PO ordered {po_qty} units, GR received {gr_qty} units"
                )

        # ── Step 5: Vendor Validation ────────────────────────────────────────
        invoice_vendor = fields.get("vendor") or ""
        po_vendor = po.get("vendor_name", "")
        if invoice_vendor and po_vendor:
            score = best_score(invoice_vendor, po_vendor)
            if score < VENDOR_THRESHOLD:
                discrepancies.append(
                    f"VENDOR_MISMATCH: Invoice '{invoice_vendor}' vs PO '{po_vendor}' "
                    f"(similarity: {score:.0f}%)"
                )

        match_score = max(0, 100 - len(discrepancies) * POINTS_PER_DISCREPANCY)
        status = "MATCHED" if not discrepancies else "FLAGGED"
        return self._result(status, discrepancies, match_score, po, gr)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _find_po(self, po_ref: str):
        """Fuzzy search for PO by po_number column."""
        best_row, best_sc = None, 0
        for _, row in self.pos.iterrows():
            sc = best_score(po_ref, str(row.get("po_number", "")))
            if sc > best_sc:
                best_sc, best_row = sc, row
        if best_sc >= DEFAULT_THRESHOLD:
            return best_row
        return None

    def _find_gr(self, po_number: str):
        """Exact lookup of GR by po_number."""
        matches = self.grs[self.grs["po_number"].astype(str) == str(po_number)]
        if matches.empty:
            return None
        return matches.iloc[0]

    @staticmethod
    def _result(status, discrepancies, score, po, gr) -> dict:
        return {
            "status": status,
            "discrepancies": discrepancies,
            "match_score": score,
            "po": po,
            "gr": gr,
        }


def _to_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
