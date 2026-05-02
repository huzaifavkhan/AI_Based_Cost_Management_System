"""
Extraction accuracy test.
Processes all synthetic PDFs and compares extracted fields against invoices.csv ground truth.
Run: python tests/test_extraction.py
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.extraction.pdf_parser import extract_text_from_pdf
from app.extraction.field_extractor import extract_fields

DATA = Path(__file__).parent.parent / "data"
PDF_DIR = DATA / "sample_invoices"
INVOICES_CSV = DATA / "invoices.csv"


def load_ground_truth() -> dict:
    gt = {}
    with open(INVOICES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["invoice_number"]] = {
                "invoice_number": row["invoice_number"],
                "po_number":      row["po_number"],
                "vendor":         row["vendor_name"],
                "date":           row["invoice_date"],
                "total_amount":   float(row["grand_total"]),
            }
    return gt


def test_extraction_accuracy():
    gt = load_ground_truth()
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("⚠  No PDFs found. Run: python scripts/generate_synthetic_data.py")
        return

    fields_list = ["invoice_number", "po_number", "vendor", "date", "total_amount"]
    correct_counts = {f: 0 for f in fields_list}
    total = 0

    for pdf_path in pdf_files:
        inv_num = pdf_path.stem  # e.g. INV-2024-0001
        if inv_num not in gt:
            continue

        raw = extract_text_from_pdf(str(pdf_path))
        if not raw:
            continue

        extracted = extract_fields(raw)
        truth = gt[inv_num]
        total += 1

        for field in fields_list:
            ext_val = extracted.get(field)
            tru_val = truth.get(field)

            if ext_val is None or tru_val is None:
                continue

            if field == "total_amount":
                if abs(float(ext_val) - float(tru_val)) <= 0.01:
                    correct_counts[field] += 1
            else:
                if str(ext_val).strip().lower() == str(tru_val).strip().lower():
                    correct_counts[field] += 1

    print(f"\n{'='*55}")
    print(f" Extraction Accuracy Report  ({total} PDFs tested)")
    print(f"{'='*55}")
    for field in fields_list:
        pct = correct_counts[field] / max(total, 1) * 100
        print(f"  {field:<25} {correct_counts[field]:3}/{total}  {pct:.1f}%")
    overall = sum(correct_counts.values()) / (len(fields_list) * max(total, 1)) * 100
    print(f"{'─'*55}")
    print(f"  {'Overall accuracy':<25}        {overall:.1f}%")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    test_extraction_accuracy()
