"""
Three-way matching accuracy test.
Runs all invoices through the matcher and computes precision/recall/F1.
Run: python tests/test_matching.py
"""
import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.matching.three_way_match import ThreeWayMatcher

DATA = Path(__file__).parent.parent / "data"


def load_invoices():
    invoices = []
    with open(DATA / "invoices.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            invoices.append(row)
    return invoices


def test_matching_accuracy():
    matcher = ThreeWayMatcher(
        str(DATA / "purchase_orders.csv"),
        str(DATA / "goods_receipts.csv"),
    )
    invoices = load_invoices()

    TP = FP = TN = FN = 0
    per_type: dict[str, dict] = {}

    for inv in invoices:
        fields = {
            "invoice_number": inv["invoice_number"],
            "po_number":      inv["po_number"],
            "vendor":         inv["vendor_name"],
            "date":           inv["invoice_date"],
            "total_amount":   float(inv["grand_total"]),
        }
        result = matcher.match_invoice(fields)
        predicted_anomalous = result["status"] in ("FLAGGED", "UNMATCHED")
        actual_anomalous    = inv["scenario"] != "correct"
        scenario            = inv["scenario"]

        per_type.setdefault(scenario, {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

        if actual_anomalous and predicted_anomalous:
            TP += 1; per_type[scenario]["TP"] += 1
        elif not actual_anomalous and predicted_anomalous:
            FP += 1; per_type[scenario]["FP"] += 1
        elif not actual_anomalous and not predicted_anomalous:
            TN += 1; per_type[scenario]["TN"] += 1
        else:
            FN += 1; per_type[scenario]["FN"] += 1

    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy  = (TP + TN) / max(TP + TN + FP + FN, 1)

    print(f"\n{'='*55}")
    print(f" Matching Accuracy Report  ({len(invoices)} invoices)")
    print(f"{'='*55}")
    print(f"  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    print(f"  Precision : {precision*100:.1f}%")
    print(f"  Recall    : {recall*100:.1f}%")
    print(f"  F1 Score  : {f1*100:.1f}%")
    print(f"  Accuracy  : {accuracy*100:.1f}%")
    print(f"\n  Per-scenario breakdown:")
    for sc, counts in per_type.items():
        det = counts["TP"] + counts["FP"]
        print(f"    {sc:<25} TP={counts['TP']} FN={counts['FN']}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    test_matching_accuracy()
