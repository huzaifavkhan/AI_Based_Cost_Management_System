"""
Anomaly detection evaluation.
Tests Isolation Forest with 5-fold CV and rule-based checks per anomaly type.
Run: python tests/test_detection.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA = Path(__file__).parent.parent / "data"
FEATURES = ["total_amount", "quantity", "unit_price", "days_between_po_and_invoice"]


def test_isolation_forest():
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, classification_report

    df = pd.read_csv(DATA / "historical_invoices.csv")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    available = [f for f in FEATURES if f in df.columns]
    X = df[available].fillna(0).values
    y = df["is_anomaly"].values  # 0=normal, 1=anomaly

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_test = y[test_idx]

        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        model.fit(X_train)

        scores = -model.decision_function(X_test)  # Higher = more anomalous
        try:
            auc = roc_auc_score(y_test, scores)
            aucs.append(auc)
        except Exception:
            pass

    print(f"\n{'='*55}")
    print(f" Isolation Forest — 5-Fold Cross-Validation")
    print(f"{'='*55}")
    for i, auc in enumerate(aucs, 1):
        print(f"  Fold {i}: ROC-AUC = {auc:.4f}")
    print(f"  Mean ROC-AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"{'='*55}\n")


def test_rule_checker():
    from app.detection.rule_checker import RuleChecker
    import csv

    checker = RuleChecker()
    per_rule: dict[str, int] = {}
    total_flagged = 0

    with open(DATA / "invoices.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fields = {
                "invoice_number": row["invoice_number"],
                "vendor":         row["vendor_name"],
                "date":           row["invoice_date"],
                "po_number":      row["po_number"],
                "total_amount":   float(row["grand_total"]),
            }
            flags = checker.check(fields)
            if flags:
                total_flagged += 1
                for flag in flags:
                    key = flag.split(":")[0].strip()
                    per_rule[key] = per_rule.get(key, 0) + 1

    print(f" Rule-Based Check Results")
    print(f"{'='*55}")
    print(f"  Total flagged: {total_flagged}")
    for rule, count in sorted(per_rule.items(), key=lambda x: -x[1]):
        print(f"  {rule:<30} {count}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    test_isolation_forest()
    test_rule_checker()
