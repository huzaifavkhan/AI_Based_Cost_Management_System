# Anomaly detectors for invoice processing pipeline.
#
# Two detectors with the same predict() interface:
#   IsolationForestDetector — unsupervised baseline, always available
#   FLModelDetector         — federated InvoiceAnomalyNet MLP (7 features),
#                             activated when fl_active.flag exists
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest

# 7 features — must match data_partitioner.FEATURES
FEATURES = [
    "total_amount",
    "quantity",
    "unit_price",
    "days_between_po_and_invoice",
    "amount_per_unit",
    "lag_normalized",
    "amount_log",
]


# ── Isolation Forest (baseline) ───────────────────────────────────────────────

class IsolationForestDetector:
    def __init__(self, historical_csv: str):
        self._model: IsolationForest | None = None
        self._train(historical_csv)

    def _train(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Raw features only (IF doesn't need derived — it handles nonlinearity)
        raw = ["total_amount", "quantity", "unit_price", "days_between_po_and_invoice"]
        available = [f for f in raw if f in df.columns]
        X = df[available].fillna(0).values

        self._model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
        )
        self._model.fit(X)
        self._feature_cols = available
        print(f"[ml_detector] Isolation Forest trained on {len(X)} samples, features: {available}")

    def predict(self, invoice_features: dict) -> dict:
        if self._model is None:
            return {"is_anomaly": False, "anomaly_score": 0.0, "label": 1}

        row = [float(invoice_features.get(f, 0) or 0) for f in self._feature_cols]
        X   = np.array([row])

        label = int(self._model.predict(X)[0])
        score = float(self._model.decision_function(X)[0])
        return {
            "is_anomaly":    label == -1,
            "anomaly_score": round(score, 4),
            "label":         label,
        }


# ── FL Model Detector (InvoiceAnomalyNet MLP) ─────────────────────────────────

_BASE_DIR      = Path(__file__).resolve().parent.parent.parent
FL_MODEL_PATH  = _BASE_DIR / "data" / "fl_model" / "fl_model.pkl"
FL_SCALER_PATH = _BASE_DIR / "data" / "fl_model" / "fl_scaler.pkl"
FL_ACTIVE_FLAG = _BASE_DIR / "data" / "fl_model" / "fl_active.flag"


def _build_full_features(raw: dict) -> np.ndarray:
    """
    Expand the 4 raw invoice features to the full 7-feature vector expected
    by InvoiceAnomalyNet and the StandardScaler fitted during FL training.
    """
    total  = float(raw.get("total_amount", 0) or 0)
    qty    = max(float(raw.get("quantity",  1) or 1), 1.0)
    unit   = float(raw.get("unit_price",   0) or 0)
    lag    = float(raw.get("days_between_po_and_invoice", 0) or 0)

    amount_per_unit = total / qty
    lag_normalized  = min(lag / 90.0, 1.0)
    amount_log      = float(np.log1p(max(total, 0)))

    return np.array([[total, qty, unit, lag, amount_per_unit, lag_normalized, amount_log]],
                    dtype=np.float32)


class FLModelDetector:
    """
    Drop-in replacement for IsolationForestDetector using the federated
    InvoiceAnomalyNet MLP.  Same predict() interface.
    """

    def __init__(self):
        self._model  = None
        self._scaler = None
        self._load()

    def _load(self) -> None:
        if FL_MODEL_PATH.exists() and FL_SCALER_PATH.exists():
            with open(FL_MODEL_PATH,  "rb") as f:
                candidate = pickle.load(f)
            # Reject stale SGDClassifier pickles from before the MLP upgrade
            if not isinstance(candidate, torch.nn.Module):
                print("[ml_detector] Stale SGDClassifier pickle found — ignoring. "
                      "Run FL training to generate a fresh InvoiceAnomalyNet model.")
                return
            with open(FL_SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)
            self._model = candidate
            self._model.eval()
            print("[ml_detector] FL InvoiceAnomalyNet loaded")

    def predict(self, invoice_features: dict) -> dict:
        if self._model is None:
            return {"is_anomaly": False, "anomaly_score": 0.0, "label": 1}

        row = _build_full_features(invoice_features)
        if self._scaler is not None:
            row = self._scaler.transform(row).astype(np.float32)

        with torch.no_grad():
            logit = self._model(torch.from_numpy(row))
            prob  = float(torch.sigmoid(logit).item())

        is_anomaly    = prob > 0.5
        # Map to IF sign convention: anomaly → negative score, normal → positive
        anomaly_score = round((prob - 0.5) * 2, 4)   # range [-1, 1]
        label         = -1 if is_anomaly else 1

        return {
            "is_anomaly":    is_anomaly,
            "anomaly_score": anomaly_score,
            "label":         label,
        }


def get_active_detector(hist_csv: str):
    """
    Factory: returns FLModelDetector if fl_active.flag exists and model loaded,
    otherwise falls back to IsolationForestDetector.
    """
    if FL_ACTIVE_FLAG.exists():
        detector = FLModelDetector()
        if detector._model is not None:
            return detector
    return IsolationForestDetector(hist_csv)


# ── Feature builder for invoice pipeline ──────────────────────────────────────

def build_features(fields: dict, po: dict | None = None) -> dict:
    """
    Convert extracted invoice fields + PO data into detector features.
    Returns the 4 raw features; FLModelDetector expands to 7 internally.
    """
    from app.detection.rule_checker import _parse_date

    total      = float(fields.get("total_amount") or 0)
    qty        = 1.0
    unit_price = total
    days_lag   = 0

    if po:
        qty        = float(po.get("quantity")   or 1)
        po_unit    = float(po.get("unit_price") or 0)
        unit_price = po_unit if po_unit > 0 else total
        po_date    = _parse_date(str(po.get("date_issued") or po.get("po_date") or ""))
        inv_date   = _parse_date(str(fields.get("date") or ""))
        if po_date and inv_date:
            days_lag = max(0, (inv_date - po_date).days)

    return {
        "total_amount":                total,
        "quantity":                    qty,
        "unit_price":                  unit_price,
        "days_between_po_and_invoice": days_lag,
    }
