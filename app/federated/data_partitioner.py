# Partitions historical_invoices.csv into 3 non-IID client datasets.
#
# Simulates realistic heterogeneous data across three healthcare sectors:
#   Client A — Hospital / Inpatient:  top 40% by invoice amount (~large stays)
#   Client B — Clinic / Outpatient:   middle 35%                (~routine visits)
#   Client C — Insurance / Payer:     bottom 25%               (~small claims)
#
# Non-IID property: each client only sees invoices from its amount range,
# giving genuinely different feature distributions — the key motivation for FL.
#
# Feature engineering (7 features instead of 4):
#   Original 4: total_amount, quantity, unit_price, days_between_po_and_invoice
#   Derived  3: amount_per_unit, lag_normalized, amount_log
#   These capture the nonlinear interactions (overbilling per unit, unusually
#   long approval lags, log-scale amount outliers) that a linear model misses.
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# 7 features — keep in sync with model.py (n_features=7) and ml_detector.py
RAW_FEATURES = [
    "total_amount",
    "quantity",
    "unit_price",
    "days_between_po_and_invoice",
]
DERIVED_FEATURES = [
    "amount_per_unit",   # total_amount / max(quantity, 1)
    "lag_normalized",    # days / 90  (caps at 1.0 for >90-day delays)
    "amount_log",        # log1p(total_amount)
]
FEATURES = RAW_FEATURES + DERIVED_FEATURES  # 7 total

# Static metadata for each client (shown in dashboard even before training)
CLIENT_META = {
    "client_a": {"name": "Client A", "sector": "Hospital / Inpatient"},
    "client_b": {"name": "Client B", "sector": "Clinic / Outpatient"},
    "client_c": {"name": "Client C", "sector": "Insurance / Payer"},
}


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add 3 derived feature columns in-place and return df."""
    qty = df["quantity"].clip(lower=1) if "quantity" in df.columns else pd.Series(1.0, index=df.index)
    amt = df["total_amount"] if "total_amount" in df.columns else pd.Series(0.0, index=df.index)
    lag = df["days_between_po_and_invoice"] if "days_between_po_and_invoice" in df.columns else pd.Series(0.0, index=df.index)

    df["amount_per_unit"] = (amt / qty).fillna(0.0)
    df["lag_normalized"]  = (lag / 90.0).clip(upper=1.0).fillna(0.0)
    df["amount_log"]      = np.log1p(amt.clip(lower=0)).fillna(0.0)
    return df


def partition_for_clients(csv_path: str, seed: int = 42) -> dict:
    """
    Load historical_invoices.csv and split into 3 non-IID client datasets.

    Returns
    -------
    dict with keys 'client_a', 'client_b', 'client_c'.
    Each value is:
        {
            "X":            np.ndarray  shape (n, 7) — feature matrix
            "y":            np.ndarray  shape (n,)   — labels (0=normal, 1=anomaly)
            "n_features":   int         = 7
            "name":         str
            "sector":       str
            "n_samples":    int
            "amount_range": (float, float)
            "anomaly_rate": float
            "feature_cols": list[str]
        }
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Coerce raw numeric columns
    for col in RAW_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Add derived features
    df = _add_derived(df)

    # Binary label: 1 = anomaly, 0 = normal
    if "is_anomaly" in df.columns:
        df["label"] = df["is_anomaly"].astype(int)
    else:
        df["label"] = 0

    # ── Split normal invoices by amount percentile ────────────────────────────
    normal_df  = df[df["label"] == 0].copy()
    anomaly_df = df[df["label"] == 1].copy()

    p25 = normal_df["total_amount"].quantile(0.25)
    p60 = normal_df["total_amount"].quantile(0.60)
    a_normal = normal_df[normal_df["total_amount"] > p60]
    b_normal = normal_df[(normal_df["total_amount"] > p25) & (normal_df["total_amount"] <= p60)]
    c_normal = normal_df[normal_df["total_amount"] <= p25]

    # ── Distribute anomalies evenly across clients ────────────────────────────
    anomaly_idx = rng.permutation(len(anomaly_df))
    n = len(anomaly_idx)
    cut1 = n // 3 + (1 if n % 3 > 0 else 0)
    cut2 = cut1 + n // 3 + (1 if n % 3 > 1 else 0)
    splits_a = anomaly_idx[:cut1]
    splits_b = anomaly_idx[cut1:cut2]
    splits_c = anomaly_idx[cut2:]

    a_df = pd.concat([a_normal, anomaly_df.iloc[splits_a]], ignore_index=True)
    b_df = pd.concat([b_normal, anomaly_df.iloc[splits_b]], ignore_index=True)
    c_df = pd.concat([c_normal, anomaly_df.iloc[splits_c]], ignore_index=True)

    # ── Per-client distributional perturbation (realistic billing cycles) ─────
    lag_col = "days_between_po_and_invoice"
    # Hospital: longer billing cycles (+30% lag)
    a_df[lag_col] = (a_df[lag_col] * 1.3).clip(upper=90).astype(float)
    # Insurance: faster pre-auth turnaround (-30% lag, min 0)
    c_df[lag_col] = (c_df[lag_col] * 0.7).clip(lower=0).astype(float)
    # Recompute lag_normalized after perturbation
    for sub_df in [a_df, b_df, c_df]:
        sub_df["lag_normalized"] = (sub_df[lag_col] / 90.0).clip(upper=1.0).fillna(0.0)

    def _to_dataset(sub_df: pd.DataFrame, client_id: str) -> dict:
        avail = [f for f in FEATURES if f in sub_df.columns]
        X = sub_df[avail].values.astype(float)
        y = sub_df["label"].values.astype(int)
        amt = sub_df["total_amount"].values
        return {
            "X":            X,
            "y":            y,
            "n_features":   len(avail),
            "name":         CLIENT_META[client_id]["name"],
            "sector":       CLIENT_META[client_id]["sector"],
            "n_samples":    len(y),
            "amount_range": (float(amt.min()), float(amt.max())),
            "anomaly_rate": float(y.mean()),
            "feature_cols": avail,
        }

    return {
        "client_a": _to_dataset(a_df, "client_a"),
        "client_b": _to_dataset(b_df, "client_b"),
        "client_c": _to_dataset(c_df, "client_c"),
    }
