# FL Coordinator — top-level orchestrator called by the FastAPI endpoint.
#
# Two operating modes (auto-detected):
#
#   DISTRIBUTED  — 3 real client servers run on ports 8001/8002/8003.
#                  Start them first:  python start_fl_clients.py
#                  The coordinator sends flat MLP params over HTTP; raw data
#                  never moves.
#
#   SIMULATION   — all clients run in-process (fallback when servers aren't up).
#                  Academically identical — proves the algorithm without the
#                  network.
#
# run_federated_training() wires together:
#   data_partitioner → FLClient × 3 → FedAvgServer → DPMechanism
# Saves model artifacts to data/fl_model/ and writes run records to SQLite.
from __future__ import annotations
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from app.federated.data_partitioner import partition_for_clients
from app.federated.client import FLClient, ClientUpdate
from app.federated.server import FedAvgServer
from app.federated.privacy import DPMechanism
from app.database import db

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE          = Path(__file__).resolve().parent.parent.parent
HIST_CSV       = str(_BASE / "data" / "historical_invoices.csv")
FL_MODEL_DIR   = _BASE / "data" / "fl_model"
FL_MODEL_PATH  = FL_MODEL_DIR / "fl_model.pkl"
FL_SCALER_PATH = FL_MODEL_DIR / "fl_scaler.pkl"
FL_META_PATH   = FL_MODEL_DIR / "fl_metadata.json"

# ── Client server URLs (override via env vars for real deployments) ───────────
import os
CLIENT_URLS = {
    "client_a": os.environ.get("FL_CLIENT_A_URL", "http://localhost:8001"),
    "client_b": os.environ.get("FL_CLIENT_B_URL", "http://localhost:8002"),
    "client_c": os.environ.get("FL_CLIENT_C_URL", "http://localhost:8003"),
}


def _to_python(obj):
    """Recursively convert numpy scalars / arrays to plain Python types for JSON."""
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, dict):         return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):return [_to_python(v) for v in obj]
    return obj


def _clients_available() -> bool:
    """Ping all 3 client servers; return True only if every one responds."""
    for url in CLIENT_URLS.values():
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code != 200:
                return False
        except Exception:
            return False
    return True


def _collect_distributed_updates(
    server:       FedAvgServer,
    sigma:        float,
    clip_norm:    float,
    local_epochs: int,
) -> list[ClientUpdate]:
    """
    Send current global flat params to each client server over HTTP.
    Each client trains locally, applies DP, and returns only the noisy ΔW
    vector — raw invoice data never crosses the network.
    """
    updates = []
    payload_base = {
        "flat_params": server.global_params.tolist(),
        "n_epochs":    local_epochs,
        "sigma":       sigma,
        "clip_norm":   clip_norm,
    }
    for cid, url in CLIENT_URLS.items():
        resp = requests.post(f"{url}/train", json=payload_base, timeout=120)
        resp.raise_for_status()
        d = resp.json()
        updates.append(ClientUpdate(
            client_id    = cid,
            delta_params = np.array(d["delta_params"], dtype=float),
            n_samples    = int(d["n_samples"]),
            local_f1     = float(d["local_f1"]),
            local_loss   = float(d["local_loss"]),
        ))
    return updates


def run_federated_training(
    n_rounds:     int   = 15,
    sigma:        float = 1.0,
    clip_norm:    float = 1.0,
    local_epochs: int   = 3,
) -> dict:
    """
    Execute a complete federated training run.

    Auto-detects mode:
      • DISTRIBUTED  if client servers on ports 8001/8002/8003 are reachable
      • SIMULATION   otherwise (all clients run in-process)

    Returns a structured result dict matching the API response schema.
    """
    distributed = _clients_available()
    mode        = "distributed" if distributed else "simulation"
    print(f"[fl_coordinator] Mode: {mode.upper()}")

    run_id     = f"FL-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    started_at = datetime.utcnow().isoformat()
    dp         = DPMechanism(clip_norm=clip_norm, sigma=sigma)

    # ── 1. Partition data ─────────────────────────────────────────────────────
    try:
        partitions = partition_for_clients(HIST_CSV)
    except FileNotFoundError:
        raise RuntimeError(f"historical_invoices.csv not found at {HIST_CSV}")

    n_features = partitions["client_a"]["n_features"]

    # ── 2. Gather client metadata ─────────────────────────────────────────────
    if distributed:
        for url in CLIENT_URLS.values():
            try:
                requests.post(f"{url}/reset", timeout=5)
            except Exception:
                pass
        client_infos = {}
        for cid, url in CLIENT_URLS.items():
            client_infos[cid] = requests.get(f"{url}/info", timeout=5).json()
    else:
        client_infos = {
            cid: {
                "sector":       p["sector"],
                "n_samples":    int(p["n_samples"]),
                "amount_range": [float(p["amount_range"][0]), float(p["amount_range"][1])],
                "anomaly_rate": float(p["anomaly_rate"]),
            }
            for cid, p in partitions.items()
        }

    # ── 3. Initialise server + (simulation) clients ───────────────────────────
    server = FedAvgServer(n_features=n_features)
    server.initialize(partitions)

    sim_clients = None
    if not distributed:
        sim_clients = [
            FLClient(cid, p["X"], p["y"], dp)
            for cid, p in partitions.items()
        ]

    # ── Register run in DB ────────────────────────────────────────────────────
    db.insert_fl_run({
        "run_id":        run_id,
        "started_at":    started_at,
        "status":        "running",
        "n_rounds":      n_rounds,
        "sigma":         sigma,
        "clip_norm":     clip_norm,
        "local_epochs":  local_epochs,
        "delta":         dp.delta,
        "client_a_size": client_infos["client_a"]["n_samples"],
        "client_b_size": client_infos["client_b"]["n_samples"],
        "client_c_size": client_infos["client_c"]["n_samples"],
    })

    # ── 4. Run FL rounds ──────────────────────────────────────────────────────
    t0            = time.time()
    round_history = []
    last_updates  = {}

    try:
        for r in range(1, n_rounds + 1):
            if distributed:
                updates = _collect_distributed_updates(server, sigma, clip_norm, local_epochs)
                last_updates = {u.client_id: u for u in updates}
                metrics = server.aggregate_round(updates, r, dp, run_id)
            else:
                metrics = server.run_round(sim_clients, r, dp, run_id)

            round_history.append({
                "round":          r,
                "f1":             round(metrics.f1,             4),
                "accuracy":       round(metrics.accuracy,       4),
                "coef_change":    round(metrics.param_change,   6),
                "epsilon_so_far": round(metrics.epsilon_so_far, 6),
            })

    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        db.update_fl_run(run_id, "failed", datetime.utcnow().isoformat(), {
            "error_message": str(exc), "training_time_s": elapsed,
        })
        raise

    elapsed = round(time.time() - t0, 2)

    # ── 5. Save model artifacts ───────────────────────────────────────────────
    FL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model, scaler = server.export_model()
    with open(FL_MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(FL_SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    final_metrics = round_history[-1] if round_history else {}
    final_f1      = final_metrics.get("f1",       0.0)
    final_acc     = final_metrics.get("accuracy", 0.0)
    final_eps     = dp.compute_epsilon(n_rounds)

    meta = {
        "run_id":         run_id,
        "trained_at":     datetime.utcnow().isoformat(),
        "n_rounds":       n_rounds,
        "n_features":     n_features,
        "model_type":     "InvoiceAnomalyNet",
        "final_f1":       final_f1,
        "final_accuracy": final_acc,
        "epsilon":        final_eps,
        "delta":          dp.delta,
        "sigma":          sigma,
        "clip_norm":      clip_norm,
        "model_version":  run_id,
        "mode":           mode,
    }
    with open(FL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    # ── 6. Update DB ──────────────────────────────────────────────────────────
    db.update_fl_run(run_id, "completed", datetime.utcnow().isoformat(), {
        "final_f1":        final_f1,
        "final_accuracy":  final_acc,
        "epsilon":         final_eps,
        "training_time_s": elapsed,
        "error_message":   None,
    })

    # ── 7. Build result dict ──────────────────────────────────────────────────
    client_stats = {}
    for cid, info in client_infos.items():
        if distributed and cid in last_updates:
            local_f1 = round(float(last_updates[cid].local_f1), 4)
        elif sim_clients:
            client = next(c for c in sim_clients if c.client_id == cid)
            local_f1 = round(client.evaluate_local()["f1"], 4)
        else:
            local_f1 = 0.0

        client_stats[cid] = {
            "sector":       info["sector"],
            "n_samples":    int(info["n_samples"]),
            "amount_range": info["amount_range"],
            "anomaly_rate": round(float(info["anomaly_rate"]), 4),
            "local_f1":     local_f1,
        }

    result = {
        "run_id":                run_id,
        "status":                "completed",
        "mode":                  mode,
        "n_rounds":              n_rounds,
        "final_f1":              final_f1,
        "final_accuracy":        final_acc,
        "epsilon":               final_eps,
        "delta":                 dp.delta,
        "sigma":                 sigma,
        "clip_norm":             clip_norm,
        "model_updated":         True,
        "training_time_seconds": elapsed,
        "round_history":         round_history,
        "client_stats":          client_stats,
        "privacy_report":        dp.privacy_report(n_rounds),
    }
    return _to_python(result)
