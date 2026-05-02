# Per-client Federated Learning server.
#
# Each client is a real separate process that holds its own data slice.
# The coordinator sends global flat MLP params → this server trains locally
# → returns noisy ΔW (parameter increment). Raw invoice data NEVER leaves
# this process.
#
# Run as:
#   CLIENT_ID=client_a uvicorn app.federated.client_server:app --port 8001
#   CLIENT_ID=client_b uvicorn app.federated.client_server:app --port 8002
#   CLIENT_ID=client_c uvicorn app.federated.client_server:app --port 8003
#
# Or just: python start_fl_clients.py
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.federated.data_partitioner import partition_for_clients
from app.federated.privacy import DPMechanism
from app.federated.client import FLClient

# ── Identity ──────────────────────────────────────────────────────────────────
CLIENT_ID = os.environ.get("CLIENT_ID", "client_a")
_VALID_IDS = {"client_a", "client_b", "client_c"}
if CLIENT_ID not in _VALID_IDS:
    raise ValueError(f"CLIENT_ID must be one of {_VALID_IDS}, got '{CLIENT_ID}'")

_BASE    = Path(__file__).resolve().parent.parent.parent
HIST_CSV = str(_BASE / "data" / "historical_invoices.csv")

app = FastAPI(
    title=f"FL Client — {CLIENT_ID}",
    description="Federated Learning client server",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Load this client's data slice once at startup ─────────────────────────────
print(f"[{CLIENT_ID}] Loading data partition from {HIST_CSV} …")
_all_partitions = partition_for_clients(HIST_CSV)
_partition      = _all_partitions[CLIENT_ID]
print(
    f"[{CLIENT_ID}] Loaded {_partition['n_samples']} samples "
    f"({_partition['sector']}) — anomaly rate {_partition['anomaly_rate']:.2%} "
    f"— {_partition['n_features']} features"
)

# Persistent FLClient across rounds (carries warm model state between rounds)
_fl_client: FLClient | None = None


# ── Request / response schemas ────────────────────────────────────────────────

class TrainRequest(BaseModel):
    flat_params: list[float]   # global MLP flat parameter vector
    n_epochs:    int   = 3
    sigma:       float = 1.0
    clip_norm:   float = 1.0


class TrainResponse(BaseModel):
    client_id:    str
    delta_params: list[float]  # noisy ΔW flat vector
    n_samples:    int
    local_f1:     float
    local_loss:   float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/info")
def info():
    """Return static metadata about this client's data partition."""
    amt = _partition["amount_range"]
    return {
        "client_id":    CLIENT_ID,
        "sector":       _partition["sector"],
        "n_samples":    int(_partition["n_samples"]),
        "n_features":   int(_partition["n_features"]),
        "amount_range": [float(amt[0]), float(amt[1])],
        "anomaly_rate": float(_partition["anomaly_rate"]),
    }


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    """
    Receive global flat MLP params → train locally for n_epochs →
    apply DP (clip + noise) to ΔW → return noisy delta.
    Raw invoice data never crosses the network.
    """
    global _fl_client

    dp = DPMechanism(clip_norm=req.clip_norm, sigma=req.sigma)

    if _fl_client is None:
        _fl_client = FLClient(CLIENT_ID, _partition["X"], _partition["y"], dp)
    else:
        _fl_client.dp = dp

    # 1. Receive global model
    _fl_client.receive_global_model(np.array(req.flat_params, dtype=float))

    # 2. Train locally + compute DP-noised ΔW
    update = _fl_client.local_train(n_epochs=req.n_epochs)

    return TrainResponse(
        client_id    = CLIENT_ID,
        delta_params = update.delta_params.tolist(),
        n_samples    = int(update.n_samples),
        local_f1     = float(update.local_f1),
        local_loss   = float(update.local_loss),
    )


@app.post("/reset")
def reset():
    """Clear local model state before a new training run."""
    global _fl_client
    _fl_client = None
    return {"status": "reset", "client_id": CLIENT_ID}


@app.get("/health")
def health():
    return {"status": "ok", "client_id": CLIENT_ID}
