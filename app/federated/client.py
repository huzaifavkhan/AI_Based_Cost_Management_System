# FL Client — local MLP training with differential privacy.
#
# Each client holds its own local dataset and a local InvoiceAnomalyNet MLP.
# On each FL round it:
#   1. Receives global flat parameters from the server
#   2. Trains locally for n_epochs (default 3 — reduced from 5 to limit client
#      drift on non-IID sector data, following FedOCR §3.1)
#   3. Computes parameter INCREMENT (ΔW = W_local − W_global)
#   4. Applies DPMechanism (clip + Gaussian noise) to ΔW
#   5. Returns a ClientUpdate with the noisy delta
#
# Sending ΔW instead of full W means DP noise is applied to the update, not
# the accumulated model state — gives tighter sensitivity bounds.
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from app.federated.model import InvoiceAnomalyNet
from app.federated.privacy import DPMechanism


@dataclass
class ClientUpdate:
    client_id:    str
    delta_params: np.ndarray  # noisy ΔW (flat 1-D), NOT full weights
    n_samples:    int
    local_f1:     float
    local_loss:   float


class FLClient:
    """
    Simulated federated learning client using InvoiceAnomalyNet (MLP).

    Trains locally, applies DP to the parameter increment ΔW, and returns
    a ClientUpdate for the server to aggregate via FedAvg on deltas.
    """

    def __init__(
        self,
        client_id:    str,
        X:            np.ndarray,
        y:            np.ndarray,
        dp_mechanism: DPMechanism,
        seed:         int = 42,
    ):
        self.client_id  = client_id
        self.dp         = dp_mechanism
        self._seed      = seed

        # ── Train / validation split (80/20, stratified) ─────────────────────
        if len(np.unique(y)) > 1:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed
            )
        else:
            X_tr, X_val, y_tr, y_val = X, X, y, y

        # ── Scaler (fit on local training data only) ──────────────────────────
        self._scaler = StandardScaler()
        self._X_tr   = self._scaler.fit_transform(X_tr).astype(np.float32)
        self._y_tr   = y_tr.astype(np.float32)
        self._X_val  = self._scaler.transform(X_val).astype(np.float32)
        self._y_val  = y_val.astype(np.float32)

        n_features = X_tr.shape[1]

        # ── Compute class weights for imbalanced anomaly data ─────────────────
        n_neg = max(1, int((self._y_tr == 0).sum()))
        n_pos = max(1, int((self._y_tr == 1).sum()))
        self._pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

        # ── Local MLP ─────────────────────────────────────────────────────────
        torch.manual_seed(seed)
        self._model = InvoiceAnomalyNet(n_features=n_features)

    # ── Receive global model ──────────────────────────────────────────────────

    def receive_global_model(self, flat_params: np.ndarray) -> None:
        """Overwrite local MLP parameters with server's global flat vector."""
        self._model.set_flat_params(torch.from_numpy(flat_params.astype(np.float32)))

    # ── Local training ────────────────────────────────────────────────────────

    def local_train(self, n_epochs: int = 3) -> ClientUpdate:
        """
        Run n_epochs of local Adam, compute ΔW, apply DP, return ClientUpdate.

        n_epochs=3 is the FedOCR recommendation for non-IID data; reduces
        client drift versus the previous default of 5.
        """
        # Save global params before local training (needed for ΔW computation)
        global_flat = self._model.get_flat_params().detach().numpy().copy()

        # ── Training loop ─────────────────────────────────────────────────────
        self._model.train()
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
        optimizer = optim.Adam(self._model.parameters(), lr=1e-3, weight_decay=1e-4)

        X_t = torch.from_numpy(self._X_tr)
        y_t = torch.from_numpy(self._y_tr).unsqueeze(1)

        last_loss = 1.0
        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits    = self._model(X_t)
            loss      = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        # ── Compute ΔW = W_local − W_global ──────────────────────────────────
        local_flat = self._model.get_flat_params().detach().numpy()
        delta_w    = local_flat - global_flat

        # ── Apply DP: clip ΔW then add Gaussian noise ─────────────────────────
        noisy_delta = self.dp.apply_flat(delta_w)

        return ClientUpdate(
            client_id    = self.client_id,
            delta_params = noisy_delta,
            n_samples    = len(self._y_tr),
            local_f1     = self._evaluate_f1(),
            local_loss   = last_loss,
        )

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def _evaluate_f1(self) -> float:
        self._model.eval()
        try:
            with torch.no_grad():
                logits = self._model(torch.from_numpy(self._X_val))
                preds  = (torch.sigmoid(logits) > 0.5).numpy().flatten().astype(int)
            return float(f1_score(self._y_val.astype(int), preds,
                                  average="macro", zero_division=0))
        except Exception:
            return 0.0
        finally:
            self._model.train()

    def evaluate_local(self) -> dict:
        """Public evaluation summary (used by coordinator for reporting)."""
        self._model.eval()
        try:
            with torch.no_grad():
                X_t    = torch.from_numpy(self._X_val)
                logits = self._model(X_t)
                preds  = (torch.sigmoid(logits) > 0.5).numpy().flatten().astype(int)
                probs  = torch.sigmoid(logits).numpy().flatten().clip(1e-9, 1 - 1e-9)
                loss   = -float(np.mean(
                    self._y_val * np.log(probs)
                    + (1 - self._y_val) * np.log(1 - probs)
                ))
            f1 = float(f1_score(self._y_val.astype(int), preds,
                                average="macro", zero_division=0))
            return {"f1": f1, "loss": loss}
        except Exception:
            return {"f1": 0.0, "loss": 1.0}
        finally:
            self._model.train()
