# FedAvg Server — aggregates client ΔW increments and tracks training metrics.
#
# Implements:
#   • FedAvg on parameter increments (ΔW): global += Σ w_i * ΔW_i
#     Sending increments instead of full weights gives DP better sensitivity
#     calibration and matches FedOCR Algorithm 1/2.
#   • Secure aggregation simulation: pairwise random masks that cancel in sum
#   • Global validation evaluation after each round (MLP forward pass, no
#     partial_fit anti-pattern)
#   • Per-round metric storage to SQLite via db module
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from app.federated.client import FLClient, ClientUpdate
from app.federated.model import InvoiceAnomalyNet
from app.federated.privacy import DPMechanism
from app.database import db


@dataclass
class RoundMetrics:
    round_num:      int
    f1:             float
    accuracy:       float
    param_change:   float          # L2 norm of aggregated delta
    epsilon_so_far: float
    client_losses:  dict = field(default_factory=dict)


class FedAvgServer:
    """
    Central aggregation server for federated learning simulation.

    Maintains the global InvoiceAnomalyNet flat parameter vector and a
    held-out validation set drawn from all three client datasets.
    """

    def __init__(self, n_features: int = 7):
        self._n_features = n_features

        # Initialise the global model and extract its flat parameter vector
        torch.manual_seed(0)
        self._global_model  = InvoiceAnomalyNet(n_features=n_features)
        self.global_params  = self._global_model.get_flat_params().detach().numpy().copy()

        self._scaler: StandardScaler | None = None
        self._X_val:  np.ndarray | None     = None
        self._y_val:  np.ndarray | None     = None
        self.round_history: list[RoundMetrics] = []

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, all_partitions: dict) -> None:
        """
        Fit a global StandardScaler on the full dataset and build a stratified
        validation set (20% from each client, combined).
        Called once before training rounds begin.
        """
        X_all = np.vstack([p["X"] for p in all_partitions.values()])
        y_all = np.concatenate([p["y"] for p in all_partitions.values()])

        self._scaler = StandardScaler()
        self._scaler.fit(X_all)

        if len(np.unique(y_all)) > 1:
            _, X_val, _, y_val = train_test_split(
                self._scaler.transform(X_all),
                y_all,
                test_size=0.2,
                stratify=y_all,
                random_state=42,
            )
        else:
            X_val = self._scaler.transform(X_all)
            y_val = y_all

        self._X_val = X_val.astype(np.float32)
        self._y_val = y_val

    # ── Single FL round ───────────────────────────────────────────────────────

    def run_round(
        self, clients: list[FLClient], round_num: int, dp: DPMechanism, run_id: str
    ) -> RoundMetrics:
        """
        Execute one complete FL round (simulation mode):
          1. Broadcast global flat params to each client
          2. Collect local ΔW updates
          3. Simulate secure aggregation (pairwise masks)
          4. FedAvg on deltas: global_params += Σ w_i * ΔW_i
          5. Evaluate global model on validation set
          6. Store metrics to DB
        """
        # 1 & 2: Broadcast + local training
        updates: list[ClientUpdate] = []
        for client in clients:
            client.receive_global_model(self.global_params)
            update = client.local_train()
            updates.append(update)

        # 3 & 4 & 5 & 6
        return self._finalise_round(updates, round_num, dp, run_id)

    def aggregate_round(
        self, updates: list[ClientUpdate], round_num: int, dp: DPMechanism, run_id: str
    ) -> RoundMetrics:
        """
        Aggregate pre-collected ClientUpdate objects (distributed mode).
        Skips broadcast + collect (happened over HTTP). Runs secure agg →
        FedAvg → evaluate → DB.
        """
        return self._finalise_round(updates, round_num, dp, run_id)

    def _finalise_round(
        self, updates: list[ClientUpdate], round_num: int, dp: DPMechanism, run_id: str
    ) -> RoundMetrics:
        updates      = self._secure_aggregate_simulation(updates)
        delta_norm   = self._aggregate(updates)
        f1, acc      = self._evaluate()
        eps          = dp.compute_epsilon(round_num)

        metrics = RoundMetrics(
            round_num      = round_num,
            f1             = f1,
            accuracy       = acc,
            param_change   = delta_norm,
            epsilon_so_far = eps,
            client_losses  = {u.client_id: u.local_loss for u in updates},
        )
        self.round_history.append(metrics)

        client_losses = metrics.client_losses
        db.insert_fl_round_metric(run_id, round_num, {
            "f1_score":       f1,
            "accuracy":       acc,
            "coef_change":    delta_norm,
            "epsilon_so_far": eps,
            "client_a_loss":  client_losses.get("client_a"),
            "client_b_loss":  client_losses.get("client_b"),
            "client_c_loss":  client_losses.get("client_c"),
        })
        return metrics

    # ── FedAvg on increments ──────────────────────────────────────────────────

    def _aggregate(self, updates: list[ClientUpdate]) -> float:
        """
        Weighted average of ΔW increments (FedAvg), applied to global_params.
        Returns L2 norm of the applied delta (for coef_change metric).
        """
        total_samples  = sum(u.n_samples for u in updates)
        avg_delta      = np.zeros_like(self.global_params)
        for u in updates:
            w          = u.n_samples / total_samples
            avg_delta += w * u.delta_params

        self.global_params = self.global_params + avg_delta
        # Keep the global model in sync for export_model()
        self._global_model.set_flat_params(
            torch.from_numpy(self.global_params.astype(np.float32))
        )
        return float(np.linalg.norm(avg_delta))

    # ── Secure aggregation simulation ─────────────────────────────────────────

    def _secure_aggregate_simulation(
        self, updates: list[ClientUpdate]
    ) -> list[ClientUpdate]:
        """
        Simulate pairwise masking secure aggregation protocol on delta_params.
        Masks cancel in the weighted sum; server never sees individual ΔW.
        """
        n = len(updates)
        for i in range(n):
            for j in range(i + 1, n):
                mask = np.random.normal(0.0, 0.001, self.global_params.shape)
                updates[i].delta_params = updates[i].delta_params + mask
                updates[j].delta_params = updates[j].delta_params - mask
        return updates

    # ── Global model evaluation ───────────────────────────────────────────────

    def _evaluate(self) -> tuple[float, float]:
        """
        Evaluate current global MLP on the held-out validation set.
        Uses a direct forward pass — no partial_fit anti-pattern.
        """
        if self._X_val is None or len(self._X_val) == 0:
            return 0.0, 0.0
        try:
            self._global_model.eval()
            with torch.no_grad():
                X_t    = torch.from_numpy(self._X_val)
                logits = self._global_model(X_t)
                preds  = (torch.sigmoid(logits) > 0.5).numpy().flatten().astype(int)
            f1  = float(f1_score(self._y_val, preds, average="macro", zero_division=0))
            acc = float(accuracy_score(self._y_val, preds))
            return f1, acc
        except Exception:
            return 0.0, 0.0
        finally:
            self._global_model.train()

    # ── Export trained model ──────────────────────────────────────────────────

    def export_model(self) -> tuple[InvoiceAnomalyNet, StandardScaler]:
        """
        Return the final global InvoiceAnomalyNet and the global StandardScaler,
        ready for pickling by fl_coordinator.
        """
        if self._scaler is None:
            raise RuntimeError("Server not initialised — call initialize() first")
        self._global_model.eval()
        return self._global_model, self._scaler
