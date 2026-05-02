# Invoice Anomaly Detection MLP — shared by FL clients and the aggregation server.
#
# Replaces the linear SGDClassifier. With 7 features and ~800 parameters this
# is still tiny (< 3 KB per FL round) yet deep enough to learn the nonlinear
# interactions — amount × lag, unit_price outliers — that a linear model cannot.
from __future__ import annotations
import torch
import torch.nn as nn


class InvoiceAnomalyNet(nn.Module):
    """
    3-layer MLP binary classifier for invoice anomaly detection.

    Architecture (n_features=7):
        Linear(7→32) → ReLU → Dropout(0.2) → Linear(32→16) → ReLU → Linear(16→1)

    Output: raw logit.  sigmoid(logit) > 0.5  →  class 1 (anomaly).

    Parameter count: 7×32+32 + 32×16+16 + 16×1+1 = 833  (< 4 KB as float32)
    """

    def __init__(self, n_features: int = 7):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ── Flat-parameter helpers (used for FL serialisation) ────────────────────

    def get_flat_params(self) -> torch.Tensor:
        """All parameters as a single 1-D tensor (detached, no grad)."""
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Overwrite every parameter from a flat 1-D tensor."""
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[offset: offset + n].reshape(p.shape))
            offset += n

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
