# FedAvg Server for LayoutLMv3 + LoRA federated training.
#
# Mirrors FedAvgServer (server.py) but operates on LoRA adapter flat-param
# vectors instead of the MLP flat-param vector.
#
# The FedAvg aggregation formula is identical:
#   global_params += Σ w_i * ΔW_i   where w_i = n_samples_i / total_samples
#
# The only parts that differ from FedAvgServer:
#   • global_params is the flat LoRA adapter vector (~295K floats, not 833)
#   • _evaluate() runs LayoutLMv3 token-classification inference, not MLP forward
#   • Per-class F1 is tracked in addition to macro F1
#   • DB logging uses a separate table prefix ("lora_") to avoid schema conflicts
#
# DPMechanism and ClientUpdate / LayoutLMClientUpdate are both compatible:
#   apply_flat() works on any 1-D numpy array regardless of length.
#   _aggregate() only reads .delta_params and .n_samples — both are present.
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.federated.layoutlm_lora_model import (
    load_base_with_lora,
    get_flat_lora_params,
    set_flat_lora_params,
    save_lora_adapter,
    LORA_R,
)
from app.federated.layoutlm_data_partitioner import (
    ImageSample,
    load_sample_words_boxes_labels,
    LABEL2ID,
)
from app.federated.layoutlm_fl_client import (
    LayoutLMClientUpdate,
    _InvoiceTokenDataset,
    _collate,
)
from app.federated.privacy import DPMechanism
from app.database import db


@dataclass
class LoRARoundMetrics:
    round_num:      int
    macro_f1:       float
    per_class_f1:   dict = field(default_factory=dict)
    param_change:   float = 0.0    # L2 norm of aggregated LoRA delta
    epsilon_so_far: float = 0.0
    client_losses:  dict = field(default_factory=dict)
    client_f1s:     dict = field(default_factory=dict)


class LayoutLMFedAvgServer:
    """
    Central aggregation server for federated LayoutLMv3 + LoRA training.

    Maintains the global LoRA adapter flat parameter vector and a held-out
    validation set drawn from all three client datasets.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device:    str = "cpu",
        lora_r:    int = LORA_R,
    ):
        self._device = torch.device(device)
        print("[LoRA-Server] Loading LayoutLMv3 + LoRA for global model...")
        self._peft_model, self._processor = load_base_with_lora(
            model_dir=model_dir,
            lora_r=lora_r,
        )
        self._peft_model = self._peft_model.to(self._device)
        self._peft_model.eval()

        # Global LoRA parameter vector (flat float32 numpy)
        self.global_params: np.ndarray = get_flat_lora_params(self._peft_model)
        print(f"[LoRA-Server] Global LoRA params: {len(self.global_params):,} floats "
              f"({len(self.global_params) * 4 / 1024:.1f} KB)")

        self._val_ds: _InvoiceTokenDataset | None = None
        self.round_history: list[LoRARoundMetrics] = []

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(
        self,
        all_partitions: dict[str, list[ImageSample]],
        val_fraction: float = 0.10,
        seed: int = 42,
    ) -> None:
        """
        Build a global held-out validation set by drawing val_fraction from
        each client's sample list.  Called once before training rounds begin.
        """
        rng = random.Random(seed)
        val_samples: list[ImageSample] = []

        for client_id, samples in all_partitions.items():
            n_val = max(1, int(len(samples) * val_fraction))
            shuffled = list(samples)
            rng.shuffle(shuffled)
            val_samples.extend(shuffled[:n_val])

        print(f"[LoRA-Server] Building validation set from {len(val_samples)} samples...")
        self._val_ds = _InvoiceTokenDataset(val_samples, self._processor)
        print(f"[LoRA-Server] Validation set ready: {len(self._val_ds)} encodings")

    # ── Single FL round ───────────────────────────────────────────────────────

    def run_round(
        self,
        clients:   list,    # list[LayoutLMFLClient]
        round_num: int,
        dp:        DPMechanism,
        run_id:    str,
        n_epochs:  int = 1,
    ) -> LoRARoundMetrics:
        """
        Execute one complete FL round (simulation mode):
          1. Broadcast global LoRA params to each client
          2. Collect local ΔW updates
          3. Simulate secure aggregation (pairwise masks, matching server.py)
          4. FedAvg on LoRA deltas
          5. Evaluate global model on held-out validation set
          6. Store metrics to DB
        """
        updates: list[LayoutLMClientUpdate] = []
        for client in clients:
            client.receive_global_model(self.global_params)
            update = client.local_train(n_epochs=n_epochs)
            updates.append(update)

        return self._finalise_round(updates, round_num, dp, run_id)

    def aggregate_round(
        self,
        updates:   list[LayoutLMClientUpdate],
        round_num: int,
        dp:        DPMechanism,
        run_id:    str,
    ) -> LoRARoundMetrics:
        """Aggregate pre-collected updates (for distributed mode)."""
        return self._finalise_round(updates, round_num, dp, run_id)

    def _finalise_round(
        self,
        updates:   list[LayoutLMClientUpdate],
        round_num: int,
        dp:        DPMechanism,
        run_id:    str,
    ) -> LoRARoundMetrics:
        updates    = self._secure_aggregate_simulation(updates)
        delta_norm = self._aggregate(updates)
        macro_f1, per_class_f1 = self._evaluate()
        eps        = dp.compute_epsilon(round_num)

        metrics = LoRARoundMetrics(
            round_num      = round_num,
            macro_f1       = macro_f1,
            per_class_f1   = per_class_f1,
            param_change   = delta_norm,
            epsilon_so_far = eps,
            client_losses  = {u.client_id: u.local_loss for u in updates},
            client_f1s     = {u.client_id: u.local_f1   for u in updates},
        )
        self.round_history.append(metrics)

        # Store to DB — uses "LORA-" prefix on run_id to separate from MLP runs
        try:
            db.insert_fl_round_metric(f"LORA-{run_id}", round_num, {
                "f1_score":          macro_f1,
                "accuracy":          macro_f1,     # no separate accuracy for NER
                "coef_change":       delta_norm,
                "epsilon_so_far":    eps,
                "client_a_loss":     metrics.client_losses.get("client_a"),
                "client_b_loss":     metrics.client_losses.get("client_b"),
                "client_c_loss":     metrics.client_losses.get("client_c"),
            })
        except Exception as e:
            print(f"[LoRA-Server] DB insert failed (non-fatal): {e}")

        vendor_f1  = per_class_f1.get("VENDOR",     per_class_f1.get("B-VENDOR",     0.0))
        date_f1    = per_class_f1.get("DATE",        per_class_f1.get("B-DATE",       0.0))
        total_f1   = per_class_f1.get("TOTAL",       per_class_f1.get("B-TOTAL",      0.0))
        print(
            f"[LoRA-Server] Round {round_num:>2}  "
            f"macro_F1={macro_f1:.4f}  "
            f"VENDOR={vendor_f1:.3f}  DATE={date_f1:.3f}  TOTAL={total_f1:.3f}  "
            f"ε={eps:.4f}  |Δ|={delta_norm:.5f}"
        )
        return metrics

    # ── FedAvg on LoRA increments ─────────────────────────────────────────────

    def _aggregate(self, updates: list[LayoutLMClientUpdate]) -> float:
        """
        Weighted average of ΔW increments (FedAvg), applied to global_params.
        Identical formula to FedAvgServer._aggregate() — just larger vector.
        Returns L2 norm of the applied delta.
        """
        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            return 0.0

        avg_delta = np.zeros_like(self.global_params)
        for u in updates:
            w         = u.n_samples / total_samples
            avg_delta = avg_delta + w * u.delta_params

        self.global_params = self.global_params + avg_delta

        # Keep the PEFT model in sync
        set_flat_lora_params(self._peft_model, self.global_params)

        return float(np.linalg.norm(avg_delta))

    # ── Secure aggregation simulation ─────────────────────────────────────────

    def _secure_aggregate_simulation(
        self, updates: list[LayoutLMClientUpdate]
    ) -> list[LayoutLMClientUpdate]:
        """
        Simulate pairwise masking secure aggregation protocol.
        Masks cancel in the weighted sum — server never sees individual ΔW.
        Matches FedAvgServer._secure_aggregate_simulation() logic exactly.
        """
        n = len(updates)
        for i in range(n):
            for j in range(i + 1, n):
                mask = np.random.normal(0.0, 0.001, self.global_params.shape)
                updates[i].delta_params = updates[i].delta_params + mask
                updates[j].delta_params = updates[j].delta_params - mask
        return updates

    # ── Global model evaluation ───────────────────────────────────────────────

    def _evaluate(self) -> tuple[float, dict[str, float]]:
        """
        Evaluate the current global LoRA model on the held-out validation set.
        Returns (macro_f1, per_class_f1_dict).
        """
        if self._val_ds is None or len(self._val_ds) == 0:
            return 0.0, {}

        self._peft_model.eval()
        try:
            loader = DataLoader(
                self._val_ds,
                batch_size=4,
                shuffle=False,
                collate_fn=_collate,
            )
            all_preds:  list[list[str]] = []
            all_labels: list[list[str]] = []
            id2label    = self._peft_model.config.id2label

            with torch.no_grad():
                for batch in loader:
                    labels_batch = batch.pop("labels", None)
                    if labels_batch is None:
                        labels_batch = batch.pop("label_ids", None)
                    batch  = {k: v.to(self._device) for k, v in batch.items()}
                    logits = self._peft_model(**batch).logits
                    preds  = logits.argmax(-1)

                    for i in range(preds.size(0)):
                        pred_seq = []
                        lbl_seq  = []
                        for j in range(preds.size(1)):
                            lbl = labels_batch[i, j].item() if labels_batch is not None else -100
                            if lbl == -100:
                                continue
                            pred_seq.append(id2label.get(preds[i, j].item(), "O"))
                            lbl_seq.append(id2label.get(lbl, "O"))
                        if pred_seq:
                            all_preds.append(pred_seq)
                            all_labels.append(lbl_seq)

            if not all_preds:
                return 0.0, {}

            try:
                from seqeval.metrics import f1_score, classification_report
                macro_f1  = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
                report    = classification_report(all_labels, all_preds, output_dict=True,
                                                  zero_division=0)
                per_class = {
                    k: round(float(v.get("f1-score", 0.0)), 4)
                    for k, v in report.items()
                    if isinstance(v, dict) and k not in ("micro avg", "macro avg",
                                                          "weighted avg", "accuracy")
                }
            except ImportError:
                flat_preds  = [p for seq in all_preds  for p in seq]
                flat_labels = [l for seq in all_labels for l in seq]
                correct     = sum(p == l for p, l in zip(flat_preds, flat_labels))
                macro_f1    = correct / max(len(flat_labels), 1)
                per_class   = {}

            return macro_f1, per_class

        except Exception as e:
            print(f"[LoRA-Server] Evaluation error: {e}")
            return 0.0, {}

    # ── Export ────────────────────────────────────────────────────────────────

    def export_adapter(self, save_dir: str | Path) -> None:
        """
        Write the current global LoRA adapter weights to *save_dir* via PEFT's
        save_pretrained.  Creates adapter_config.json + adapter_model.safetensors.
        """
        set_flat_lora_params(self._peft_model, self.global_params)
        save_lora_adapter(self._peft_model, save_dir)
        print(f"[LoRA-Server] Adapter saved to {save_dir}")
