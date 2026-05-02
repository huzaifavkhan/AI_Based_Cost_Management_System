# FL Client for LayoutLMv3 + LoRA federated training.
#
# Mirrors the interface of FLClient (client.py) exactly:
#   __init__        → takes samples (list[ImageSample]) instead of tabular X, y
#   receive_global_model(flat_params) → writes LoRA adapter weights
#   local_train(n_epochs)             → trains LoRA, returns LayoutLMClientUpdate
#   evaluate_local()                  → returns {f1, loss, per_class_f1}
#
# The LayoutLMClientUpdate dataclass is structurally identical to ClientUpdate:
#   delta_params: np.ndarray  (flat LoRA ΔW, DP-noised)
#   n_samples:    int
#   local_f1:     float
#   local_loss:   float
# …so FedAvgServer._aggregate() accepts it unchanged.
#
# Privacy: DPMechanism.apply_flat() is called on the LoRA delta (clip_norm=0.3,
# sigma=0.5 recommended).  Recalibration vs. the MLP path is needed because
# LoRA deltas after 1 epoch are larger (~0.1-0.5 L2 norm) than MLP deltas.
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from app.federated.layoutlm_lora_model import (
    load_base_with_lora,
    get_flat_lora_params,
    set_flat_lora_params,
    LABEL2ID,
    LORA_R,
)
from app.federated.layoutlm_data_partitioner import (
    ImageSample,
    load_sample_words_boxes_labels,
)
from app.federated.privacy import DPMechanism

_MAX_LENGTH = 256


@dataclass
class LayoutLMClientUpdate:
    """
    Identical interface to ClientUpdate (client.py) — FedAvgServer._aggregate()
    uses only delta_params and n_samples, both present here.
    """
    client_id:    str
    delta_params: np.ndarray   # noisy ΔW of LoRA adapters only, flat float32
    n_samples:    int
    local_f1:     float        # macro F1 across all entity classes on val set
    local_loss:   float
    per_class_f1: dict = field(default_factory=dict)  # {VENDOR: 0.xx, DATE: 0.xx, ...}


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class _InvoiceTokenDataset(Dataset):
    """
    Preloads all samples as processor encodings so DataLoader batching is fast.
    Samples that fail to load are silently skipped at construction time.
    """

    def __init__(
        self,
        samples:   list[ImageSample],
        processor,
        max_length: int = _MAX_LENGTH,
    ):
        from PIL import Image

        self._encodings: list[dict] = []

        for sample in samples:
            result = load_sample_words_boxes_labels(sample)
            if result is None:
                continue
            words, boxes, bio_labels = result
            word_label_ids = [LABEL2ID.get(l, 0) for l in bio_labels]

            try:
                img = Image.open(sample.img_path).convert("RGB")
                encoding = processor(
                    img,
                    words,
                    boxes=boxes,
                    word_labels=word_label_ids,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                img.close()
                self._encodings.append({k: v.squeeze(0) for k, v in encoding.items()})
            except Exception:
                pass

    def __len__(self):
        return len(self._encodings)

    def __getitem__(self, idx):
        return self._encodings[idx]


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── FL Client ─────────────────────────────────────────────────────────────────

class LayoutLMFLClient:
    """
    Federated learning client that fine-tunes LoRA adapters of LayoutLMv3
    on a local set of annotated invoice images.

    Mirrors FLClient (client.py) interface exactly:
      receive_global_model(flat_params) → writes LoRA weights
      local_train(n_epochs=1)           → returns LayoutLMClientUpdate
      evaluate_local()                  → returns {f1, loss, per_class_f1}
    """

    def __init__(
        self,
        client_id:    str,
        samples:      list[ImageSample],
        dp_mechanism: DPMechanism,
        model_dir:    str | Path | None = None,
        device:       str = "cpu",
        seed:         int = 42,
        val_fraction: float = 0.15,
        batch_size:   int = 1,
        lora_r:       int = LORA_R,
        mu:           float = 0.01,
        shared_peft_model=None,
        shared_processor=None,
    ):
        self.client_id = client_id
        self.dp        = dp_mechanism
        self._device   = torch.device(device)
        self._seed     = seed
        self._mu       = mu

        random.seed(seed)
        torch.manual_seed(seed)

        # ── Load model + processor (or reuse shared instance to save memory) ──
        if shared_peft_model is not None:
            self._peft_model = shared_peft_model
            self._processor  = shared_processor
            print(f"[{client_id}] Using shared LayoutLMv3 model (memory-efficient mode)")
        else:
            print(f"[{client_id}] Loading LayoutLMv3 + LoRA (r={lora_r})...")
            self._peft_model, self._processor = load_base_with_lora(
                model_dir=model_dir,
                lora_r=lora_r,
            )
            self._peft_model = self._peft_model.to(self._device)

        # ── Train / val split ────────────────────────────────────────────────
        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n_val       = max(1, int(len(shuffled) * val_fraction))
        train_samps = shuffled[n_val:]
        val_samps   = shuffled[:n_val]

        # ── Build datasets (preloaded) ────────────────────────────────────────
        print(f"[{client_id}] Preprocessing {len(train_samps)} train "
              f"+ {len(val_samps)} val samples...")
        self._train_ds = _InvoiceTokenDataset(train_samps, self._processor)
        self._val_ds   = _InvoiceTokenDataset(val_samps,   self._processor)

        self._batch_size = batch_size
        print(f"[{client_id}] Ready: {len(self._train_ds)} train encodings, "
              f"{len(self._val_ds)} val encodings")

    # ── Receive global model ──────────────────────────────────────────────────

    def receive_global_model(self, flat_params: np.ndarray) -> None:
        """
        Overwrite local LoRA adapter parameters with the server's global flat
        vector.  Base LayoutLMv3 weights are NOT touched.
        """
        set_flat_lora_params(self._peft_model, flat_params)

    # ── Local training ────────────────────────────────────────────────────────

    def local_train(self, n_epochs: int = 1) -> LayoutLMClientUpdate:
        """
        Run n_epochs of local AdamW on the LoRA adapters, compute ΔW,
        apply DP (clip + Gaussian noise), return LayoutLMClientUpdate.

        n_epochs=1 is recommended for transformers in FL (reduces client drift
        on non-IID data; LoRA fine-tuning converges quickly).
        """
        if len(self._train_ds) == 0:
            # No training data — return a zero delta
            n_params = len(get_flat_lora_params(self._peft_model))
            return LayoutLMClientUpdate(
                client_id    = self.client_id,
                delta_params = np.zeros(n_params, dtype=np.float32),
                n_samples    = 0,
                local_f1     = 0.0,
                local_loss   = 0.0,
            )

        # Save global LoRA params before local training (used for ΔW and FedProx)
        global_flat = get_flat_lora_params(self._peft_model).copy()

        # Snapshot global tensors per-parameter for FedProx proximal term
        global_tensors = [
            p.data.clone()
            for n, p in self._peft_model.named_parameters()
            if "lora_" in n and p.requires_grad
        ]

        loader = DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=_collate,
        )

        # Only LoRA parameters receive gradients; base model is frozen
        trainable_params = [
            p for n, p in self._peft_model.named_parameters()
            if "lora_" in n and p.requires_grad
        ]
        optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

        self._peft_model.train()
        last_loss = 0.0

        for _ in range(n_epochs):
            epoch_loss = 0.0
            n_batches  = 0
            for batch in loader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self._peft_model(**batch)
                loss    = outputs.loss
                if self._mu > 0:
                    prox = sum(
                        ((p - g.to(self._device)) ** 2).sum()
                        for p, g in zip(trainable_params, global_tensors)
                    )
                    loss = loss + (self._mu / 2) * prox
                loss.backward()
                # Clip gradients to prevent exploding updates in LoRA layers
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches  += 1
            last_loss = epoch_loss / max(n_batches, 1)

        # ── Compute ΔW = W_local − W_global (LoRA params only) ───────────────
        local_flat = get_flat_lora_params(self._peft_model)
        delta_w    = local_flat - global_flat

        # ── Apply DP: clip ΔW then add Gaussian noise ─────────────────────────
        noisy_delta = self.dp.apply_flat(delta_w)

        # Evaluate on val set (model in inference mode)
        macro_f1, per_class = self._evaluate_f1()

        return LayoutLMClientUpdate(
            client_id    = self.client_id,
            delta_params = noisy_delta,
            n_samples    = len(self._train_ds),
            local_f1     = macro_f1,
            local_loss   = last_loss,
            per_class_f1 = per_class,
        )

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def _evaluate_f1(self) -> tuple[float, dict[str, float]]:
        """
        Run inference on the val set, compute token-level F1 per entity class
        and macro F1 across all non-O classes.

        Returns (macro_f1, {class_name: f1, ...}).
        """
        if len(self._val_ds) == 0:
            return 0.0, {}

        self._peft_model.eval()
        try:
            loader = DataLoader(
                self._val_ds,
                batch_size=self._batch_size,
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
                        # labels key may differ depending on peft version
                        labels_batch = batch.pop("label_ids", None)
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    logits = self._peft_model(**batch).logits
                    preds  = logits.argmax(-1)  # (B, seq_len)

                    for i in range(preds.size(0)):
                        pred_seq  = []
                        label_seq = []
                        for j in range(preds.size(1)):
                            lbl = labels_batch[i, j].item() if labels_batch is not None else -100
                            if lbl == -100:
                                continue
                            pred_seq.append(id2label.get(preds[i, j].item(), "O"))
                            label_seq.append(id2label.get(lbl, "O"))
                        if pred_seq:
                            all_preds.append(pred_seq)
                            all_labels.append(label_seq)

            if not all_preds:
                return 0.0, {}

            try:
                from seqeval.metrics import f1_score, classification_report
                macro_f1   = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
                report     = classification_report(all_labels, all_preds, output_dict=True,
                                                   zero_division=0)
                per_class  = {
                    k: round(float(v.get("f1-score", 0.0)), 4)
                    for k, v in report.items()
                    if isinstance(v, dict) and k not in ("micro avg", "macro avg",
                                                          "weighted avg", "accuracy")
                }
            except ImportError:
                # seqeval not available — fall back to flat token accuracy
                flat_preds  = [p for seq in all_preds  for p in seq]
                flat_labels = [l for seq in all_labels for l in seq]
                correct     = sum(p == l for p, l in zip(flat_preds, flat_labels))
                macro_f1    = correct / max(len(flat_labels), 1)
                per_class   = {}

            return macro_f1, per_class

        except Exception as e:
            print(f"[{self.client_id}] Evaluation error: {e}")
            return 0.0, {}
        finally:
            self._peft_model.train()

    def evaluate_local(self) -> dict:
        """Public evaluation summary (used by coordinator for reporting)."""
        macro_f1, per_class = self._evaluate_f1()
        return {
            "f1":           macro_f1,
            "loss":         0.0,   # not re-computed here to avoid extra forward pass
            "per_class_f1": per_class,
        }
