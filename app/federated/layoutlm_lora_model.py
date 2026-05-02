# LoRA adapter setup and flat-numpy serialisation for LayoutLMv3.
#
# Used by LayoutLMFLClient and LayoutLMFedAvgServer to:
#   - Load the fine-tuned LayoutLMv3 from data/layoutlm_invoice/ and wrap it
#     with PEFT LoRA adapters (only adapter weights are trainable / transmitted)
#   - Serialise / deserialise LoRA adapter parameters as a flat float32 numpy
#     array so FedAvg delta accumulation works identically to the MLP path
#   - Merge the trained LoRA adapters back into the base model after FL so
#     layoutlm_extractor.py can load the result with no code changes
#
# LoRA configuration:
#   target_modules = ["query", "value"]   (attention Q + V projections)
#   r = 8, lora_alpha = 16  (scaling = 2.0 — standard for fine-tuning)
#   lora_dropout = 0.05
#   Trainable params ≈ 295K floats ≈ 1.2 MB per FL round  (vs 481 MB full model)
#
# Flat-param ordering: peft_model.named_parameters() filtered to "lora_" names,
# iterated in insertion order (deterministic in Python dicts since 3.7).
# get_flat_lora_params and set_flat_lora_params MUST use the same filter and
# iteration order — do not change one without updating the other.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

_BASE      = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _BASE / "data" / "layoutlm_invoice"

# LoRA hyperparameters — kept as module constants so coordinator can override
LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["query", "value"]

LABEL2ID = {
    "O":              0,
    "B-VENDOR":       1,  "I-VENDOR":       2,
    "B-DATE":         3,  "I-DATE":         4,
    "B-TOTAL":        5,  "I-TOTAL":        6,
    "B-SUBTOTAL":     7,  "I-SUBTOTAL":     8,
    "B-INVOICE_NO":   9,  "I-INVOICE_NO":  10,
    "B-PO_NO":       11,  "I-PO_NO":       12,
    "B-CURRENCY":    13,  "I-CURRENCY":    14,
}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_base_with_lora(
    model_dir: str | Path | None = None,
    lora_adapter_path: str | Path | None = None,
    lora_r: int = LORA_R,
) -> tuple:
    """
    Load the fine-tuned LayoutLMv3 from *model_dir* and wrap it with LoRA.

    Parameters
    ----------
    model_dir
        Directory containing the fine-tuned LayoutLMv3 weights
        (config.json + model.safetensors).  Defaults to data/layoutlm_invoice/.
    lora_adapter_path
        If provided, load a previously-saved LoRA adapter instead of
        initialising fresh adapter matrices (A=random init, B=0 so the
        initial output is identical to the base model — standard PEFT).
    lora_r
        LoRA rank; overrides the module constant LORA_R.

    Returns
    -------
    (peft_model, processor)
    """
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    from peft import LoraConfig, get_peft_model, PeftModel

    base_dir = Path(model_dir) if model_dir else _MODEL_DIR

    processor = LayoutLMv3Processor.from_pretrained(str(base_dir))
    base_model = LayoutLMv3ForTokenClassification.from_pretrained(
        str(base_dir),
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if lora_adapter_path is not None:
        # Load previously-saved adapter on top of the base model
        peft_model = PeftModel.from_pretrained(base_model, str(lora_adapter_path))
    else:
        # Fresh LoRA adapters: B matrices initialised to 0 so output = base model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="TOKEN_CLS",
        )
        peft_model = get_peft_model(base_model, lora_config)

    return peft_model, processor


# ── Flat-numpy serialisation ───────────────────────────────────────────────────

def get_flat_lora_params(peft_model) -> np.ndarray:
    """
    Extract all LoRA adapter parameters (lora_A and lora_B weight tensors)
    as a contiguous flat float32 numpy array.

    Iteration order: peft_model.named_parameters() filtered to names that
    contain "lora_".  This order is deterministic and must match set_flat_lora_params.
    """
    parts = []
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            parts.append(param.data.cpu().float().detach().flatten())
    if not parts:
        raise RuntimeError(
            "No LoRA parameters found. Ensure the model was wrapped with get_peft_model()."
        )
    return torch.cat(parts).numpy()


def set_flat_lora_params(peft_model, flat: np.ndarray) -> None:
    """
    Write a flat float32 numpy array back into the LoRA adapter tensors in-place.

    Must iterate in the SAME ORDER as get_flat_lora_params.
    Raises AssertionError if the total number of parameters does not match.
    """
    flat_t = torch.from_numpy(flat.astype(np.float32))
    offset = 0
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            n = param.numel()
            param.data.copy_(flat_t[offset:offset + n].reshape(param.shape))
            offset += n
    if offset != len(flat_t):
        raise AssertionError(
            f"Shape mismatch: consumed {offset} params, expected {len(flat_t)}. "
            "Ensure model architecture has not changed between get and set calls."
        )


def count_lora_params(peft_model) -> int:
    """Return the number of trainable LoRA parameters."""
    return sum(
        p.numel() for name, p in peft_model.named_parameters()
        if "lora_" in name
    )


# ── Adapter persistence ────────────────────────────────────────────────────────

def save_lora_adapter(peft_model, save_dir: str | Path) -> None:
    """
    Save LoRA adapter weights via PEFT's save_pretrained.
    Creates adapter_config.json + adapter_model.safetensors in *save_dir*.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(save_dir))


# ── Merge LoRA into base (integration bridge) ─────────────────────────────────

def merge_lora_into_base(
    base_model_dir: str | Path,
    lora_adapter_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Load the base fine-tuned LayoutLMv3, apply the LoRA adapter, merge the
    adapter weights into the base weights via PEFT's merge_and_unload(), then
    save the resulting standalone LayoutLMv3ForTokenClassification to *output_dir*.

    The output is a drop-in replacement for data/layoutlm_invoice/ — it can be
    loaded by layoutlm_extractor.py via LayoutLMv3ForTokenClassification.from_pretrained()
    with no code changes.
    """
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    from peft import PeftModel

    print(f"[merge_lora] Loading base from {base_model_dir}")
    base_model = LayoutLMv3ForTokenClassification.from_pretrained(
        str(base_model_dir),
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print(f"[merge_lora] Loading LoRA adapter from {lora_adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, str(lora_adapter_dir))

    print("[merge_lora] Merging and unloading LoRA adapters...")
    merged_model = peft_model.merge_and_unload()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir))

    # Copy processor config from base dir
    processor = LayoutLMv3Processor.from_pretrained(str(base_model_dir))
    processor.save_pretrained(str(output_dir))

    print(f"[merge_lora] Merged model saved to {output_dir}")
