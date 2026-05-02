# FL Coordinator for LayoutLMv3 + LoRA federated training.
#
# Mirrors the structure of fl_coordinator.py but for the LoRA training path:
#   partition_layoutlm_images → LayoutLMFLClient × 3 → LayoutLMFedAvgServer
#   → DPMechanism → export adapter → merge into base
#
# SIMULATION ONLY — transformer fine-tuning is too memory-intensive to run
# distributed across real HTTP clients.  This is standard practice in FL
# research (FedOCR, FedProx, etc.): partition the dataset across simulated
# clients and run the training in-process.  The FL algorithm is identical.
#
# run_layoutlm_fl_training() is called by the FastAPI endpoint
# POST /api/fl/layoutlm/train and also runnable as a standalone script:
#
#   python -m app.federated.layoutlm_fl_coordinator
#
# Default hyperparameters:
#   n_rounds   = 15   (enough for convergence curves)
#   sigma      = 0.5  (lower than MLP path — LoRA deltas are larger)
#   clip_norm  = 0.3  (calibrated to LoRA delta L2 norm after 1 epoch)
#   local_epochs = 1  (1 epoch per round reduces client drift on non-IID data)
#   lora_r     = 8
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from app.federated.layoutlm_data_partitioner import partition_layoutlm_images
from app.federated.layoutlm_fl_client import LayoutLMFLClient
from app.federated.layoutlm_fl_server import LayoutLMFedAvgServer
from app.federated.layoutlm_lora_model import merge_lora_into_base, LORA_R
from app.federated.privacy import DPMechanism
from app.database import db

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE             = Path(__file__).resolve().parent.parent.parent
_TRAIN_DIR        = _BASE / "data" / "real_invoices" / "train"
_MODEL_DIR        = _BASE / "data" / "layoutlm_invoice"
LORA_ADAPTER_DIR  = _BASE / "data" / "layoutlm_lora"
LORA_META_PATH    = LORA_ADAPTER_DIR / "fl_lora_metadata.json"
LORA_MERGED_DIR   = _BASE / "data" / "layoutlm_lora_merged"


def _to_python(obj):
    """Recursively convert numpy scalars / arrays to plain Python types for JSON."""
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, dict):          return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_python(v) for v in obj]
    return obj


def run_layoutlm_fl_training(
    n_rounds:          int   = 15,
    sigma:             float = 0.5,
    clip_norm:         float = 0.3,
    local_epochs:      int   = 1,
    lora_r:            int   = LORA_R,
    device:            str   = "cpu",
    auto_merge:        bool  = True,
    seed:              int   = 42,
    mu:                float = 0.01,
    train_dir:         str | Path | None = None,
    model_dir:         str | Path | None = None,
    progress_callback = None,   # callable(event: dict) — fired after each round
    stop_event        = None,   # threading.Event — set it to cancel mid-run
) -> dict:
    """
    Run federated LayoutLMv3 + LoRA training across 3 simulated clients.

    Parameters
    ----------
    n_rounds     : number of FL rounds
    sigma        : Gaussian noise multiplier for DP (noise_std = clip_norm × sigma)
    clip_norm    : L2 clip bound for LoRA delta (0.3 is calibrated for LoRA)
    local_epochs : epochs per round per client (1 recommended for transformers)
    lora_r       : LoRA rank
    device       : "cpu" or "cuda"
    auto_merge   : if True, merge LoRA adapter into base after training
    seed         : random seed for reproducibility

    Returns
    -------
    dict with run_id, n_rounds, final_macro_f1, per_class_f1_final,
    round_history, epsilon, privacy_report, adapter_saved_to,
    merged_model_path (if auto_merge).
    """
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start   = time.time()
    base_dir  = Path(model_dir)  if model_dir  else _MODEL_DIR
    data_dir  = Path(train_dir)  if train_dir  else _TRAIN_DIR

    print(f"\n{'='*60}")
    print(f" Federated LayoutLMv3 + LoRA Training  (run_id={run_id})")
    print(f"  Rounds={n_rounds}  σ={sigma}  clip={clip_norm}  "
          f"epochs/round={local_epochs}  LoRA_r={lora_r}  device={device}")
    print(f"{'='*60}\n")

    # ── 1. Partition data ─────────────────────────────────────────────────────
    print("[1/5] Partitioning invoice images...")
    partitions = partition_layoutlm_images(train_dir=data_dir, seed=seed)
    for cid, samples in partitions.items():
        xml_n = sum(1 for s in samples if s.source == "xml")
        txt_n = sum(1 for s in samples if s.source == "txt")
        print(f"  {cid}: {len(samples)} images  (xml={xml_n}, txt={txt_n})")

    # ── 2. Build DP mechanism ──────────────────────────────────────────────────
    dp = DPMechanism(
        clip_norm     = clip_norm,
        sigma         = sigma,
        delta         = 1e-5,
        sampling_rate = 3 / max(len(partitions), 1),  # 3 clients per round
    )
    print(f"\n[2/5] DP: clip_norm={clip_norm}  σ={sigma}  "
          f"estimated ε@{n_rounds} rounds = {dp.compute_epsilon(n_rounds):.4f}\n")

    # ── 3. Initialise server and clients ──────────────────────────────────────
    print("[3/5] Initialising server and clients (loads LayoutLMv3 × 4)...")
    server = LayoutLMFedAvgServer(model_dir=base_dir, device=device, lora_r=lora_r)
    server.initialize(partitions, val_fraction=0.10, seed=seed)

    # Share the server's model across all clients — reduces peak RAM from
    # 4× model loads to 1×. Clients train sequentially so sharing is safe.
    clients = []
    for client_id, samples in partitions.items():
        client = LayoutLMFLClient(
            client_id          = client_id,
            samples            = samples,
            dp_mechanism       = dp,
            model_dir          = base_dir,
            device             = device,
            seed               = seed,
            lora_r             = lora_r,
            mu                 = mu,
            shared_peft_model  = server._peft_model,
            shared_processor   = server._processor,
        )
        clients.append(client)

    # ── 4. FL rounds ──────────────────────────────────────────────────────────
    print(f"\n[4/5] Running {n_rounds} FL rounds...\n")

    # Log run to DB
    try:
        db.insert_fl_run(run_id, {
            "model_type":    "layoutlm_lora",
            "n_rounds":      n_rounds,
            "sigma":         sigma,
            "clip_norm":     clip_norm,
            "local_epochs":  local_epochs,
            "lora_r":        lora_r,
            "mu":            mu,
        })
    except Exception:
        pass  # DB logging is non-fatal

    if progress_callback:
        progress_callback({
            "type":         "started",
            "run_id":       run_id,
            "total_rounds": n_rounds,
            "sigma":        sigma,
            "clip_norm":    clip_norm,
            "lora_r":       lora_r,
        })

    round_history = []
    stopped_early = False
    for rnd in range(1, n_rounds + 1):
        if stop_event and stop_event.is_set():
            print(f"[fl_coordinator] Stop requested before round {rnd}. Stopping.")
            stopped_early = True
            break

        metrics = server.run_round(
            clients   = clients,
            round_num = rnd,
            dp        = dp,
            run_id    = run_id,
            n_epochs  = local_epochs,
        )
        entry = {
            "round":         rnd,
            "macro_f1":      metrics.macro_f1,
            "per_class_f1":  metrics.per_class_f1,
            "param_change":  metrics.param_change,
            "epsilon":       metrics.epsilon_so_far,
            "client_losses": metrics.client_losses,
            "client_f1s":    metrics.client_f1s,
        }
        round_history.append(entry)

        if progress_callback:
            progress_callback({
                "type":          "round_complete",
                "round":         rnd,
                "total_rounds":  n_rounds,
                "macro_f1":      metrics.macro_f1,
                "per_class_f1":  _to_python(metrics.per_class_f1),
                "param_change":  float(metrics.param_change),
                "epsilon":       float(metrics.epsilon_so_far),
                "client_losses": _to_python(metrics.client_losses),
                "client_f1s":    _to_python(metrics.client_f1s),
                "elapsed_s":     round(time.time() - t_start, 1),
            })

    # ── 5. Save artifacts ─────────────────────────────────────────────────────
    if progress_callback:
        evt_type = "stopped" if stopped_early else "done"
        final_m  = round_history[-1] if round_history else {}
        progress_callback({
            "type":           evt_type,
            "run_id":         run_id,
            "rounds_done":    len(round_history),
            "total_rounds":   n_rounds,
            "final_macro_f1": final_m.get("macro_f1", 0.0),
            "epsilon":        final_m.get("epsilon", 0.0),
            "elapsed_s":      round(time.time() - t_start, 1),
        })
    print(f"\n[5/5] Saving adapter to {LORA_ADAPTER_DIR}...")
    LORA_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    server.export_adapter(LORA_ADAPTER_DIR)

    final_metrics = round_history[-1] if round_history else {}
    privacy_rpt   = dp.privacy_report(n_rounds)

    metadata = {
        "run_id":             run_id,
        "timestamp":          datetime.now().isoformat(),
        "n_rounds":           n_rounds,
        "final_macro_f1":     final_metrics.get("macro_f1", 0.0),
        "per_class_f1_final": final_metrics.get("per_class_f1", {}),
        "epsilon":            privacy_rpt["epsilon"],
        "delta":              privacy_rpt["delta"],
        "sigma":              sigma,
        "clip_norm":          clip_norm,
        "lora_r":             lora_r,
        "local_epochs":       local_epochs,
        "mu":                 mu,
        "device":             device,
        "training_seconds":   round(time.time() - t_start, 1),
        "client_sizes": {
            cid: len(samples) for cid, samples in partitions.items()
        },
        "privacy_report":  privacy_rpt,
        "round_history":   round_history,
    }
    LORA_META_PATH.write_text(
        json.dumps(_to_python(metadata), indent=2), encoding="utf-8"
    )
    print(f"  Metadata saved to {LORA_META_PATH}")

    # ── 6. Merge (optional) ───────────────────────────────────────────────────
    merged_path = None
    if auto_merge:
        print(f"\n[6/6] Merging LoRA adapter into base model → {LORA_MERGED_DIR}...")
        try:
            merge_lora_into_base(
                base_model_dir   = base_dir,
                lora_adapter_dir = LORA_ADAPTER_DIR,
                output_dir       = LORA_MERGED_DIR,
            )
            merged_path = str(LORA_MERGED_DIR)
            print(f"  Merged model ready at {LORA_MERGED_DIR}")
        except Exception as e:
            print(f"  Merge failed (non-fatal): {e}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f" Training complete in {elapsed:.0f}s  "
          f"final_macro_F1={metadata['final_macro_f1']:.4f}  "
          f"ε={metadata['epsilon']:.4f}")
    print(f"{'='*60}\n")

    result = {
        "run_id":             run_id,
        "n_rounds":           n_rounds,
        "final_macro_f1":     metadata["final_macro_f1"],
        "per_class_f1_final": metadata["per_class_f1_final"],
        "round_history":      round_history,
        "epsilon":            metadata["epsilon"],
        "privacy_report":     privacy_rpt,
        "adapter_saved_to":   str(LORA_ADAPTER_DIR),
        "merged_model_path":  merged_path,
        "training_seconds":   metadata["training_seconds"],
    }
    return _to_python(result)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Federated LayoutLMv3 + LoRA training")
    parser.add_argument("--rounds",       type=int,   default=15)
    parser.add_argument("--sigma",        type=float, default=0.5)
    parser.add_argument("--clip-norm",    type=float, default=0.3, dest="clip_norm")
    parser.add_argument("--local-epochs", type=int,   default=1,   dest="local_epochs")
    parser.add_argument("--lora-r",       type=int,   default=8,    dest="lora_r")
    parser.add_argument("--mu",           type=float, default=0.01)
    parser.add_argument("--device",       type=str,   default="cpu")
    parser.add_argument("--no-merge",     action="store_true")
    args = parser.parse_args()

    run_layoutlm_fl_training(
        n_rounds     = args.rounds,
        sigma        = args.sigma,
        clip_norm    = args.clip_norm,
        local_epochs = args.local_epochs,
        lora_r       = args.lora_r,
        mu           = args.mu,
        device       = args.device,
        auto_merge   = not args.no_merge,
    )
