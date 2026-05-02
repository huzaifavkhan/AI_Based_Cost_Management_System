# FastAPI backend — orchestrates the full invoice processing pipeline
from __future__ import annotations
import uuid
import shutil
import tempfile
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.extraction.pdf_parser         import extract_text_from_pdf
from app.extraction.ocr_engine         import extract_text_from_image
from app.extraction.spatial_pdf_parser import extract_text_spatial
from app.extraction.got_ocr            import extract_text_with_got_ocr
from app.extraction.groq_ocr           import (
    extract_fields_with_groq,
    extract_fields_from_pdf_with_groq,
)
from app.extraction.layoutlm_extractor import (
    extract_fields_with_layoutlm,
    extract_fields_from_pdf_with_layoutlm,
    is_available as layoutlm_available,
)
from app.extraction.excel_parser import parse_single_invoice_row
from app.extraction.field_extractor import extract_fields
from app.matching.three_way_match   import ThreeWayMatcher
from app.detection.rule_checker     import RuleChecker
from app.detection.ml_detector      import (
    IsolationForestDetector, build_features,
    get_active_detector, FL_ACTIVE_FLAG,
)
from app.federated.fl_coordinator import run_federated_training
from app.extraction.layoutlm_fl_extractor import (
    is_fl_model_available as fl_layoutlm_available,
    is_fl_model_active as fl_layoutlm_active,
    activate_fl_model as fl_layoutlm_activate,
    deactivate_fl_model as fl_layoutlm_deactivate,
    get_fl_model_metadata as fl_layoutlm_metadata,
    reload_fl_model as fl_layoutlm_reload,
)
from app.models.schemas import (
    ProcessingResponse, InvoiceFields, MatchResult,
    AnomalyResult, FeedbackRequest, DashboardStats,
    FLTrainRequest, ApplyModelRequest,
)
from app.database import db

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
PO_CSV   = BASE / "data" / "purchase_orders.csv"
GR_CSV   = BASE / "data" / "goods_receipts.csv"
HIST_CSV = BASE / "data" / "historical_invoices.csv"

# ── Module singletons (loaded once at startup) ───────────────────────────────
matcher      = ThreeWayMatcher(str(PO_CSV), str(GR_CSV))
rule_checker = RuleChecker()
ml_detector  = get_active_detector(str(HIST_CSV))
_fl_training_lock: bool = False   # prevents concurrent FL runs

# ── LayoutLM live-streaming state ────────────────────────────────────────────
_layoutlm_live: dict = {
    "is_training":     False,
    "run_id":          None,
    "current_round":   0,
    "total_rounds":    0,
    "round_history":   [],   # accumulated per-round events for polling clients
}
_layoutlm_stop_event: threading.Event = threading.Event()
_layoutlm_listeners: list[asyncio.Queue] = []
_layoutlm_executor   = ThreadPoolExecutor(max_workers=1)
_layoutlm_main_loop: asyncio.AbstractEventLoop | None = None


def _fields_incomplete(fields: dict | None) -> bool:
    """Return True if fields is missing or has >= 2 empty values — triggers fallback."""
    if not fields:
        return True
    empty = sum(1 for v in fields.values() if v is None or v == "" or v == 0)
    return empty >= 2


def _layoutlm_broadcast(event: dict) -> None:
    """Thread-safe broadcast: push event to every active SSE listener queue."""
    if event.get("type") == "round_complete":
        _layoutlm_live["current_round"] = event.get("round", 0)
        _layoutlm_live["round_history"].append(event)
    for q in list(_layoutlm_listeners):
        if _layoutlm_main_loop:
            _layoutlm_main_loop.call_soon_threadsafe(q.put_nowait, event)

# Pre-load LayoutLMv3 at startup so the first request doesn't time out
if layoutlm_available():
    from app.extraction.layoutlm_extractor import _load as _layoutlm_load
    print("[startup] Pre-loading LayoutLMv3 model...")
    _layoutlm_load()
    print("[startup] LayoutLMv3 ready.")

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Based Cost Management System",
    description="Automated invoice verification — FYP IBA Karachi",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

db.init_db()

FL_DASHBOARD    = BASE / "fl_dashboard.html"
FL_SIMULATION   = BASE / "fl_simulation.html"


@app.on_event("startup")
async def _capture_event_loop() -> None:
    global _layoutlm_main_loop
    _layoutlm_main_loop = asyncio.get_event_loop()


# ── GET /fl-dashboard ────────────────────────────────────────────────────────
@app.get("/fl-dashboard", include_in_schema=False)
def fl_dashboard():
    return FileResponse(FL_DASHBOARD, media_type="text/html")


# ── GET /fl-simulation ────────────────────────────────────────────────────────
@app.get("/fl-simulation", include_in_schema=False)
def fl_simulation():
    return FileResponse(FL_SIMULATION, media_type="text/html")


# ── POST /api/process-invoice ────────────────────────────────────────────────
@app.post("/api/process-invoice", response_model=ProcessingResponse)
async def process_invoice(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    invoice_id = str(uuid.uuid4())[:8].upper()

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # ── Extraction ───────────────────────────────────────────────────────
        if suffix in (".csv", ".xlsx", ".xls"):

            row = parse_single_invoice_row(tmp_path)
            fields = {
                "invoice_number": str(row.get("invoice_number") or row.get("invoice_no") or ""),
                "po_number":      str(row.get("po_number") or row.get("po_no") or ""),
                "vendor":         str(row.get("vendor") or row.get("vendor_name") or ""),
                "date":           str(row.get("date") or row.get("invoice_date") or ""),
                "total_amount":   _safe_float(row.get("total_amount") or row.get("grand_total")),
            }
            extraction_method = "excel"
        elif suffix == ".pdf":
            # Tier 1: Base LayoutLMv3 — primary extractor
            fields = extract_fields_from_pdf_with_layoutlm(tmp_path) if layoutlm_available() else None
            if not _fields_incomplete(fields):
                extraction_method = "layoutlm-local"
            else:
                # Tier 2: Groq Vision LLM
                fields = extract_fields_from_pdf_with_groq(tmp_path)
                if not _fields_incomplete(fields):
                    extraction_method = "groq-vision"
                else:
                    # Tier 3: FL-LoRA LayoutLMv3
                    if fl_layoutlm_active() and fl_layoutlm_available():
                        from app.extraction.layoutlm_fl_extractor import extract_fields_from_pdf_with_fl_layoutlm
                        fields = extract_fields_from_pdf_with_fl_layoutlm(tmp_path)
                        if not _fields_incomplete(fields):
                            extraction_method = "layoutlm-fl-lora"
                    if _fields_incomplete(fields):
                        # Tier 4: Spatial parser — LiteParse-equivalent grid projection
                        raw = extract_text_spatial(tmp_path)
                        if raw:
                            extraction_method = "spatial"
                        else:
                            # Tier 5: pdfplumber plain extraction
                            raw = extract_text_from_pdf(tmp_path)
                            extraction_method = "pdfplumber"
                        if not raw:
                            # Tier 6: Tesseract OCR — scanned/image-only PDFs
                            raw = _ocr_scanned_pdf(tmp_path)
                            extraction_method = "tesseract"
                        fields = extract_fields(raw)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            # Tier 1: Base LayoutLMv3 — primary extractor
            fields = extract_fields_with_layoutlm(tmp_path) if layoutlm_available() else None
            if not _fields_incomplete(fields):
                extraction_method = "layoutlm-local"
            else:
                # Tier 2: Groq Vision LLM
                fields = extract_fields_with_groq(tmp_path)
                if not _fields_incomplete(fields):
                    extraction_method = "groq-vision"
                else:
                    # Tier 3: FL-LoRA LayoutLMv3
                    if fl_layoutlm_active() and fl_layoutlm_available():
                        from app.extraction.layoutlm_fl_extractor import extract_fields_with_fl_layoutlm
                        fields = extract_fields_with_fl_layoutlm(tmp_path)
                        if not _fields_incomplete(fields):
                            extraction_method = "layoutlm-fl-lora"
                    if _fields_incomplete(fields):
                        # Tier 4: GOT-OCR2.0 — local VLM fallback
                        raw = extract_text_with_got_ocr(tmp_path)
                        if raw:
                            extraction_method = "got-ocr"
                        else:
                            # Tier 5: Tesseract — spatial reconstruction fallback
                            raw = extract_text_from_image(tmp_path, preprocess=True)
                            extraction_method = "tesseract"
                        fields = extract_fields(raw)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        # ── Three-Way Matching ───────────────────────────────────────────────
        match_result = matcher.match_invoice(fields)

        # ── Anomaly Detection ────────────────────────────────────────────────
        rule_flags = rule_checker.check(fields, match_result.get("po"))
        inv_features = build_features(fields, match_result.get("po"))
        ml_result = ml_detector.predict(inv_features)

        # ── Combine & decide overall status ──────────────────────────────────
        has_anomaly = bool(rule_flags) or ml_result["is_anomaly"]
        if match_result["status"] == "UNMATCHED":
            overall_status = "UNMATCHED"
        elif match_result["status"] == "FLAGGED" or has_anomaly:
            overall_status = "FLAGGED"
        else:
            overall_status = "VERIFIED"

        # ── Build response ───────────────────────────────────────────────────
        response = ProcessingResponse(
            invoice_id=invoice_id,
            filename=file.filename,
            overall_status=overall_status,
            extraction_method=extraction_method,
            extracted_fields=InvoiceFields(
                invoice_number=fields.get("invoice_number"),
                vendor=fields.get("vendor"),
                date=fields.get("date"),
                po_number=fields.get("po_number"),
                total_amount=_safe_float(fields.get("total_amount")),
            ),
            match_result=MatchResult(
                status=match_result["status"],
                discrepancies=match_result["discrepancies"],
                match_score=match_result["match_score"],
                po=match_result.get("po"),
                gr=match_result.get("gr"),
            ),
            anomaly_result=AnomalyResult(
                rule_flags=rule_flags,
                is_statistical_outlier=ml_result["is_anomaly"],
                anomaly_score=ml_result["anomaly_score"],
            ),
        )

        db.insert_invoice(response.model_dump())
        return response

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] process_invoice failed:\n{tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── GET /api/invoices ─────────────────────────────────────────────────────────
@app.get("/api/invoices")
def get_invoices():
    return db.get_all_invoices()


# ── POST /api/feedback ────────────────────────────────────────────────────────
@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    db.insert_feedback(req.invoice_id, req.field_name, req.corrected_value, req.user or "reviewer")
    return {"status": "ok", "message": "Feedback recorded"}


# ── GET /api/dashboard-stats ──────────────────────────────────────────────────
@app.get("/api/dashboard-stats", response_model=DashboardStats)
def dashboard_stats():
    return db.get_dashboard_stats()


# ── GET /api/audit-log ────────────────────────────────────────────────────────
@app.get("/api/audit-log")
def audit_log():
    return db.get_audit_log()


# ── POST /api/fl/train ────────────────────────────────────────────────────────
@app.post("/api/fl/train")
async def fl_train(req: FLTrainRequest):
    global _fl_training_lock, ml_detector
    if _fl_training_lock:
        raise HTTPException(status_code=409, detail="FL training already running")
    _fl_training_lock = True
    try:
        result = run_federated_training(
            n_rounds=req.n_rounds,
            sigma=req.sigma,
            clip_norm=req.clip_norm,
            local_epochs=req.local_epochs,
        )
        return JSONResponse(result)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[FL ERROR]\n{tb}")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {str(exc)}")
    finally:
        _fl_training_lock = False


# ── GET /api/fl/status ────────────────────────────────────────────────────────
@app.get("/api/fl/status")
def fl_status():
    import json
    active = FL_ACTIVE_FLAG.exists() and (BASE / "data" / "fl_model" / "fl_model.pkl").exists()
    meta_path = BASE / "data" / "fl_model" / "fl_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    latest = db.get_latest_fl_run()
    return {"fl_model_active": active, **meta, "latest_run": latest}


# ── GET /api/fl/history ───────────────────────────────────────────────────────
@app.get("/api/fl/history")
def fl_history():
    runs = db.get_fl_runs()
    for r in runs:
        r["round_metrics"] = db.get_fl_round_metrics(r["run_id"])
    return {"runs": runs}


# ── POST /api/fl/apply-model ──────────────────────────────────────────────────
@app.post("/api/fl/apply-model")
def apply_fl_model(req: ApplyModelRequest):
    global ml_detector
    if req.use_fl_model:
        FL_ACTIVE_FLAG.touch()
    else:
        FL_ACTIVE_FLAG.unlink(missing_ok=True)
    ml_detector = get_active_detector(str(HIST_CSV))
    active_model = "fl_sgd" if req.use_fl_model else "isolation_forest"
    return {"status": "ok", "active_model": active_model}


# ── POST /api/fl/layoutlm/train ───────────────────────────────────────────────
_lora_training_lock: bool = False

@app.post("/api/fl/layoutlm/train")
async def fl_layoutlm_train(
    n_rounds:     int   = 15,
    sigma:        float = 0.5,
    clip_norm:    float = 0.3,
    local_epochs: int   = 1,
    lora_r:       int   = 8,
    device:       str   = "cpu",
):
    """
    Launch federated LayoutLMv3 + LoRA training.
    Runs synchronously (can take hours on CPU) — call from a background task
    or run the coordinator directly for long training runs.
    """
    global _lora_training_lock
    if _lora_training_lock:
        raise HTTPException(status_code=409, detail="LoRA FL training already in progress")
    _lora_training_lock = True
    try:
        from app.federated.layoutlm_fl_coordinator import run_layoutlm_fl_training
        result = run_layoutlm_fl_training(
            n_rounds     = n_rounds,
            sigma        = sigma,
            clip_norm    = clip_norm,
            local_epochs = local_epochs,
            lora_r       = lora_r,
            device       = device,
            auto_merge   = True,
        )
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _lora_training_lock = False


# ── POST /api/fl/layoutlm/apply-model ────────────────────────────────────────

@app.post("/api/fl/layoutlm/apply-model")
def fl_layoutlm_apply(use_fl_model: bool = True):
    """
    Toggle the FL-LoRA extraction model on or off.
    When on, /api/process-invoice routes through the merged LoRA model.
    """
    if use_fl_model:
        if not fl_layoutlm_available():
            raise HTTPException(
                status_code=404,
                detail="No merged FL-LoRA model found. Run /api/fl/layoutlm/train first."
            )
        fl_layoutlm_activate()
        fl_layoutlm_reload()
        return {"status": "ok", "fl_layoutlm_active": True}
    else:
        fl_layoutlm_deactivate()
        return {"status": "ok", "fl_layoutlm_active": False}


# ── GET /api/fl/layoutlm/status ──────────────────────────────────────────────

@app.get("/api/fl/layoutlm/status")
def fl_layoutlm_status():
    """Return FL-LoRA model availability, active state, and training metadata."""
    meta = fl_layoutlm_metadata() or {}
    return {
        "fl_model_available": fl_layoutlm_available(),
        "fl_model_active":    fl_layoutlm_active(),
        **meta,
    }


# ── GET /api/fl/layoutlm/live-status ─────────────────────────────────────────
@app.get("/api/fl/layoutlm/live-status")
def fl_layoutlm_live_status():
    """Return the current real-time training state (used by frontend on mount)."""
    return dict(_layoutlm_live)


# ── GET /api/fl/layoutlm/stream ───────────────────────────────────────────────
@app.get("/api/fl/layoutlm/stream")
async def fl_layoutlm_stream(request: Request):
    """
    Server-Sent Events endpoint that streams per-round training events.
    Each event is a JSON object on a `data:` line.
    The stream closes when a 'done' or 'stopped' event is received, or the
    client disconnects.
    """
    q: asyncio.Queue = asyncio.Queue()
    _layoutlm_listeners.append(q)

    async def generator():
        try:
            # Immediately send current state so a reconnecting client can sync.
            yield {"data": json.dumps({"type": "status", **_layoutlm_live})}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield {"data": json.dumps(event)}
                    if event.get("type") in ("done", "stopped"):
                        break
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"type": "heartbeat"})}
        finally:
            if q in _layoutlm_listeners:
                _layoutlm_listeners.remove(q)

    return EventSourceResponse(generator())


# ── POST /api/fl/layoutlm/train-async ────────────────────────────────────────
@app.post("/api/fl/layoutlm/train-async")
async def fl_layoutlm_train_async(
    n_rounds:     int   = 5,
    sigma:        float = 0.5,
    clip_norm:    float = 0.3,
    local_epochs: int   = 1,
    lora_r:       int   = 8,
    device:       str   = "cpu",
):
    """
    Start federated LayoutLMv3+LoRA training in the background.
    Returns immediately with a run_id; stream progress via GET /api/fl/layoutlm/stream.
    """
    global _lora_training_lock, _layoutlm_main_loop

    if _lora_training_lock or _layoutlm_live["is_training"]:
        raise HTTPException(status_code=409, detail="LoRA FL training already in progress")

    _lora_training_lock = True
    _layoutlm_stop_event.clear()
    run_id = f"LLMFL-{__import__('datetime').datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    _layoutlm_live.update({
        "is_training":   True,
        "run_id":        run_id,
        "current_round": 0,
        "total_rounds":  n_rounds,
        "round_history": [],
    })

    def _run():
        try:
            from app.federated.layoutlm_fl_coordinator import run_layoutlm_fl_training
            run_layoutlm_fl_training(
                n_rounds          = n_rounds,
                sigma             = sigma,
                clip_norm         = clip_norm,
                local_epochs      = local_epochs,
                lora_r            = lora_r,
                device            = device,
                auto_merge        = True,
                progress_callback = _layoutlm_broadcast,
                stop_event        = _layoutlm_stop_event,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[FL-ASYNC ERROR]\n{tb}")
            _layoutlm_broadcast({"type": "error", "message": str(exc)})
        finally:
            global _lora_training_lock
            _lora_training_lock = False
            _layoutlm_stop_event.clear()
            _layoutlm_live["is_training"] = False

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_layoutlm_executor, _run)

    return {"status": "started", "run_id": run_id}


# ── POST /api/fl/layoutlm/stop ───────────────────────────────────────────────
@app.post("/api/fl/layoutlm/stop")
def fl_layoutlm_stop():
    """Signal the running LoRA FL training to stop after the current round."""
    if not _layoutlm_live["is_training"]:
        return {"status": "idle", "message": "No training in progress"}
    _layoutlm_stop_event.set()
    return {"status": "stopping"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _ocr_scanned_pdf(pdf_path: str) -> str:
    """Convert scanned PDF pages to images and run Tesseract on each."""
    import pdfplumber
    from app.extraction.ocr_engine import extract_text_from_pil
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=200).original
            text_parts.append(extract_text_from_pil(img, preprocess=True))
    return "\n".join(text_parts)
