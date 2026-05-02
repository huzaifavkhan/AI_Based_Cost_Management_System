# Pydantic data models for FastAPI request/response validation
from __future__ import annotations
from typing import Optional, List, Any
from pydantic import BaseModel


class InvoiceFields(BaseModel):
    invoice_number: Optional[str] = None
    vendor:         Optional[str] = None
    date:           Optional[str] = None
    po_number:      Optional[str] = None
    total_amount:   Optional[float] = None
    currency:       Optional[str] = None   # e.g. "$", "€", "RM", "£"


class MatchResult(BaseModel):
    status:        str                   # MATCHED | FLAGGED | UNMATCHED
    discrepancies: List[str] = []
    match_score:   int = 100             # 0–100
    po:            Optional[dict] = None
    gr:            Optional[dict] = None


class AnomalyResult(BaseModel):
    rule_flags:            List[str] = []
    is_statistical_outlier: bool = False
    anomaly_score:         float = 0.0


class ProcessingResponse(BaseModel):
    invoice_id:      str
    filename:        str
    overall_status:  str                 # VERIFIED | FLAGGED | UNMATCHED
    extracted_fields: InvoiceFields
    match_result:    MatchResult
    anomaly_result:  AnomalyResult
    extraction_method: str = "unknown"  # pdfplumber | tesseract | excel


class FeedbackRequest(BaseModel):
    invoice_id:      str
    field_name:      str
    corrected_value: str
    user:            Optional[str] = "reviewer"


class DashboardStats(BaseModel):
    total_processed: int
    verified_count:  int
    flagged_count:   int
    unmatched_count: int
    estimated_savings: float
    top_anomaly_types: List[dict] = []


class FLTrainRequest(BaseModel):
    n_rounds:     int   = 15
    sigma:        float = 1.0
    clip_norm:    float = 1.0
    local_epochs: int   = 3   # 3 epochs reduces client drift on non-IID data (FedOCR §3.1)


class ApplyModelRequest(BaseModel):
    use_fl_model: bool
