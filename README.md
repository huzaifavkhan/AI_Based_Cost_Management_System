# AI-Based Cost Management System

An intelligent invoice processing and verification platform built as a Final Year Project at IBA Karachi. The system automates the full accounts-payable pipeline — from raw document ingestion to anomaly detection — using a multi-tier AI extraction stack, three-way matching, and federated learning.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Data Instructions](#data-instructions)
- [Model Weights Instructions](#model-weights-instructions)
- [Example Usage](#example-usage)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Problem Statement

Manual invoice processing is slow, error-prone, and expensive. Organizations struggle with:
- Extracting structured fields from unstructured invoice documents (PDFs, images, scans)
- Matching invoices against Purchase Orders and Goods Receipts (three-way matching)
- Detecting fraudulent or anomalous invoices
- Doing all of the above while keeping sensitive financial data private

This system solves all four problems with a fully automated AI pipeline.

---

## Features

### Document Ingestion
- Accepts **PDF, PNG, JPG, TIFF, BMP, Excel (.xlsx/.xls), and CSV** invoices
- Handles text-based PDFs, scanned documents, and image invoices

### Multi-Tier AI Extraction
A cascading extraction pipeline — each tier activates only if the previous one returns incomplete results:

| Tier | Method | When Used |
|------|--------|-----------|
| 1 | LayoutLMv3 (fine-tuned, local) | Primary extractor for all documents |
| 2 | Groq Vision LLM (LLaMA-4 Scout) | Cloud fallback when LayoutLM is incomplete |
| 3 | FL-LoRA LayoutLMv3 | Federated fine-tuned model (if trained and active) |
| 4 | Spatial PDF / GOT-OCR 2.0 | Layout-aware text extraction |
| 5 | pdfplumber / Tesseract OCR | Final fallback for scanned documents |

### Three-Way Matching
- Matches extracted invoice fields against **Purchase Orders** and **Goods Receipts**
- Detects discrepancies in vendor, amount, date, and PO number
- Returns match score and VERIFIED / FLAGGED / UNMATCHED status

### Anomaly Detection
- **Rule-based checks**: duplicate invoices, future dates, round-number amounts, missing PO
- **ML-based**: Isolation Forest outlier detection trained on historical invoices
- **Federated model**: privacy-preserving SGD model trained across simulated clients

### Federated Learning
- Simulated federated training with differential privacy (noise injection + gradient clipping)
- Supports both anomaly detection (IsolationForest → FL-SGD) and LayoutLMv3 + LoRA fine-tuning
- Real-time training progress via Server-Sent Events (SSE)

### Streamlit Dashboard
- Upload & process invoices interactively
- View processing results, match status, and anomaly flags
- Human review and feedback interface
- Audit log
- FL Training Monitor with live round metrics

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Backend API | FastAPI + Uvicorn |
| Dashboard UI | Streamlit |
| ML / NLP | LayoutLMv3, HuggingFace Transformers, PEFT (LoRA) |
| OCR | Tesseract OCR (pytesseract), pdfplumber, GOT-OCR 2.0 |
| Vision LLM | Groq API (LLaMA-4 Scout) |
| Anomaly Detection | scikit-learn (IsolationForest), custom FL-SGD |
| Federated Learning | Custom FL coordinator with differential privacy |
| Data | SQLite (runtime DB), pandas, CSV reference data |
| Deep Learning | PyTorch |
| Document Parsing | pdfplumber, opencv-python, Pillow |
| Fuzzy Matching | rapidfuzz |
| Language | Python 3.10+ |

---

## System Architecture

```
Invoice (PDF / Image / Excel)
        │
        ▼
┌─────────────────────────────────────────────┐
│           Extraction Pipeline               │
│  Tier 1: LayoutLMv3 (local fine-tuned)      │
│  Tier 2: Groq Vision LLM                    │
│  Tier 3: FL-LoRA LayoutLMv3                 │
│  Tier 4: Spatial Parser / GOT-OCR           │
│  Tier 5: pdfplumber / Tesseract             │
└────────────────┬────────────────────────────┘
                 │  Extracted Fields
                 ▼
┌─────────────────────────────────────────────┐
│         Three-Way Matching                  │
│   Invoice ↔ Purchase Order ↔ Goods Receipt  │
└────────────────┬────────────────────────────┘
                 │  Match Result
                 ▼
┌─────────────────────────────────────────────┐
│         Anomaly Detection                   │
│   Rule Checker + Isolation Forest / FL-SGD  │
└────────────────┬────────────────────────────┘
                 │  Final Status
                 ▼
       VERIFIED / FLAGGED / UNMATCHED
                 │
                 ▼
        SQLite DB + Streamlit Dashboard
```

---

## Project Structure

```
cost-management-system/
├── app/                          # FastAPI backend
│   ├── main.py                   # API entry point, all endpoints
│   ├── database/
│   │   └── db.py                 # SQLite operations
│   ├── detection/
│   │   ├── ml_detector.py        # Isolation Forest + FL-SGD anomaly detection
│   │   └── rule_checker.py       # Rule-based anomaly flags
│   ├── extraction/
│   │   ├── layoutlm_extractor.py # LayoutLMv3 two-stage extractor (primary)
│   │   ├── layoutlm_fl_extractor.py # FL-LoRA LayoutLMv3 extractor
│   │   ├── groq_ocr.py           # Groq Vision LLM extractor
│   │   ├── ocr_engine.py         # Tesseract OCR wrapper
│   │   ├── pdf_parser.py         # pdfplumber text extraction
│   │   ├── spatial_pdf_parser.py # Spatial grid-projection PDF parser
│   │   ├── got_ocr.py            # GOT-OCR 2.0 vision model
│   │   ├── excel_parser.py       # Excel/CSV invoice parser
│   │   ├── field_extractor.py    # Regex-based field extraction
│   │   ├── currency_detector.py  # Currency symbol detection
│   │   └── preprocessor.py       # Image preprocessing for OCR
│   ├── federated/
│   │   ├── fl_coordinator.py     # Anomaly detection FL coordinator
│   │   ├── layoutlm_fl_coordinator.py # LayoutLMv3 FL coordinator
│   │   ├── layoutlm_fl_client.py # FL client for LayoutLM training
│   │   ├── layoutlm_fl_server.py # FL server for LayoutLM aggregation
│   │   ├── layoutlm_lora_model.py # LoRA model wrapper
│   │   ├── privacy.py            # Differential privacy utilities
│   │   └── model.py              # Base FL model definitions
│   ├── matching/
│   │   ├── three_way_match.py    # Invoice ↔ PO ↔ GR matching
│   │   └── fuzzy_matcher.py      # Fuzzy string matching utilities
│   └── models/
│       └── schemas.py            # Pydantic request/response schemas
├── dashboard/
│   └── app.py                    # Streamlit dashboard (main UI)
├── data/
│   ├── README.md                 # Data download instructions
│   ├── purchase_orders.csv       # Reference PO data
│   ├── goods_receipts.csv        # Reference GR data
│   ├── historical_invoices.csv   # Historical data for anomaly detection
│   ├── layoutlm_invoice/         # Fine-tuned LayoutLMv3 model files
│   ├── layoutlm_lora/            # LoRA adapter weights + metadata
│   ├── layoutlm_lora_merged/     # Merged FL-LoRA model files
│   └── fl_model/                 # Federated anomaly detection model
├── scripts/
│   ├── generate_synthetic_data.py  # Generate synthetic invoice CSV data
│   ├── generate_receipt_images.py  # Generate synthetic receipt images
│   ├── finetune_layoutlm_kaggle.py # LayoutLMv3 fine-tuning script
│   ├── fl_layoutlm_kaggle.py       # FL LayoutLMv3 training on Kaggle
│   ├── label_invoices.py           # Manual invoice labeling tool
│   └── generate_box_files.py       # Generate Tesseract .box files
├── tests/
│   ├── test_extraction.py
│   ├── test_matching.py
│   └── test_detection.py
├── fl_dashboard.html             # Federated learning dashboard (served at /fl-dashboard)
├── fl_simulation.html            # FL simulation viewer (served at /fl-simulation)
├── start_fl_clients.py           # Script to launch multiple FL clients
├── requirements.txt
└── .env                          # API keys (not committed)
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system
- Git

### 1. Clone the repository

```bash
git clone https://github.com/huzaifavkhan/AI_Based_Cost_Management_System.git
cd AI_Based_Cost_Management_System
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Tesseract path

The system auto-detects Tesseract at common install locations. If yours differs, edit the path list in [app/extraction/ocr_engine.py](app/extraction/ocr_engine.py) lines 12–16:

```python
_WINDOWS_PATHS = [
    r"D:\Extras\Tesseract\tesseract.exe",          # add your path here
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ...
]
```

On Linux/macOS, Tesseract on PATH is detected automatically.

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
XAI_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [https://console.groq.com/keys](https://console.groq.com/keys).

> The system works without a Groq key — it just won't use the cloud vision LLM fallback (Tier 2).

### 5. Download model weights

See [data/README.md](data/README.md) for download links and placement instructions.

The minimum required file is:
```
data/layoutlm_invoice/model.safetensors   (~480 MB)
```

Without it, the system skips Tier 1 and falls through to Groq / Tesseract.

---

## Running the Project

### Start the Backend API

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API runs at **http://localhost:8000**
Interactive API docs at **http://localhost:8000/docs**

### Start the Streamlit Dashboard

Open a second terminal:

```bash
python -m streamlit run dashboard/app.py
```

Dashboard opens at **http://localhost:8501**

> **Start the backend first.** The dashboard calls the API at `http://localhost:8000`.

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Data Instructions

Reference CSVs (`purchase_orders.csv`, `goods_receipts.csv`, `historical_invoices.csv`) are included in the repository and loaded automatically at startup.

For training datasets (not required for inference), see [data/README.md](data/README.md).

To regenerate synthetic training data locally:

```bash
python scripts/generate_synthetic_data.py
python scripts/generate_receipt_images.py
```

---

## Model Weights Instructions

Two model weight files are required for full functionality:

| Model | Path | Size | Required for |
|-------|------|------|-------------|
| Fine-tuned LayoutLMv3 | `data/layoutlm_invoice/model.safetensors` | ~480 MB | Tier 1 extraction |
| FL-LoRA merged model | `data/layoutlm_lora_merged/model.safetensors` | ~480 MB | Tier 3 extraction |

Download links are in [data/README.md](data/README.md).

After downloading, place each file in its respective directory. The system will detect and load them automatically on startup:

```
[startup] Pre-loading LayoutLMv3 model...
[startup] LayoutLMv3 ready.
```

If the model file is absent, the system logs a warning and falls back to the next extraction tier — it does not crash.

---

## Example Usage

### Process an invoice via API

```bash
curl -X POST http://localhost:8000/api/process-invoice \
  -F "file=@invoice.pdf"
```

**Response:**
```json
{
  "invoice_id": "A1B2C3D4",
  "filename": "invoice.pdf",
  "overall_status": "VERIFIED",
  "extraction_method": "layoutlm-local",
  "extracted_fields": {
    "invoice_number": "INV-2024-001",
    "vendor": "Acme Supplies Ltd.",
    "date": "01/15/2024",
    "po_number": "PO-5001",
    "total_amount": 12500.00
  },
  "match_result": {
    "status": "MATCHED",
    "match_score": 0.97,
    "discrepancies": []
  },
  "anomaly_result": {
    "rule_flags": [],
    "is_statistical_outlier": false,
    "anomaly_score": -0.12
  }
}
```

### Trigger Federated Learning (anomaly detection model)

```bash
curl -X POST "http://localhost:8000/api/fl/train" \
  -H "Content-Type: application/json" \
  -d '{"n_rounds": 10, "sigma": 0.1, "clip_norm": 1.0, "local_epochs": 3}'
```

### Use the Streamlit Dashboard

1. Open **http://localhost:8501**
2. Select **Upload & Process** from the sidebar
3. Upload a PDF, image, or Excel invoice
4. View extracted fields, match result, and anomaly flags
5. Use **Human Review** to correct and submit feedback

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/process-invoice` | Process an uploaded invoice file |
| GET | `/api/invoices` | List all processed invoices |
| GET | `/api/dashboard-stats` | Aggregated dashboard statistics |
| GET | `/api/audit-log` | Full audit log |
| POST | `/api/feedback` | Submit a field correction |
| POST | `/api/fl/train` | Start federated anomaly detection training |
| GET | `/api/fl/status` | FL model status |
| POST | `/api/fl/apply-model` | Switch active anomaly model |
| POST | `/api/fl/layoutlm/train-async` | Start async FL LayoutLMv3 training |
| GET | `/api/fl/layoutlm/stream` | SSE stream for live training progress |
| GET | `/fl-dashboard` | Federated learning HTML dashboard |

Full interactive docs: **http://localhost:8000/docs**

---

## Troubleshooting

### `TesseractNotFoundError`
Tesseract is not installed or the path is wrong.
- Install from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Add your install path to `_WINDOWS_PATHS` in `app/extraction/ocr_engine.py`

### `No Python at '...'` when running uvicorn
Your virtual environment was created on a different machine. Run directly with the system Python:
```bash
python -m uvicorn app.main:app --reload --port 8000
python -m streamlit run dashboard/app.py
```

### LayoutLMv3 not being used (falls back to Groq/Tesseract)
- Check that `data/layoutlm_invoice/model.safetensors` exists
- On first startup, the model takes 30–60 seconds to load — check console for `[startup] LayoutLMv3 ready.`
- If results are incomplete (< 3 fields extracted), the system automatically cascades to the next tier

### Groq always returning empty
- Ensure `XAI_API_KEY` is set in `.env`
- Restart the server after editing `.env` — the key is loaded at import time

### Dashboard shows no data
- Make sure the backend is running on port 8000 before opening the dashboard

### `Fatal error in launcher` on Windows
The venv `.exe` launchers have hardcoded paths. Use `python -m` prefix for all commands instead of direct executables.

---

## Future Improvements

- Multi-language invoice support with improved translation pipeline
- Support for additional document types (contracts, receipts with QR codes)
- HuggingFace Hub integration for automatic model weight downloads
- Dockerized deployment once confirmed working across machines
- User authentication and role-based access (reviewer vs. admin)
- Email/Slack notifications for flagged invoices
- Real federated learning across distributed nodes (currently simulated)
