# Structured file parser for CSV and Excel invoices
import pandas as pd
from pathlib import Path


def parse_structured_file(file_path: str) -> list[dict]:
    """
    Parse a CSV or Excel file into a list of record dicts.
    For structured inputs, field extraction via regex is skipped —
    column names map directly to invoice fields.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df.to_dict(orient="records")


def parse_single_invoice_row(file_path: str, row_index: int = 0) -> dict:
    """
    Parse a single invoice from a structured file (for one-at-a-time upload).
    Returns the first row by default.
    """
    records = parse_structured_file(file_path)
    if not records:
        return {}
    return records[min(row_index, len(records) - 1)]
