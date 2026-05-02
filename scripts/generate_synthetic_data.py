"""
Synthetic data generator for the AI-Based Cost Management System.
Run once: python scripts/generate_synthetic_data.py
Produces: data/purchase_orders.csv, goods_receipts.csv, contracts.csv,
          invoices.csv, historical_invoices.csv, data/sample_invoices/*.pdf
"""
from __future__ import annotations
import random
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

# reportlab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
PDF_DIR = DATA / "sample_invoices"
DATA.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# ── Master data ──────────────────────────────────────────────────────────────

VENDORS = [
    {"id": "V001", "name": "Apex Office Supplies Ltd.",   "email": "billing@apexoffice.com",   "address": "12 Commerce St, Karachi"},
    {"id": "V002", "name": "TechCore Solutions Pvt.",     "email": "accounts@techcore.pk",      "address": "45 IT Tower, Lahore"},
    {"id": "V003", "name": "Green Planet Cleaning Co.",   "email": "invoice@greenplanet.com",   "address": "78 Industrial Zone, Faisalabad"},
    {"id": "V004", "name": "National Utilities Corp.",    "email": "billing@nationalutil.pk",   "address": "1 Power House Rd, Islamabad"},
    {"id": "V005", "name": "Premier Raw Materials Inc.",  "email": "ar@premierraw.com",         "address": "99 Factory Lane, Hyderabad"},
    {"id": "V006", "name": "ProServ Consulting Group",    "email": "finance@proserv.pk",        "address": "Suite 5, Dolmen Mall, Karachi"},
    {"id": "V007", "name": "FastTrack Logistics Ltd.",    "email": "billing@fasttrack.com",     "address": "22 Port Area, Karachi"},
    {"id": "V008", "name": "Horizon IT Equipment Co.",    "email": "invoices@horizonit.pk",     "address": "67 Tech Park, Lahore"},
]

CATEGORIES = {
    "Office Supplies":      {"items": ["A4 Paper Ream", "Ballpoint Pen Set", "Stapler", "File Folders", "Whiteboard Marker"], "price_range": (5, 80),   "qty_range": (10, 200), "vendor_ids": ["V001"]},
    "IT Equipment":         {"items": ["Laptop", "USB Hub", "Keyboard & Mouse Set", "Monitor 24\"", "Network Switch"], "price_range": (50, 2000), "qty_range": (1, 20),  "vendor_ids": ["V002", "V008"]},
    "Cleaning Supplies":    {"items": ["Floor Cleaner 5L", "Disinfectant Spray", "Mop & Bucket Set", "Trash Bags 100pc"], "price_range": (10, 120),  "qty_range": (5, 100), "vendor_ids": ["V003"]},
    "Utilities":            {"items": ["Electricity Bill", "Gas Bill", "Internet Service", "Water Supply"], "price_range": (200, 3000),"qty_range": (1, 1),   "vendor_ids": ["V004"]},
    "Raw Materials":        {"items": ["Steel Sheet 1m²", "PVC Pipe 3m", "Copper Wire Spool", "Cement Bag 50kg"], "price_range": (30, 500),  "qty_range": (20, 500),"vendor_ids": ["V005"]},
    "Professional Services":{"items": ["Consulting Day Rate", "Legal Advisory Fee", "Audit Services", "Training Session"], "price_range": (500, 5000),"qty_range": (1, 10),  "vendor_ids": ["V006"]},
}

PAYMENT_TERMS = ["Net 30", "Net 45", "Net 60", "Due on Receipt", "2/10 Net 30"]

TAX_RATE = 0.10


def rand_date(start: datetime, days_range: int) -> datetime:
    return start + timedelta(days=random.randint(0, days_range))


def format_date(dt: datetime) -> str:
    return dt.strftime("%d/%m/%Y")


# ── 1. Purchase Orders ───────────────────────────────────────────────────────

def generate_purchase_orders(n: int = 50) -> list[dict]:
    pos = []
    start = datetime(2024, 1, 15)
    categories = list(CATEGORIES.keys())

    for i in range(1, n + 1):
        cat_name = random.choice(categories)
        cat = CATEGORIES[cat_name]
        vendor = random.choice([v for v in VENDORS if v["id"] in cat["vendor_ids"]])
        item = random.choice(cat["items"])
        qty = random.randint(*cat["qty_range"])
        unit_price = round(random.uniform(*cat["price_range"]), 2)
        subtotal = round(qty * unit_price, 2)
        tax_amount = round(subtotal * TAX_RATE, 2)
        grand_total = round(subtotal + tax_amount, 2)
        po_date = rand_date(start, 300)

        pos.append({
            "po_number":    f"PO-{2024}-{i:04d}",
            "vendor_id":    vendor["id"],
            "vendor_name":  vendor["name"],
            "category":     cat_name,
            "item_description": item,
            "quantity":     qty,
            "unit_price":   unit_price,
            "subtotal":     subtotal,
            "tax_amount":   tax_amount,
            "grand_total":  grand_total,
            "date_issued":  format_date(po_date),
            "payment_terms": random.choice(PAYMENT_TERMS),
            "status":       "Active",
            "_po_date_obj": po_date,  # internal, removed before CSV write
        })
    return pos


# ── 2. Goods Receipts ────────────────────────────────────────────────────────

def generate_goods_receipts(pos: list[dict], pending_count: int = 5) -> list[dict]:
    grs = []
    pending_pos = random.sample([p["po_number"] for p in pos], pending_count)
    employees = ["Ahmad Raza", "Sara Khan", "Bilal Adnan", "Zainab Mir", "Usman Ali"]

    for po in pos:
        if po["po_number"] in pending_pos:
            continue  # No GR for pending POs

        po_date_obj = po["_po_date_obj"]
        recv_date = rand_date(po_date_obj + timedelta(days=3), 14)

        # ~15% have quantity discrepancy
        ordered_qty = po["quantity"]
        if random.random() < 0.15:
            received_qty = max(1, ordered_qty + random.randint(-int(ordered_qty * 0.2), int(ordered_qty * 0.2)))
        else:
            received_qty = ordered_qty

        grs.append({
            "gr_number":          f"GR-{po['po_number'].replace('PO-', '')}",
            "po_number":          po["po_number"],
            "vendor_name":        po["vendor_name"],
            "item_description":   po["item_description"],
            "quantity_ordered":   ordered_qty,
            "quantity_received":  received_qty,
            "date_received":      format_date(recv_date),
            "condition":          random.choice(["Good", "Good", "Good", "Damaged - partial"]),
            "received_by":        random.choice(employees),
        })
    return grs


# ── 3. Contracts ─────────────────────────────────────────────────────────────

def generate_contracts(vendors: list[dict]) -> list[dict]:
    contracts = []
    for v in vendors:
        contracts.append({
            "contract_id":    f"CTR-{v['id']}",
            "vendor_id":      v["id"],
            "vendor_name":    v["name"],
            "agreed_tax_rate": TAX_RATE,
            "payment_terms":   random.choice(PAYMENT_TERMS),
            "contract_start":  "01/01/2024",
            "contract_end":    "31/12/2025",
        })
    return contracts


# ── 4. Invoices with anomaly injection ───────────────────────────────────────

ANOMALY_TYPES = [
    "price_mismatch", "quantity_overbill", "duplicate_submission",
    "vendor_mismatch", "missing_po", "tax_error", "round_amount",
]


def generate_invoices(pos: list[dict], grs: list[dict]) -> list[dict]:
    gr_by_po = {gr["po_number"]: gr for gr in grs}
    invoices = []
    seen_pos = set()
    inv_num = 1

    for po in pos:
        if po["po_number"] not in gr_by_po:
            continue  # Skip POs with no GR

        gr = gr_by_po[po["po_number"]]
        scenario = "correct"
        anomaly_desc = ""
        po_date_obj = po["_po_date_obj"]
        inv_date = rand_date(po_date_obj + timedelta(days=5), 20)

        # 50% anomaly rate
        if random.random() < 0.50:
            scenario = random.choice(ANOMALY_TYPES)

        # Build base invoice values
        qty = po["quantity"]
        unit_price = po["unit_price"]
        vendor_name = po["vendor_name"]
        po_ref = po["po_number"]
        tax_rate = TAX_RATE

        # ── Inject anomaly ────────────────────────────────────────────────
        if scenario == "price_mismatch":
            multiplier = random.uniform(1.05, 1.25)
            unit_price = round(unit_price * multiplier, 2)
            anomaly_desc = f"Unit price inflated by {(multiplier-1)*100:.0f}% above agreed rate"

        elif scenario == "quantity_overbill":
            qty = int(qty * random.uniform(1.1, 1.5))
            anomaly_desc = f"Invoice claims {qty} units; GR shows {gr['quantity_received']} received"

        elif scenario == "duplicate_submission":
            if seen_pos:
                po_ref = random.choice(list(seen_pos))
                anomaly_desc = f"Duplicate: PO reference {po_ref} already invoiced"
            else:
                scenario = "correct"  # Can't duplicate yet

        elif scenario == "vendor_mismatch":
            other_vendors = [v["name"] for v in VENDORS if v["name"] != vendor_name]
            vendor_name = random.choice(other_vendors)
            anomaly_desc = f"Invoice vendor '{vendor_name}' ≠ PO vendor '{po['vendor_name']}'"

        elif scenario == "missing_po":
            po_ref = ""
            anomaly_desc = "No PO reference number on invoice"

        elif scenario == "tax_error":
            tax_rate = random.choice([0.07, 0.15])
            anomaly_desc = f"Tax rate {tax_rate*100:.0f}% instead of expected 10%"

        elif scenario == "round_amount":
            # Make total a round number > $10,000
            subtotal = round(random.randint(10, 50) * 1000 / (1 + tax_rate), 2)
            unit_price = round(subtotal / max(qty, 1), 2)
            anomaly_desc = f"Suspicious round total amount"

        subtotal = round(qty * unit_price, 2)
        tax_amount = round(subtotal * tax_rate, 2)
        grand_total = round(subtotal + tax_amount, 2)
        seen_pos.add(po["po_number"])

        invoices.append({
            "invoice_number":   f"INV-{2024}-{inv_num:04d}",
            "po_number":        po_ref,
            "vendor_name":      vendor_name,
            "item_description": po["item_description"],
            "quantity":         qty,
            "unit_price":       unit_price,
            "subtotal":         subtotal,
            "tax_rate":         tax_rate,
            "tax_amount":       tax_amount,
            "grand_total":      grand_total,
            "invoice_date":     format_date(inv_date),
            "payment_terms":    po["payment_terms"],
            "scenario":         scenario,
            "expected_anomaly": anomaly_desc,
            "_inv_date_obj":    inv_date,
        })
        inv_num += 1

    return invoices


# ── 5. Historical invoices for Isolation Forest ──────────────────────────────

def generate_historical(n: int = 800) -> list[dict]:
    records = []
    start = datetime(2022, 1, 1)

    for _ in range(int(n * 0.92)):  # 92% normal
        total = round(random.uniform(50, 5000), 2)
        qty = random.randint(5, 200)
        unit_price = round(total / qty, 2)
        days_lag = random.randint(3, 45)
        records.append({
            "total_amount": total, "quantity": qty,
            "unit_price": unit_price, "days_between_po_and_invoice": days_lag,
            "is_anomaly": 0,
        })

    for _ in range(int(n * 0.08)):  # 8% anomalies
        anomaly_type = random.choice(["high_amount", "negative_lag", "extreme_qty"])
        if anomaly_type == "high_amount":
            total = round(random.uniform(60000, 200000), 2)
            qty = random.randint(1, 5)
        elif anomaly_type == "extreme_qty":
            total = round(random.uniform(50, 500), 2)
            qty = random.randint(1000, 5000)
        else:
            total = round(random.uniform(100, 2000), 2)
            qty = random.randint(5, 50)
        unit_price = round(total / max(qty, 1), 2)
        days_lag = random.choice([random.randint(-30, -1), random.randint(120, 365)])
        records.append({
            "total_amount": total, "quantity": qty,
            "unit_price": unit_price, "days_between_po_and_invoice": days_lag,
            "is_anomaly": 1,
        })

    random.shuffle(records)
    return records


# ── 6. Invoice PDFs via reportlab ─────────────────────────────────────────────

def generate_invoice_pdf(inv: dict, po: dict, vendor: dict, pdf_path: str) -> None:
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Header
    title_style = ParagraphStyle("title", fontSize=24, fontName="Helvetica-Bold",
                                 alignment=TA_CENTER, textColor=colors.HexColor("#1a3c5e"))
    story.append(Paragraph("INVOICE", title_style))
    story.append(Spacer(1, 0.3*cm))

    # Vendor info + Invoice meta
    info_data = [
        [Paragraph(f"<b>From:</b> {inv['vendor_name']}", styles["Normal"]),
         Paragraph(f"<b>Invoice #:</b> {inv['invoice_number']}", styles["Normal"])],
        [Paragraph(f"{vendor['address']}", styles["Normal"]),
         Paragraph(f"<b>Date:</b> {inv['invoice_date']}", styles["Normal"])],
        [Paragraph(f"{vendor['email']}", styles["Normal"]),
         Paragraph(f"<b>PO Ref:</b> {inv['po_number'] or 'N/A'}", styles["Normal"])],
        ["",
         Paragraph(f"<b>Payment Terms:</b> {inv['payment_terms']}", styles["Normal"])],
    ]
    info_table = Table(info_data, colWidths=[9*cm, 8*cm])
    info_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    # Line items
    header = [["Description", "Qty", "Unit Price", "Amount"]]
    rows = [[
        inv["item_description"],
        str(inv["quantity"]),
        f"${inv['unit_price']:,.2f}",
        f"${inv['subtotal']:,.2f}",
    ]]
    line_table = Table(header + rows, colWidths=[9*cm, 2*cm, 4*cm, 3*cm])
    line_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(line_table)
    story.append(Spacer(1, 0.4*cm))

    # Totals
    tax_pct = int(inv["tax_rate"] * 100)
    totals_data = [
        ["", "Subtotal:", f"${inv['subtotal']:,.2f}"],
        ["", f"Tax ({tax_pct}%):", f"${inv['tax_amount']:,.2f}"],
        ["", "Grand Total:", f"${inv['grand_total']:,.2f}"],
    ]
    totals_table = Table(totals_data, colWidths=[9*cm, 4*cm, 4*cm])
    totals_table.setStyle(TableStyle([
        ("ALIGN",     (1, 0), (-1, -1), "RIGHT"),
        ("FONTNAME",  (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 2), (-1, 2), 11),
        ("LINEABOVE", (0, 2), (-1, 2), 1, colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(totals_table)
    story.append(Spacer(1, 1*cm))

    # Footer
    footer_style = ParagraphStyle("footer", fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph(
        f"Please remit payment per terms: {inv['payment_terms']} | "
        f"Make payable to: {inv['vendor_name']}",
        footer_style
    ))

    doc.build(story)


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_csv(path: str, rows: list[dict], exclude_keys: list[str] | None = None) -> None:
    if not rows:
        return
    exclude_keys = exclude_keys or []
    fieldnames = [k for k in rows[0].keys() if k not in exclude_keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"  ✓ {path} ({len(rows)} records)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic dataset...\n")

    # POs
    pos = generate_purchase_orders(50)
    write_csv(str(DATA / "purchase_orders.csv"), pos, exclude_keys=["_po_date_obj"])

    # GRs
    grs = generate_goods_receipts(pos, pending_count=5)
    write_csv(str(DATA / "goods_receipts.csv"), grs)

    # Contracts
    contracts = generate_contracts(VENDORS)
    write_csv(str(DATA / "contracts.csv"), contracts)

    # Invoices
    invoices = generate_invoices(pos, grs)
    write_csv(str(DATA / "invoices.csv"), invoices, exclude_keys=["_inv_date_obj"])
    print(f"  Anomalous: {sum(1 for i in invoices if i['scenario'] != 'correct')} / {len(invoices)}")

    # Historical
    historical = generate_historical(800)
    write_csv(str(DATA / "historical_invoices.csv"), historical)

    # PDFs — generate for first 30 invoices
    vendor_map = {v["id"]: v for v in VENDORS}
    po_map = {p["po_number"]: p for p in pos}
    print(f"\nGenerating PDF invoices in {PDF_DIR}/ ...")

    pdf_invoices = [i for i in invoices if i["scenario"] != "missing_po"][:30]
    for inv in pdf_invoices:
        po = po_map.get(inv.get("po_number", ""), pos[0])  # fallback
        vendor_id = po.get("vendor_id", "V001")
        vendor = vendor_map.get(vendor_id, VENDORS[0])
        pdf_path = str(PDF_DIR / f"{inv['invoice_number']}.pdf")
        generate_invoice_pdf(inv, po, vendor, pdf_path)

    print(f"  ✓ {len(pdf_invoices)} PDFs generated in data/sample_invoices/")
    print("\n✅ Data generation complete!")
    print(f"   POs: {len(pos)}, GRs: {len(grs)}, Invoices: {len(invoices)}, Historical: {len(historical)}")


if __name__ == "__main__":
    main()
