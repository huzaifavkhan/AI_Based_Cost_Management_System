"""
generate_receipt_images.py
--------------------------
Generates synthetic thermal receipt images + PDF versions + JSON annotations
for LayoutLMv3 fine-tuning.

Layout variation is the core design goal.  Every receipt is different:
  - Date may appear at the TOP (after store header) OR at the BOTTOM (near TC#)
  - Subtotal may appear mid-receipt (Walmart sectional) or only at the bottom
  - Some receipts have an ID# line at the very top before the store name
  - Payment section varies: cash / card / card+approval block / EFT block
  - Header taglines, manager lines, store numbers randomly included/excluded
  - Item count: 1-8 items per receipt
  - Some receipts have two item sections (Walmart style)

Stores: WAL*MART, Walmart, Trader Joe's, Whole Foods, TARGET, CVS,
        SPAR, KROGER, ALDI, DOLLAR GENERAL

Output:
  data/synthetic_receipts/img/       <- JPEG images  SYNTH_0001.jpg ...
  data/synthetic_receipts/pdf/       <- PDF receipts  SYNTH_0001.pdf ...
  data/synthetic_receipts/entities/  <- JSON annotations SYNTH_0001.txt ...

Usage:
  pip install Pillow faker reportlab
  python scripts/generate_receipt_images.py --n 500 --seed 42
  python scripts/generate_receipt_images.py --n 10 --pdf-only   # PDFs only
"""

import argparse, json, random
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "data" / "synthetic_receipts"
IMG_DIR = OUT_DIR / "img"
PDF_DIR = OUT_DIR / "pdf"
ENT_DIR = OUT_DIR / "entities"
IMG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
ENT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fonts ─────────────────────────────────────────────────────────────────────
WIDTH  = 420
LINE_H = 18
PAD    = 14
FSZ    = 14

def _font(sz=FSZ):
    for name in ["cour.ttf", "Courier_New.ttf", "DejaVuSansMono.ttf",
                 "LiberationMono-Regular.ttf", "FreeMono.ttf"]:
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            pass
    return ImageFont.load_default()

F      = _font(FSZ)
F_BIG  = _font(FSZ + 3)
F_SM   = _font(FSZ - 2)

fake = Faker()

# ── Store catalogue ───────────────────────────────────────────────────────────
STORES = [
    {
        "name": "WAL*MART",
        "taglines": ["Save money. Live better.", "ALWAYS LOW PRICES.",
                     "Low Prices You Can Trust. Every Day."],
        "id_key": "tc_number",
        "cities": ["CHESTERFIELD, MO 63005", "DALLAS, TX 75201",
                   "FLORENCE, SC 29505", "WESTMINSTER, CA 92683",
                   "PHOENIX, AZ 85001", "MARYDVILLE, CA 95901"],
        "sym": "$",
    },
    {
        "name": "Walmart",
        "taglines": ["Save money. Live better.", "Low Prices Every Day"],
        "id_key": "tc_number",
        "cities": ["ORLANDO, FL 32801", "SEATTLE, WA 98101",
                   "BIRMINGHAM, AL 35201", "MEMPHIS, TN 38101"],
        "sym": "$",
    },
    {
        "name": "Trader Joe's",
        "taglines": ["Thank you for shopping Trader Joe's", ""],
        "id_key": "reference",
        "cities": ["Dallas TX 75206", "Boston MA 02101",
                   "Denver CO 80201", "Portland OR 97201"],
        "sym": "$",
    },
    {
        "name": "Whole Foods Market",
        "taglines": ["", "365 Every Day Value"],
        "id_key": "reference",
        "cities": ["Sharon Rd.", "Austin TX 78701",
                   "Chicago IL 60601", "Nashville TN 37201"],
        "sym": "$",
    },
    {
        "name": "TARGET",
        "taglines": ["Expect More. Pay Less.", ""],
        "id_key": "reference",
        "cities": ["Minneapolis MN 55401", "Atlanta GA 30301",
                   "Portland OR 97201", "San Jose CA 95101"],
        "sym": "$",
    },
    {
        "name": "CVS/pharmacy",
        "taglines": ["CVS ExtraCare Member", "Health is Everything"],
        "id_key": "receipt_number",
        "cities": ["Woonsocket RI 02895", "Boston MA 02101",
                   "New York NY 10001", "Chicago IL 60601"],
        "sym": "$",
    },
    {
        "name": "SPAR",
        "taglines": ["Together we are more", ""],
        "id_key": "slip_number",
        "cities": ["Johannesburg, SA", "London, UK",
                   "Frankfurt, DE", "Cape Town, SA"],
        "sym": None,   # randomised per receipt
    },
    {
        "name": "KROGER",
        "taglines": ["Fresh for Everyone", ""],
        "id_key": "tc_number",
        "cities": ["Cincinnati OH 45202", "Nashville TN 37201",
                   "Memphis TN 38101", "Columbus OH 43201"],
        "sym": "$",
    },
    {
        "name": "ALDI",
        "taglines": ["Good Food. Everyday.", ""],
        "id_key": "slip_number",
        "cities": ["Batavia IL 60510", "Chicago IL 60601",
                   "Columbus OH 43201", "Indianapolis IN 46201"],
        "sym": "$",
    },
    {
        "name": "DOLLAR GENERAL",
        "taglines": ["Serving Others", ""],
        "id_key": "tc_number",
        "cities": ["Goodlettsville TN 37072", "Atlanta GA 30301",
                   "Nashville TN 37201"],
        "sym": "$",
    },
    {
        "name": "THE GRILLE",
        "taglines": ["Thank You For Dining With Us!", "Enjoy Your Meal"],
        "id_key": "bill_number",
        "type": "restaurant",
        "cities": ["123 Main St, Austin TX 78701", "456 Oak Ave, Chicago IL 60601",
                   "789 Elm Rd, New York NY 10001", "321 Pine Blvd, Seattle WA 98101"],
        "sym": "$",
    },
    {
        "name": "FUEL STOP",
        "taglines": ["Drive Safe!", "Thanks for stopping by"],
        "id_key": "receipt_number",
        "type": "gas",
        "cities": ["I-10 Exit 42, Phoenix AZ 85001", "Hwy 66, Dallas TX 75201",
                   "Exit 15 I-95, Atlanta GA 30301", "Hwy 101, Los Angeles CA 90001"],
        "sym": "$",
    },
    {
        "name": "COFFEE BEAN",
        "taglines": ["Brewed with love", "Thank you! See you soon."],
        "id_key": "order_number",
        "type": "cafe",
        "cities": ["234 Brew St, Portland OR 97201", "88 Java Ave, Denver CO 80201",
                   "12 Roast Rd, Austin TX 78701", "99 Latte Ln, Boston MA 02101"],
        "sym": "$",
    },
]

FOOD = [
    "BANANAS", "WHOLE MILK 1GAL", "BREAD WHEAT", "EGGS 12CT", "BUTTER UNSALT",
    "APPLE BAG 3LB", "CHICKEN BREAST", "GROUND BEEF 1LB", "PASTA PENNE",
    "TOMATO SAUCE", "ORANGE JUICE", "YOGURT VANILLA", "CHEESE SLICED",
    "FROZEN PIZZA", "POTATO CHIPS", "COFFEE GROUND", "CEREAL FLAKES",
    "BABY SPINACH", "STRAWBERRIES", "AVOCADO HASS", "SALSA MEDIUM",
    "TORTILLAS 10CT", "BLACK BEANS CAN", "SOUP CHKN NOD", "CRACKERS RITZ",
]

GENERAL = [
    "SHAMPOO", "TOOTHPASTE", "BODY WASH", "LAUNDRY DET", "PAPER TOWELS 6PK",
    "TOILET PAPER", "DISH SOAP", "FACE WASH", "DEODORANT", "RAZORS 4CT",
    "HAND SOAP", "VITAMINS C", "IBUPROFEN", "BANDAGES BOX", "COTTON SWABS",
    "NOTEBOOK", "PENS 10CT", "BATTERIES AA", "LIGHT BULB 2PK", "SW FIGURES",
    "WOMEN SLIPPE", "GIFT CARD", "USB CABLE", "PHONE CASE", "HEADPHONES",
]

# ── Text helpers ──────────────────────────────────────────────────────────────
W = 40

def _sep(c="-"):        return c * W
def _cen(t):            return t.center(W)
def _rgt(left, right):
    sp = W - len(left) - len(right)
    return left + " " * max(sp, 1) + right
def _p(v, sym="$"):     return f"{sym}{v:.2f}"

def _mk_tc(rng):
    parts = [str(rng.randint(1000, 9999)) for _ in range(4)]
    return " ".join(parts), "".join(parts)

def _mk_slip(rng):  return str(rng.randint(10000, 99999))
def _mk_ref(rng):   return str(rng.randint(100000000000, 999999999999))
def _mk_order(rng): return str(rng.randint(100, 9999))

# ── Date format variation ─────────────────────────────────────────────────────
_DATE_FORMATS = [
    "%m/%d/%Y",      # 03/21/2016  (classic US)
    "%m/%d/%y",      # 03/21/16
    "%d/%m/%Y",      # 21/03/2016
    "%Y-%m-%d",      # 2016-03-21
    "%d-%m-%Y",      # 21-03-2016
    "%d.%m.%Y",      # 21.03.2016
    "%d %B %Y",      # 21 March 2016
    "%d %b %Y",      # 21 Mar 2016
    "%d-%b-%Y",      # 21-Mar-2016
    "%d-%B-%Y",      # 21-March-2016
    "%B %d, %Y",     # March 21, 2016
    "%b %d, %Y",     # Mar 21, 2016
    "%d %b %y",      # 21 Mar 16
]

_DATE_LABELS = ["Date", "DATE", "Transaction Date", "Billing Date", "Issued On",
                "TRANS DATE", "TXN DATE", "Date Purchased", "Date Paid"]

def _format_date(dt, rng):
    """Format a datetime using a randomly chosen date format."""
    return dt.strftime(rng.choice(_DATE_FORMATS))

# ── Field label variations ────────────────────────────────────────────────────
_TOTAL_LABELS    = ["TOTAL", "Total", "GRAND TOTAL", "AMOUNT DUE",
                    "BALANCE DUE", "TTL", "AMOUNT PAYABLE", "NET AMOUNT"]
_SUBTOTAL_LABELS = ["SUBTOTAL", "Sub Total", "SUB-TOTAL", "PRE-TAX", "NET"]

# ── OCR noise ─────────────────────────────────────────────────────────────────
_CONFUSABLES = {
    '0': ['O', 'Q'], '1': ['l', 'I'], '5': ['S'], '8': ['B'],
    'O': ['0', 'Q'], 'I': ['1', 'l'], 'S': ['5'], 'B': ['8'],
    'l': ['1', 'I'], 'Z': ['2'],
}

def _add_ocr_noise(text: str, rng: random.Random, rate: float = 0.015) -> str:
    """Simulate OCR misread characters at the given error rate."""
    result = []
    for ch in text:
        if rng.random() < rate and ch in _CONFUSABLES:
            result.append(rng.choice(_CONFUSABLES[ch]))
        else:
            result.append(ch)
    return "".join(result)

# ── Restaurant food items ─────────────────────────────────────────────────────
RESTAURANT_ITEMS = [
    "BURGER DELUXE", "FRIES LARGE", "SODA 32OZ", "CHICKEN WRAP",
    "CAESAR SALAD", "ONION RINGS", "MILKSHAKE", "FISH TACOS",
    "GRILLED CHEESE", "SOUP OF DAY", "LEMONADE", "ICE CREAM",
    "STEAK PLATE", "PASTA SPECIAL", "GARLIC BREAD", "CHEESECAKE",
    "HOT WINGS", "NACHOS", "CORN CHOWDER", "APPLE PIE",
]

CAFE_ITEMS = [
    "LATTE", "CAPPUCCINO", "ESPRESSO", "AMERICANO", "FLAT WHITE",
    "CROISSANT", "MUFFIN BLUEBERRY", "BAGEL + CC", "ICED COFFEE",
    "COLD BREW", "GREEN TEA", "CHAI LATTE", "SCONE", "BROWNIE",
]

GAS_FUEL_GRADES = ["REGULAR", "PLUS", "PREMIUM", "DIESEL"]

# ── Receipt builder ───────────────────────────────────────────────────────────

def build_receipt(rng: random.Random) -> tuple[list, dict]:
    """
    Returns (lines, annotation).
    lines = list of (text_str, font_obj)
    """
    store   = rng.choice(STORES)
    sname   = store["name"]
    sym     = store["sym"] or rng.choice(["$", "$", "$", "R ", "EUR ", "GBP "])
    city    = rng.choice(store["cities"])
    tagline = rng.choice(store["taglines"])
    dt         = datetime.now() - timedelta(days=rng.randint(0, 900))
    id_key     = store["id_key"]
    store_type = store.get("type", "retail")
    if store_type == "restaurant":
        pool = RESTAURANT_ITEMS
    elif store_type == "cafe":
        pool = CAFE_ITEMS
    elif store_type == "gas":
        pool = FOOD + GENERAL   # convenience items
    else:
        pool = FOOD if sname in ("Trader Joe's", "Whole Foods Market",
                                 "KROGER", "ALDI", "SPAR") else FOOD + GENERAL

    # Transaction ID
    if id_key == "tc_number":
        tc_disp, tc_raw = _mk_tc(rng)
        ann_id = {"tc_number": tc_raw}
        tc_label = "TC#"
    elif id_key == "slip_number":
        tc_raw = _mk_slip(rng); tc_disp = tc_raw
        ann_id = {"slip_number": tc_raw}
        tc_label = "SLIP#"
    elif id_key == "reference":
        tc_raw = _mk_ref(rng); tc_disp = tc_raw
        ann_id = {"reference": tc_raw}
        tc_label = "REF#"
    else:
        tc_raw = _mk_order(rng); tc_disp = tc_raw
        ann_id = {"order_number": tc_raw}
        tc_label = "ORDER#"

    # Items - sectional (Walmart) or flat
    use_sections = (sname in ("WAL*MART", "Walmart")) and rng.random() > 0.4
    if use_sections:
        sec_a = [(rng.choice(pool), round(rng.uniform(0.5, 25), 2))
                 for _ in range(rng.randint(1, 3))]
        sec_b = [(rng.choice(pool), round(rng.uniform(0.5, 25), 2))
                 for _ in range(rng.randint(1, 3))]
        all_items = sec_a + sec_b
        sub_a = round(sum(p for _, p in sec_a), 2)
        sub_b = round(sum(p for _, p in sec_b), 2)
    else:
        all_items = [(rng.choice(pool), round(rng.uniform(0.5, 40), 2))
                     for _ in range(rng.randint(1, 8))]
        sec_a = sec_b = None

    subtotal = round(sum(p for _, p in all_items), 2)

    # Tax
    tax_style = rng.choice(["single", "split", "none"])
    if tax_style == "single":
        tax_r = rng.choice([0.05, 0.065, 0.07, 0.08, 0.09, 0.10])
        tax   = round(subtotal * tax_r, 2)
        total = round(subtotal + tax, 2)
        t1_r = t2_r = tax1 = tax2 = None
    elif tax_style == "split":
        t1_r  = rng.choice([0.031, 0.05, 0.06])
        t2_r  = rng.choice([0.04, 0.0435, 0.08])
        tax1  = round(subtotal * t1_r, 2)
        tax2  = round(subtotal * t2_r, 2)
        tax   = round(tax1 + tax2, 2)
        total = round(subtotal + tax, 2)
        tax_r = None
    else:
        # Add a small service fee so subtotal ≠ total (avoids model confusion)
        svc_fee = round(rng.uniform(0.01, 0.99), 2)
        tax     = svc_fee
        total   = round(subtotal + svc_fee, 2)
        t1_r = t2_r = tax1 = tax2 = tax_r = None

    # TC placement: Walmart/tc_number stores vary; others always footer
    if id_key == "tc_number":
        tc_placement = rng.choices(["header", "inline", "footer"],
                                   weights=[40, 20, 40])[0]
    else:
        tc_placement = "footer"

    # Layout flags - randomly pick per receipt
    date_at_top   = rng.random() > 0.45
    has_id_hdr    = rng.random() > 0.55
    has_manager   = rng.random() > 0.50
    has_phone     = rng.random() > 0.40
    has_st_line   = rng.random() > 0.35
    show_n_items  = rng.random() > 0.40
    has_barcode   = rng.random() > 0.20
    has_promo_ftr = rng.random() > 0.50
    pay_style     = rng.choice(["cash", "card", "card_approval", "eft"])
    show_sku      = rng.random() > 0.50

    lines = []
    def L(t, f=F):     lines.append((t, f))
    def B(t):          lines.append((t, F_BIG))
    def S(t):          lines.append((t, F_SM))

    # Very top ID line
    if has_id_hdr:
        fid = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=10))
        S(f"ID #: {fid}")
        L("")

    # Store header
    B(_cen(sname))
    if tagline:
        L(_cen(tagline))
    if has_phone:
        ph = f"({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"
        L(_cen(ph))
    if has_manager:
        mgr = fake.first_name().upper() + " " + fake.last_name().upper()
        L(_cen(f"MANAGER {mgr}"))
    L(_cen(city))
    if has_st_line:
        st = rng.randint(100, 9999)
        op = str(rng.randint(10000000, 99999999)).zfill(8)
        te = rng.randint(10, 99)
        tr = str(rng.randint(1000, 99999)).zfill(5)
        S(f"ST# {st} OP# {op} TE# {te} TR# {tr}")

    # TC in header position
    if tc_placement == "header":
        S(f"{tc_label} {tc_disp}")

    # Date at top
    if date_at_top:
        date_label = rng.choice(_DATE_LABELS)
        L(f"{date_label}: {_format_date(dt, rng)}   {dt.strftime('%H:%M:%S')}")

    L(_sep())

    # Store-type specific header lines
    if store_type == "restaurant":
        L(_cen(f"TABLE {rng.randint(1, 30)}  SERVER {rng.randint(1, 20)}"))
        L(_cen(f"GUESTS: {rng.randint(1, 6)}"))
        L("")
    elif store_type == "gas":
        pump_no = rng.randint(1, 12)
        grade   = rng.choice(GAS_FUEL_GRADES)
        gallons = round(rng.uniform(3.0, 20.0), 3)
        ppg     = round(rng.uniform(2.80, 5.50), 3)
        fuel_total = round(gallons * ppg, 2)
        S(f"PUMP #{pump_no}  {grade}")
        L(_rgt(f"{gallons:.3f} GAL @ {sym}{ppg:.3f}/GAL", _p(fuel_total, sym)))
        L("")
    elif store_type == "cafe":
        L(_cen(f"ORDER #{rng.randint(100, 999)}"))
        L("")

    # Items
    if use_sections:
        for iname, price in sec_a:
            sku = str(rng.randint(10000000000, 99999999999))
            L(_rgt(f"{iname[:12]:<12} {sku}", _p(price, sym)))
        L(_rgt("   SUBTOTAL", _p(sub_a, sym)))
        for iname, price in sec_b:
            sku = str(rng.randint(10000000000, 99999999999))
            L(_rgt(f"{iname[:12]:<12} {sku}", _p(price, sym)))
        L(_rgt("   SUBTOTAL", _p(sub_b, sym)))
    else:
        for iname, price in all_items:
            if show_sku:
                sku = str(rng.randint(1000000000, 9999999999))
                L(_rgt(f"{iname[:12]:<12} {sku}", _p(price, sym)))
            else:
                L(_rgt(iname[:22], _p(price, sym)))

    L(_sep())

    # TC in inline position (after items, before subtotal)
    if tc_placement == "inline":
        S(f"{tc_label} {tc_disp}")
        L("")

    # Subtotal / tax / total
    if not use_sections:
        subtotal_lbl = rng.choice(_SUBTOTAL_LABELS)
        L(_rgt(subtotal_lbl, _p(subtotal, sym)))

    show_tax_rate = rng.random() > 0.4  # 60% show rate; 40% just dollar amount
    if tax_style == "single":
        tax_lbl = f"TAX ({tax_r*100:.1f}%)" if show_tax_rate else "TAX"
        L(_rgt(tax_lbl, _p(tax, sym)))
    elif tax_style == "split":
        lbl1 = f"TAX 1  {t1_r*100:.3f} %" if show_tax_rate else "TAX 1"
        lbl2 = f"TAX 3  {t2_r*100:.3f} %" if show_tax_rate else "TAX 2"
        L(_rgt(lbl1, _p(tax1, sym)))
        L(_rgt(lbl2, _p(tax2, sym)))
    else:
        # "none" case: show service fee line
        L(_rgt("SERVICE FEE", _p(tax, sym)))

    L(_sep("="))
    total_lbl = rng.choice(_TOTAL_LABELS)
    B(_rgt(f"  {total_lbl}", _p(total, sym)))

    # Cafe: add tip line after total (tip is not part of annotated total)
    if store_type in ("restaurant", "cafe"):
        tip_pct = rng.choice([0.15, 0.18, 0.20, 0.22, 0.25])
        tip_amt = round(total * tip_pct, 2)
        L("")
        L(_rgt(f"TIP ({tip_pct*100:.0f}%)", _p(tip_amt, sym)))
        L(_rgt("SUGGESTED TIP", _p(tip_amt, sym)))
        L("")

    # Payment section
    if pay_style == "cash":
        cash   = total + rng.choice([0, 0.25, 0.50, 1.0, 2.0, 5.0])
        change = round(cash - total, 2)
        L(_rgt("CASH TEND", _p(cash, sym)))
        L(_rgt("CHANGE DUE", _p(change, sym)))

    elif pay_style == "card":
        ct = rng.choice(["VISA", "MASTERCARD", "DEBIT", "MCARD"])
        L(_rgt(f"{ct} TEND", _p(total, sym)))
        L(_rgt("CHANGE DUE", "0.00"))

    elif pay_style == "card_approval":
        ct    = rng.choice(["VISA", "MASTERCARD", "DEBIT", "AMEX"])
        last4 = rng.randint(1000, 9999)
        appr  = str(rng.randint(100000, 999999))
        L(_rgt(f"{ct} *{last4}", _p(total, sym)))
        L("")
        S(f"ACCOUNT #{rng.randint(1000,9999)}")
        S(f"APPROVAL #{appr}")
        S(f"TRANS ID - {rng.randint(100000000, 999999999)}")
        S("VALIDATION -")
        S("PAYMENT SERVICE - A")
        L(_rgt("CHANGE DUE", "0.00"))

    elif pay_style == "eft":
        ct = rng.choice(["VISA", "DEBIT", "EFT DEBIT"])
        L(_rgt(f"{ct} TEND", _p(total, sym)))
        L("")
        S(f"EFT {ct}")
        S(f"TOTAL PURCHASES  {_p(total, sym)}")
        S(f"REF # {rng.randint(1000000000, 9999999999)}")
        S(f"NETWORK ID: {rng.randint(100000, 999999)}")
        S(f"RESP CODE:  {rng.randint(100000, 999999)}")

    L("")

    # Items count
    if show_n_items:
        B(f"# ITEMS SOLD {len(all_items)}")

    # Transaction ID (footer position)
    if tc_placement == "footer":
        L(f"{tc_label} {tc_disp}")

    # Date at bottom
    if not date_at_top:
        date_label = rng.choice(_DATE_LABELS)
        L(f"{date_label}: {_format_date(dt, rng)}   {dt.strftime('%H:%M:%S')}")

    # Promo footer
    if has_promo_ftr and tagline:
        L("")
        S(_cen(tagline))

    # ASCII barcode
    if has_barcode:
        L("")
        bc = ""
        for _ in range(38):
            bc += rng.choice(["I", "l", "1", "|", " ", " "])
        L(bc)

    L("")
    L(_cen("***CUSTOMER COPY***"))

    annotation = {
        "store_name": sname,
        "date":       dt.strftime("%m/%d/%Y"),
        "total":      f"{total:.2f}",
        "subtotal":   f"{subtotal:.2f}",
        **ann_id,
    }
    return lines, annotation


# ── JPEG renderer ────────────────────────────────────────────────────────────

def render_jpg(lines: list, rng: random.Random) -> Image.Image:
    height = LINE_H * (len(lines) + 4) + PAD * 2
    bg     = rng.randint(245, 255)
    img    = Image.new("L", (WIDTH, height), color=bg)
    draw   = ImageDraw.Draw(img)

    y = PAD
    for text, font in lines:
        ink = rng.randint(15, 50)
        # Apply OCR-like character noise 30% of the time, never on amount lines
        display_text = text
        if rng.random() > 0.70 and "$" not in text and "." not in text[-5:]:
            display_text = _add_ocr_noise(text, rng)
        draw.text((PAD, y), display_text, fill=ink, font=font)
        y += LINE_H + (4 if font is F_BIG else 0)

    # Salt-and-pepper noise
    pixels  = img.load()
    w, h    = img.size
    n_noise = int(w * h * rng.uniform(0.003, 0.025))
    for _ in range(n_noise):
        px = rng.randint(0, w - 1)
        py = rng.randint(0, h - 1)
        pixels[px, py] = rng.choice([0, 255, bg - 10])

    # Occasional vertical crease
    if rng.random() > 0.70:
        sx = rng.randint(PAD, WIDTH - PAD)
        for py in range(0, h, rng.randint(1, 3)):
            if 0 <= sx < w and 0 <= py < h:
                pixels[sx, py] = min(pixels[sx, py] + rng.randint(20, 60), 255)

    return img


# ── PDF renderer ──────────────────────────────────────────────────────────────

def render_pdf(lines: list, out_path: Path):
    """
    Render receipt lines to a narrow thermal-style PDF using reportlab.
    Receipt width: 80mm (standard thermal roll).
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    # Thermal receipt dimensions: 85 mm wide (extra 5mm prevents bold text clipping)
    PAGE_W    = 85 * mm
    LINE_PH   = 5.2 * mm        # line height in PDF points
    LINE_BIG  = 6.5 * mm
    MARGIN    = 3 * mm
    FONT_N    = "Courier"
    FONT_B    = "Courier-Bold"
    SZ_NORM   = 7.5
    SZ_BIG    = 9.5
    SZ_SM     = 6.5

    # Calculate total height needed
    total_h = MARGIN * 2
    for _, font in lines:
        total_h += LINE_BIG if font is F_BIG else LINE_PH

    PAGE_H = max(total_h + MARGIN * 2, 50 * mm)

    c = canvas.Canvas(str(out_path), pagesize=(PAGE_W, PAGE_H))
    c.setFillColorRGB(0, 0, 0)

    # Draw white background
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)

    # Draw lines from top to bottom (PDF coords: 0 = bottom)
    y = PAGE_H - MARGIN

    for text, font in lines:
        if font is F_BIG:
            c.setFont(FONT_B, SZ_BIG)
            lh = LINE_BIG
        elif font is F_SM:
            c.setFont(FONT_N, SZ_SM)
            lh = LINE_PH - 0.8 * mm
        else:
            c.setFont(FONT_N, SZ_NORM)
            lh = LINE_PH

        y -= lh
        # Clip text to page width
        max_chars = int((PAGE_W - MARGIN * 2) / (SZ_NORM * 0.6))
        c.drawString(MARGIN, y, text[:max_chars])

    c.save()


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(n: int = 100, seed: int = 42, pdf_only: bool = False, start: int = 1):
    rng = random.Random(seed)
    fake.seed_instance(seed)

    for i in range(start, start + n):
        lines, annotation = build_receipt(rng)
        stem = f"SYNTH_{i:04d}"

        # JPEG
        if not pdf_only:
            img = render_jpg(lines, rng)
            img.save(str(IMG_DIR / f"{stem}.jpg"), "JPEG", quality=93)

        # PDF
        render_pdf(lines, PDF_DIR / f"{stem}.pdf")

        # Annotation JSON
        (ENT_DIR / f"{stem}.txt").write_text(
            json.dumps(annotation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        done = i - start + 1
        if done % 50 == 0 or done == n:
            print(f"  {done}/{n} receipts generated  (last: {stem})")

    print(f"\nDone -> {OUT_DIR}")
    if not pdf_only:
        print(f"  JPEGs    -> {IMG_DIR}")
    print(f"  PDFs     -> {PDF_DIR}")
    print(f"  Entities -> {ENT_DIR}")
    print("\nAdd JPEGs to training pipeline:")
    print(f"  copy {IMG_DIR}\\* <train_dir>\\img\\")
    print(f"  copy {ENT_DIR}\\* <train_dir>\\entities\\")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n",        type=int,  default=100, help="Number of receipts (default 100)")
    p.add_argument("--seed",     type=int,  default=42,  help="Random seed (default 42)")
    p.add_argument("--start",    type=int,  default=1,   help="Starting file index (default 1)")
    p.add_argument("--pdf-only", action="store_true",    help="Generate PDFs only, skip JPEGs")
    args = p.parse_args()
    generate(n=args.n, seed=args.seed, pdf_only=args.pdf_only, start=args.start)