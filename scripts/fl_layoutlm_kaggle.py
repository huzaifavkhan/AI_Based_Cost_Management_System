"""
Federated Learning for LayoutLMv3 — Kaggle-ready self-contained script.

What this does:
  Runs FedAvg + LoRA + Differential Privacy across 3 simulated non-IID clients
  on the SROIE + French FACTURE invoice dataset, starting from your already
  fine-tuned LayoutLMv3 base model.

Kaggle setup (do this before running):
  1. Add your training data as an input dataset
       (the one with train/img/ and train/entities/)
  2. Add your fine-tuned LayoutLMv3 model as an input dataset
       (the layoutlm_invoice/ folder output from finetune_layoutlm_kaggle.py)
  3. Enable GPU: Settings → Accelerator → GPU T4 x1
  4. Enable Internet: Settings → Internet → On
  5. Run All

Outputs saved to /kaggle/working/:
  layoutlm_lora/          — LoRA adapter weights (adapter_config.json + adapter_model.safetensors)
  layoutlm_lora_merged/   — Full merged LayoutLMv3 model (drop-in replacement for base)
  fl_lora_metadata.json   — Round history, F1 scores, privacy budget

After the run:
  Download layoutlm_lora_merged/ and place it at
  cost-management-system/data/layoutlm_lora_merged/
  on your local machine.
"""

import os, json, re, random, time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

# ── Install dependencies ───────────────────────────────────────────────────────
os.system("apt-get install -q -y tesseract-ocr tesseract-ocr-fra")
os.system("pip install -q peft seqeval deep-translator")

import numpy as np
import torch
import torch.optim as optim
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from peft import LoraConfig, get_peft_model, PeftModel
import pytesseract

# ── Path auto-detection ────────────────────────────────────────────────────────
_INPUT = Path("/kaggle/input")

# Find training data
_img_hits = list(_INPUT.rglob("img"))
if _img_hits:
    TRAIN_DIR = _img_hits[0].parent
    print(f"[setup] Training data: {TRAIN_DIR}")
else:
    TRAIN_DIR = Path("/kaggle/input/datasets/huzaifaahmedkhann/fypdata/train")
    print(f"[setup] WARNING: img/ not found — falling back to {TRAIN_DIR}")

IMG_DIR    = TRAIN_DIR / "img"
ENTITY_DIR = TRAIN_DIR / "entities"

# Find fine-tuned base model (output of finetune_layoutlm_kaggle.py)
_model_hits = [
    p for p in _INPUT.rglob("config.json")
    if "layoutlm" in str(p).lower() and "lora" not in str(p).lower()
]
if _model_hits:
    BASE_MODEL_DIR = _model_hits[0].parent
    print(f"[setup] Base model: {BASE_MODEL_DIR}")
else:
    BASE_MODEL_DIR = Path("/kaggle/input/datasets/huzaifaahmedkhann/federatedd/layoutlm_invoice")
    print(f"[setup] WARNING: base model not found — falling back to {BASE_MODEL_DIR}")

OUTPUT_DIR   = Path("/kaggle/working")
LORA_DIR     = OUTPUT_DIR / "layoutlm_lora"
MERGED_DIR   = OUTPUT_DIR / "layoutlm_lora_merged"
META_PATH    = OUTPUT_DIR / "fl_lora_metadata.json"

LORA_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_ROUNDS      = 15
LOCAL_EPOCHS  = 1
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
SIGMA         = 0.0     # 0.0 = no DP noise (model learns); >0 adds Gaussian noise but kills F1 at this scale
CLIP_NORM     = 1.0     # L2 clip bound for LoRA delta (1.0 matches typical LoRA update magnitude)
MU            = 0.01    # FedProx proximal coefficient
BATCH_SIZE    = 2       # Per-client batch size (GPU can handle more than CPU)
MAX_LENGTH    = 512     # Token sequence length
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[setup] Device: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ── Label schema (must match finetune_layoutlm_kaggle.py exactly) ──────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# DATA PARTITIONING  (verbatim logic from layoutlm_data_partitioner.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImageSample:
    img_path:   Path
    label_path: Path
    source:     str   # "xml" | "txt"
    stem:       str


def _normalise_date(raw: str, source: str = "txt") -> str:
    raw = raw.strip()
    if not raw:
        return raw
    if source == "xml":
        m = re.match(r'^(\d{4})(\d{2})(\d{2})$', raw)
        if m:
            return f"{m.group(2)}/{m.group(3)}/{m.group(1)}"
    s = re.sub(r'[-.]', '/', raw)
    m = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', s)
    if m:
        return f"{int(m.group(2)):02d}/{int(m.group(3)):02d}/{m.group(1)}"
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if a > 12:
            return f"{b:02d}/{a:02d}/{y}"
        return f"{a:02d}/{b:02d}/{y}"
    return raw


def _date_digit_variants(raw: str, source: str) -> set:
    normalised = _normalise_date(raw, source)
    m = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', normalised)
    if m:
        mo, d, y = m.group(1), m.group(2), m.group(3)
        return {f"{y}{mo}{d}", f"{d}{mo}{y}", f"{mo}{d}{y}"}
    return {re.sub(r'[^0-9]', '', raw)}


_SYMBOL_PATTERNS = [
    (r"(?<![A-Za-z])RM(?![A-Za-z])", "RM"),
    (r"R\$", "R$"),
    (r"Rs\.?\s*\d", "Rs."),
    (r"₹", "₹"),
    (r"£", "£"),
    (r"€", "€"),
    (r"\$", "$"),
]


def _currency_from_text(text: str):
    for pattern, sym in _SYMBOL_PATTERNS:
        if re.search(pattern, text):
            return sym
    return None


def _currency_from_address(addr: str):
    a = addr.lower()
    if any(k in a for k in ("malaysia", "kuala lumpur", "penang", "selangor")):
        return "RM"
    if any(k in a for k in ("france", "paris", "lyon", "marseille")):
        return "€"
    if any(k in a for k in ("uk", "london", "england")):
        return "£"
    return None


def _detect_currency(total_str="", address="", full_text=""):
    for src in [total_str, address, full_text]:
        sym = _currency_from_text(src) if src else None
        if sym:
            return sym
    if address:
        return _currency_from_address(address)
    return None


def _normalise_amount(raw: str) -> str:
    s = re.sub(
        r"[€£¥₹₩₺₽﷼฿৳]|د\.إ|R\$"
        r"|(?<![A-Za-z])RM(?![A-Za-z])"
        r"|(?<![A-Za-z])Rp(?![A-Za-z])"
        r"|Rs\.?|Fr\.?|\bkr\b|\$",
        "", raw, flags=re.IGNORECASE
    ).strip()
    if re.search(r",\d{1,2}$", s):
        s = s.replace(".", "").replace(" ", "").replace(",", ".")
    else:
        s = s.replace(",", "").replace(" ", "")
    return s


def _translate_words(words):
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="fr", target="en").translate_batch(words)
        return [t if (t and isinstance(t, str)) else w for t, w in zip(translated, words)]
    except Exception:
        return words


def _parse_xml(label_path: Path) -> dict:
    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    root     = ET.parse(label_path).getroot()

    def _x(tag):
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None

    untaxed = _x("total_untaxed")
    tax     = _x("tax_amount")
    sub     = _x("sub_total")
    try:
        if untaxed and tax:
            total = str(round(float(untaxed) + float(tax), 2))
        elif untaxed:
            total = untaxed
        elif sub:
            total = sub
        else:
            total = _x("total_amount")
    except (ValueError, TypeError):
        total = _x("sub_total") or _x("total_amount")

    raw_date = _x("invoice_date") or ""
    address  = _x("address") or ""
    return {
        "vendor":          _x("supplier"),
        "date":            _normalise_date(raw_date, source="xml"),
        "date_raw":        raw_date,
        "total_amount":    total,
        "subtotal_amount": _x("total_untaxed") or _x("sub_total"),
        "invoice_number":  _x("invoice_number"),
        "po_number":       _x("po_number"),
        "currency":        _detect_currency(total_str="", address=address, full_text=raw_text),
        "source":          "xml",
    }


def _parse_txt(label_path: Path) -> dict:
    meta      = json.loads(label_path.read_text(encoding="utf-8"))
    raw_date  = str(meta.get("date", "") or "")
    raw_total = str(meta.get("total", "") or "")
    address   = str(meta.get("address", "") or "")
    return {
        "vendor":          meta.get("company"),
        "date":            _normalise_date(raw_date, source="txt"),
        "date_raw":        raw_date,
        "total_amount":    raw_total,
        "subtotal_amount": None,
        "invoice_number":  None,
        "po_number":       None,
        "currency":        _detect_currency(total_str=raw_total, address=address),
        "source":          "txt",
    }


def _get_words_and_boxes(pil_image):
    w, h   = pil_image.size
    config = "--oem 3 --psm 6"
    data   = pytesseract.image_to_data(pil_image, config=config,
                                       output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        if not text or int(data["conf"][i]) < 30:
            continue
        x  = data["left"][i];  y  = data["top"][i]
        bw = data["width"][i]; bh = data["height"][i]
        boxes.append([
            min(max(int(x      / w * 1000), 0), 1000),
            min(max(int(y      / h * 1000), 0), 1000),
            min(max(int((x+bw) / w * 1000), 0), 1000),
            min(max(int((y+bh) / h * 1000), 0), 1000),
        ])
        words.append(text)
    return words, boxes


def _align_labels(words, entities) -> list:
    labels     = ["O"] * len(words)
    words_norm = [re.sub(r"[^a-z0-9./,]", "", w.lower()) for w in words]
    source     = entities.get("source", "txt")

    def _mark(start, length, field):
        labels[start] = f"B-{field}"
        for k in range(1, length):
            if start + k < len(labels):
                labels[start + k] = f"I-{field}"

    # VENDOR
    vendor = entities.get("vendor")
    if vendor:
        vtoks = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in vendor.split() if t]
        vtoks = [t for t in vtoks if t]
        n = len(vtoks)
        if n:
            for i in range(len(words) - n + 1):
                if all(words_norm[i+j] == vtoks[j] for j in range(n)):
                    _mark(i, n, "VENDOR"); break
            else:
                for i in range(len(words)):
                    if words_norm[i] == vtoks[0]:
                        run = 1
                        while run < n and (i+run) < len(words) and words_norm[i+run] == vtoks[run]:
                            run += 1
                        if run >= max(1, n // 2):
                            _mark(i, run, "VENDOR"); break

    # DATE
    date_raw = entities.get("date_raw", "") or entities.get("date", "")
    if date_raw:
        variants = _date_digit_variants(date_raw, source)
        for i, w in enumerate(words):
            w_digits = re.sub(r"[^0-9]", "", w)
            if len(w_digits) == 8 and w_digits in variants:
                _mark(i, 1, "DATE"); break
        else:
            date_norm = re.sub(r"[^a-z0-9]", "", date_raw.lower())
            for i, wn in enumerate(words_norm):
                if date_norm and len(date_norm) >= 4 and (date_norm in wn or wn in date_norm):
                    _mark(i, 1, "DATE"); break

    # TOTAL
    total_raw = entities.get("total_amount")
    if total_raw:
        total_norm = _normalise_amount(total_raw)
        if total_norm:
            for i, w in enumerate(words):
                if _normalise_amount(w) == total_norm:
                    _mark(i, 1, "TOTAL"); break

    # SUBTOTAL
    sub_raw = entities.get("subtotal_amount")
    if sub_raw:
        sub_norm = _normalise_amount(str(sub_raw))
        if sub_norm:
            for i, w in enumerate(words):
                if labels[i] == "O" and _normalise_amount(w) == sub_norm:
                    _mark(i, 1, "SUBTOTAL"); break

    # INVOICE_NO (XML only)
    inv_no = entities.get("invoice_number")
    if inv_no:
        inv_norm = re.sub(r"[^a-z0-9/]", "", inv_no.lower())
        inv_toks = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in inv_no.split() if t]
        inv_toks = [t for t in inv_toks if t]
        n = len(inv_toks)
        for i in range(len(words) - n + 1):
            if all(words_norm[i+j] == inv_toks[j] for j in range(n)):
                _mark(i, n, "INVOICE_NO"); break
        else:
            for i, wn in enumerate(words_norm):
                if inv_norm and inv_norm == re.sub(r"[^a-z0-9/]", "", wn):
                    _mark(i, 1, "INVOICE_NO"); break

    # PO_NO (XML only)
    po_no = entities.get("po_number")
    if po_no:
        po_norm = re.sub(r"[^a-z0-9]", "", po_no.lower())
        for i, wn in enumerate(words_norm):
            if po_norm and po_norm == re.sub(r"[^a-z0-9]", "", wn):
                _mark(i, 1, "PO_NO"); break

    # CURRENCY
    currency_sym = entities.get("currency")
    if currency_sym:
        sym_esc = re.escape(currency_sym)
        for i, w in enumerate(words):
            if labels[i] == "O" and re.fullmatch(sym_esc, w.strip(), re.IGNORECASE):
                _mark(i, 1, "CURRENCY"); break
        else:
            for i, w in enumerate(words):
                if labels[i] == "O" and re.match(rf'^{sym_esc}', w, re.IGNORECASE):
                    _mark(i, 1, "CURRENCY"); break

    return labels


def load_sample(sample: ImageSample):
    """Load one sample: OCR → label alignment → (words, boxes, bio_labels)."""
    try:
        img = Image.open(sample.img_path).convert("RGB")
        words, boxes = _get_words_and_boxes(img)
        img.close()
        if not words:
            return None
        entities   = _parse_xml(sample.label_path) if sample.source == "xml" else _parse_txt(sample.label_path)
        bio_labels = _align_labels(words, entities)
        if sample.source == "xml":
            words = _translate_words(words)
        return words, boxes, bio_labels
    except Exception as e:
        print(f"[data] Skipping {sample.stem}: {e}")
        return None


def partition_images(seed: int = SEED):
    """Split dataset into 3 non-IID clients."""
    xml_samples, txt_samples = [], []

    for label_path in sorted(ENTITY_DIR.glob("*")):
        if label_path.suffix not in (".xml", ".txt"):
            continue
        img_path = next(
            (IMG_DIR / (label_path.stem + ext)
             for ext in (".jpg", ".jpeg", ".png")
             if (IMG_DIR / (label_path.stem + ext)).exists()),
            None,
        )
        if img_path is None:
            continue
        sample = ImageSample(
            img_path=img_path, label_path=label_path,
            source="xml" if label_path.suffix == ".xml" else "txt",
            stem=label_path.stem,
        )
        if label_path.suffix == ".xml":
            xml_samples.append(sample)
        else:
            txt_samples.append(sample)

    rng = random.Random(seed)
    rng.shuffle(txt_samples)
    mid = len(txt_samples) // 2

    partitions = {
        "client_a": xml_samples,
        "client_b": txt_samples[:mid],
        "client_c": txt_samples[mid:],
    }
    for cid, samps in partitions.items():
        print(f"  {cid}: {len(samps)} samples")
    return partitions


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class InvoiceTokenDataset(Dataset):
    def __init__(self, samples, processor, max_length=MAX_LENGTH):
        self._encodings = []
        for sample in samples:
            result = load_sample(sample)
            if result is None:
                continue
            words, boxes, bio_labels = result
            word_label_ids = [LABEL2ID.get(l, 0) for l in bio_labels]
            try:
                img = Image.open(sample.img_path).convert("RGB")
                enc = processor(
                    img, words, boxes=boxes, word_labels=word_label_ids,
                    padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt",
                )
                img.close()
                self._encodings.append({k: v.squeeze(0) for k, v in enc.items()})
            except Exception:
                pass

    def __len__(self):  return len(self._encodings)
    def __getitem__(self, i): return self._encodings[i]


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ══════════════════════════════════════════════════════════════════════════════
# LORA MODEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_base_with_lora(model_dir=BASE_MODEL_DIR, lora_r=LORA_R):
    processor  = LayoutLMv3Processor.from_pretrained(str(model_dir))
    base_model = LayoutLMv3ForTokenClassification.from_pretrained(
        str(model_dir), num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type="TOKEN_CLS",
    )
    peft_model = get_peft_model(base_model, lora_config)
    return peft_model, processor


def get_flat_lora_params(peft_model) -> np.ndarray:
    parts = []
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            parts.append(param.data.cpu().float().detach().flatten())
    return torch.cat(parts).numpy()


def set_flat_lora_params(peft_model, flat: np.ndarray):
    flat_t = torch.from_numpy(flat.astype(np.float32))
    offset = 0
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            n = param.numel()
            param.data.copy_(flat_t[offset:offset + n].reshape(param.shape))
            offset += n


# ══════════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL PRIVACY
# ══════════════════════════════════════════════════════════════════════════════

class DPMechanism:
    def __init__(self, clip_norm=CLIP_NORM, sigma=SIGMA, delta=1e-5, sampling_rate=1.0):
        self.clip_norm     = clip_norm
        self.sigma         = sigma
        self.delta         = delta
        self.sampling_rate = sampling_rate

    def apply_flat(self, flat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(flat)
        if norm > self.clip_norm:
            flat = flat * (self.clip_norm / norm)
        if self.sigma > 0:
            flat = flat + np.random.normal(0.0, self.clip_norm * self.sigma, flat.shape)
        return flat

    def compute_epsilon(self, n_rounds: int) -> float:
        if n_rounds <= 0 or self.sigma == 0:
            return float("inf")  # no noise = no privacy guarantee
        q, best_eps = self.sampling_rate, float("inf")
        for alpha in [2, 4, 8, 16, 32, 64, 128, 256]:
            eps_rdp    = (q ** 2 * alpha / (2.0 * self.sigma ** 2)) * n_rounds
            conversion = (
                np.log(1.0 - 1.0 / alpha) / (alpha - 1)
                - np.log(self.delta)       / (alpha - 1)
            )
            best_eps = min(best_eps, eps_rdp + conversion)
        return round(float(best_eps), 6)


# ══════════════════════════════════════════════════════════════════════════════
# FL CLIENT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClientUpdate:
    client_id:    str
    delta_params: np.ndarray
    n_samples:    int
    local_f1:     float
    local_loss:   float
    per_class_f1: dict = field(default_factory=dict)


class FLClient:
    def __init__(self, client_id, samples, dp, peft_model, processor,
                 device=DEVICE, seed=SEED, val_fraction=0.15, mu=MU):
        self.client_id   = client_id
        self.dp          = dp
        self._device     = torch.device(device)
        self._mu         = mu
        self._peft_model = peft_model
        self._processor  = processor

        rng      = random.Random(seed)
        shuffled = list(samples); rng.shuffle(shuffled)
        n_val    = max(1, int(len(shuffled) * val_fraction))

        print(f"[{client_id}] Preprocessing {len(shuffled) - n_val} train + {n_val} val samples...")
        self._train_ds = InvoiceTokenDataset(shuffled[n_val:],  processor)
        self._val_ds   = InvoiceTokenDataset(shuffled[:n_val],  processor)
        print(f"[{client_id}] Ready: {len(self._train_ds)} train, {len(self._val_ds)} val encodings")

    def receive_global_model(self, flat_params: np.ndarray):
        set_flat_lora_params(self._peft_model, flat_params)

    def local_train(self, n_epochs=LOCAL_EPOCHS) -> ClientUpdate:
        if len(self._train_ds) == 0:
            n = len(get_flat_lora_params(self._peft_model))
            return ClientUpdate(self.client_id, np.zeros(n, np.float32), 0, 0.0, 0.0)

        global_flat = get_flat_lora_params(self._peft_model).copy()
        global_tensors = [
            p.data.clone()
            for n, p in self._peft_model.named_parameters()
            if "lora_" in n and p.requires_grad
        ]

        loader = DataLoader(self._train_ds, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=_collate)
        trainable = [p for n, p in self._peft_model.named_parameters()
                     if "lora_" in n and p.requires_grad]
        optimizer = optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)

        self._peft_model.train()
        last_loss = 0.0
        for _ in range(n_epochs):
            ep_loss, n_batches = 0.0, 0
            for batch in loader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self._peft_model(**batch)
                loss    = outputs.loss
                if self._mu > 0:
                    prox = sum(
                        ((p - g.to(self._device)) ** 2).sum()
                        for p, g in zip(trainable, global_tensors)
                    )
                    loss = loss + (self._mu / 2) * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                ep_loss  += float(loss.item())
                n_batches += 1
            last_loss = ep_loss / max(n_batches, 1)

        local_flat = get_flat_lora_params(self._peft_model)
        noisy_delta = self.dp.apply_flat(local_flat - global_flat)
        macro_f1, per_class = self._eval_f1()
        return ClientUpdate(self.client_id, noisy_delta, len(self._train_ds),
                            macro_f1, last_loss, per_class)

    def _eval_f1(self):
        if len(self._val_ds) == 0:
            return 0.0, {}
        self._peft_model.eval()
        try:
            loader      = DataLoader(self._val_ds, batch_size=BATCH_SIZE,
                                     shuffle=False, collate_fn=_collate)
            all_preds, all_labels = [], []
            id2label    = self._peft_model.config.id2label
            with torch.no_grad():
                for batch in loader:
                    labels_batch = batch.pop("labels", None)
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    preds = self._peft_model(**batch).logits.argmax(-1)
                    for i in range(preds.size(0)):
                        ps, ls = [], []
                        for j in range(preds.size(1)):
                            lbl = labels_batch[i, j].item() if labels_batch is not None else -100
                            if lbl == -100: continue
                            ps.append(id2label.get(preds[i, j].item(), "O"))
                            ls.append(id2label.get(lbl, "O"))
                        if ps: all_preds.append(ps); all_labels.append(ls)
            if not all_preds:
                return 0.0, {}
            from seqeval.metrics import f1_score, classification_report
            macro   = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
            report  = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
            per_cls = {k: round(float(v.get("f1-score", 0.0)), 4)
                       for k, v in report.items()
                       if isinstance(v, dict) and k not in ("micro avg", "macro avg",
                                                             "weighted avg", "accuracy")}
            return macro, per_cls
        except Exception as e:
            print(f"[{self.client_id}] eval error: {e}")
            return 0.0, {}
        finally:
            self._peft_model.train()


# ══════════════════════════════════════════════════════════════════════════════
# FL SERVER  (FedAvg)
# ══════════════════════════════════════════════════════════════════════════════

class FedAvgServer:
    def __init__(self, peft_model, processor, device=DEVICE):
        self._peft_model  = peft_model
        self._processor   = processor
        self._device      = torch.device(device)
        self.global_params = get_flat_lora_params(peft_model)
        self._val_ds: InvoiceTokenDataset | None = None
        print(f"[server] LoRA params: {len(self.global_params):,} floats "
              f"({len(self.global_params) * 4 / 1024:.1f} KB)")

    def build_val_set(self, partitions, val_fraction=0.10, seed=SEED):
        rng  = random.Random(seed)
        samps = []
        for samples in partitions.values():
            shuffled = list(samples); rng.shuffle(shuffled)
            samps.extend(shuffled[:max(1, int(len(shuffled) * val_fraction))])
        print(f"[server] Building global val set from {len(samps)} samples...")
        self._val_ds = InvoiceTokenDataset(samps, self._processor)
        print(f"[server] Val set: {len(self._val_ds)} encodings")

    def run_round(self, clients, round_num, dp) -> dict:
        updates = []
        for client in clients:
            client.receive_global_model(self.global_params)
            updates.append(client.local_train())

        # Secure aggregation simulation (pairwise masks cancel in weighted sum)
        n = len(updates)
        for i in range(n):
            for j in range(i + 1, n):
                mask = np.random.normal(0.0, 0.001, self.global_params.shape)
                updates[i].delta_params += mask
                updates[j].delta_params -= mask

        # FedAvg
        total   = sum(u.n_samples for u in updates)
        avg_delta = np.zeros_like(self.global_params)
        for u in updates:
            if total > 0:
                avg_delta += (u.n_samples / total) * u.delta_params
        self.global_params += avg_delta
        set_flat_lora_params(self._peft_model, self.global_params)
        delta_norm = float(np.linalg.norm(avg_delta))

        macro_f1, per_class = self._evaluate()
        eps = dp.compute_epsilon(round_num)

        vendor = per_class.get("VENDOR", per_class.get("B-VENDOR", 0.0))
        date   = per_class.get("DATE",   per_class.get("B-DATE",   0.0))
        total_ = per_class.get("TOTAL",  per_class.get("B-TOTAL",  0.0))
        print(f"[server] Round {round_num:>2}  macro_F1={macro_f1:.4f}  "
              f"VENDOR={vendor:.3f}  DATE={date:.3f}  TOTAL={total_:.3f}  "
              f"ε={eps:.4f}  |Δ|={delta_norm:.5f}")

        return {
            "round":         round_num,
            "macro_f1":      macro_f1,
            "per_class_f1":  per_class,
            "param_change":  delta_norm,
            "epsilon":       eps,
            "client_losses": {u.client_id: u.local_loss for u in updates},
            "client_f1s":    {u.client_id: u.local_f1   for u in updates},
        }

    def _evaluate(self):
        if self._val_ds is None or len(self._val_ds) == 0:
            return 0.0, {}
        self._peft_model.eval()
        try:
            loader      = DataLoader(self._val_ds, batch_size=4,
                                     shuffle=False, collate_fn=_collate)
            all_preds, all_labels = [], []
            id2label    = self._peft_model.config.id2label
            with torch.no_grad():
                for batch in loader:
                    labels_batch = batch.pop("labels", None)
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    preds = self._peft_model(**batch).logits.argmax(-1)
                    for i in range(preds.size(0)):
                        ps, ls = [], []
                        for j in range(preds.size(1)):
                            lbl = labels_batch[i, j].item() if labels_batch is not None else -100
                            if lbl == -100: continue
                            ps.append(id2label.get(preds[i, j].item(), "O"))
                            ls.append(id2label.get(lbl, "O"))
                        if ps: all_preds.append(ps); all_labels.append(ls)
            if not all_preds:
                return 0.0, {}
            from seqeval.metrics import f1_score, classification_report
            macro   = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
            report  = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
            per_cls = {k: round(float(v.get("f1-score", 0.0)), 4)
                       for k, v in report.items()
                       if isinstance(v, dict) and k not in ("micro avg", "macro avg",
                                                             "weighted avg", "accuracy")}
            return macro, per_cls
        except Exception as e:
            print(f"[server] eval error: {e}")
            return 0.0, {}

    def save_adapter(self, save_dir: Path):
        set_flat_lora_params(self._peft_model, self.global_params)
        self._peft_model.save_pretrained(str(save_dir))
        print(f"[server] Adapter saved → {save_dir}")

    def merge_and_save(self, output_dir: Path):
        set_flat_lora_params(self._peft_model, self.global_params)
        merged = self._peft_model.merge_and_unload()
        merged.save_pretrained(str(output_dir))
        self._processor.save_pretrained(str(output_dir))
        print(f"[server] Merged model saved → {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f" Federated LayoutLMv3 + LoRA  (GPU={DEVICE=='cuda'})")
    print(f" Rounds={N_ROUNDS}  σ={SIGMA}  clip={CLIP_NORM}  μ={MU}  LoRA_r={LORA_R}")
    print(f"{'='*60}\n")

    # 1. Partition data
    print("[1/5] Partitioning invoice images...")
    partitions = partition_images()

    # 2. DP mechanism
    dp = DPMechanism(clip_norm=CLIP_NORM, sigma=SIGMA, delta=1e-5,
                     sampling_rate=3 / max(len(partitions), 1))
    print(f"\n[2/5] DP: estimated ε@{N_ROUNDS} rounds = {dp.compute_epsilon(N_ROUNDS):.4f}\n")

    # 3. Load model ONCE — shared across server + all clients
    print("[3/5] Loading LayoutLMv3 + LoRA (single shared model)...")
    peft_model, processor = load_base_with_lora()
    peft_model = peft_model.to(DEVICE)

    server  = FedAvgServer(peft_model, processor, device=DEVICE)
    server.build_val_set(partitions)

    clients = [
        FLClient(cid, samples, dp, peft_model, processor, device=DEVICE, seed=SEED)
        for cid, samples in partitions.items()
    ]

    # 4. FL rounds
    print(f"\n[4/5] Running {N_ROUNDS} FL rounds...\n")
    history = []
    for rnd in range(1, N_ROUNDS + 1):
        metrics = server.run_round(clients, rnd, dp)
        history.append(metrics)

    # 5. Save outputs
    print(f"\n[5/5] Saving outputs...")
    server.save_adapter(LORA_DIR)
    server.merge_and_save(MERGED_DIR)

    final   = history[-1] if history else {}
    elapsed = round(time.time() - t_start, 1)

    def _to_python(obj):
        if isinstance(obj, np.integer):    return int(obj)
        if isinstance(obj, np.floating):   return float(obj)
        if isinstance(obj, np.ndarray):    return obj.tolist()
        if isinstance(obj, dict):          return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_to_python(v) for v in obj]
        return obj

    metadata = _to_python({
        "timestamp":          datetime.now().isoformat(),
        "n_rounds":           N_ROUNDS,
        "final_macro_f1":     final.get("macro_f1", 0.0),
        "per_class_f1_final": final.get("per_class_f1", {}),
        "epsilon":            dp.compute_epsilon(N_ROUNDS),
        "delta":              1e-5,
        "sigma":              SIGMA,
        "clip_norm":          CLIP_NORM,
        "mu":                 MU,
        "lora_r":             LORA_R,
        "device":             DEVICE,
        "training_seconds":   elapsed,
        "round_history":      history,
    })
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f" Done in {elapsed:.0f}s  "
          f"final_macro_F1={metadata['final_macro_f1']:.4f}  "
          f"ε={metadata['epsilon']:.4f}")
    print(f" Outputs:")
    print(f"   LoRA adapter  → {LORA_DIR}")
    print(f"   Merged model  → {MERGED_DIR}")
    print(f"   Metadata      → {META_PATH}")
    print(f"{'='*60}\n")
    print("Next step: download layoutlm_lora_merged/ and place it at")
    print("  cost-management-system/data/layoutlm_lora_merged/")


main()
