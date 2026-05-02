"""
Download 25 sample images from SROIE and CORD datasets for OCR generalization testing.
Run: python scripts/download_external_samples.py

Requires: pip install datasets
Saves images to: data/external/sroie/ and data/external/cord/
"""
from pathlib import Path

SROIE_DIR = Path(__file__).parent.parent / "data" / "external" / "sroie"
CORD_DIR  = Path(__file__).parent.parent / "data" / "external" / "cord"
SROIE_DIR.mkdir(parents=True, exist_ok=True)
CORD_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 15   # images per dataset


def download_sroie():
    print("Downloading SROIE samples...")
    try:
        from datasets import load_dataset
        ds = load_dataset("darentang/sroie", split="test", trust_remote_code=True)
        for i, sample in enumerate(ds):
            if i >= SAMPLE_SIZE:
                break
            img = sample["image"]
            img.save(str(SROIE_DIR / f"sroie_{i:03d}.png"))
            print(f"  Saved sroie_{i:03d}.png")
        print(f"✓ {SAMPLE_SIZE} SROIE images saved to {SROIE_DIR}\n")
    except Exception as e:
        print(f"⚠ SROIE download failed: {e}")
        print("  Try: pip install datasets && huggingface-cli login (if private)\n")


def download_cord():
    print("Downloading CORD samples...")
    try:
        from datasets import load_dataset
        ds = load_dataset("naver-clova-ix/cord-v1", split="test", trust_remote_code=True)
        for i, sample in enumerate(ds):
            if i >= SAMPLE_SIZE:
                break
            img = sample["image"]
            img.save(str(CORD_DIR / f"cord_{i:03d}.png"))
            print(f"  Saved cord_{i:03d}.png")
        print(f"✓ {SAMPLE_SIZE} CORD images saved to {CORD_DIR}\n")
    except Exception as e:
        print(f"⚠ CORD download failed: {e}")
        print("  Try: pip install datasets\n")


if __name__ == "__main__":
    download_sroie()
    download_cord()
    print("✅ External samples downloaded.")
    print(f"   SROIE: {SROIE_DIR}")
    print(f"   CORD:  {CORD_DIR}")
