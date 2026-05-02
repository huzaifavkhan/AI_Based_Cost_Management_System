# Image preprocessing with OpenCV — improves Tesseract accuracy on noisy/skewed scans
import cv2
import numpy as np
from PIL import Image


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Full preprocessing pipeline for receipt/invoice images:
      1. Convert to grayscale
      2. Upscale to at least 2400px on the long side (Tesseract loves high DPI)
      3. Denoise
      4. Adaptive threshold (binarize)
      5. Deskew
    Falls back gracefully if any step fails.
    """
    try:
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Step 1: Upscale if image is small — Tesseract accuracy drops below ~150 DPI
        gray = _upscale(gray, min_long_side=2400)

        # Step 2: Denoise — removes background texture (wood grain, paper noise)
        gray = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

        # Step 3: Increase contrast with CLAHE before thresholding
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Step 4: Adaptive threshold — handles uneven lighting across the image
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 15          # larger block size (31) works better on real photos
        )

        # Step 5: Morphological cleanup — remove tiny noise specks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Step 6: Deskew
        try:
            binary = _deskew(binary)
        except Exception:
            pass

        return Image.fromarray(binary)

    except Exception:
        # Last resort: grayscale only
        return pil_image.convert("L")


def _upscale(gray: np.ndarray, min_long_side: int = 2400) -> np.ndarray:
    """Upscale image so the long side is at least min_long_side pixels."""
    h, w = gray.shape
    long_side = max(h, w)
    if long_side >= min_long_side:
        return gray
    scale = min_long_side / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _deskew(binary_img: np.ndarray) -> np.ndarray:
    """Detect and correct rotation angle using minAreaRect on dark pixel coords."""
    coords = np.column_stack(np.where(binary_img < 128))
    if len(coords) < 50:
        return binary_img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:
        return binary_img

    (h, w) = binary_img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary_img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
