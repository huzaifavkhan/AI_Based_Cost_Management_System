# RapidFuzz-based fuzzy string matching utilities
from rapidfuzz import fuzz

DEFAULT_THRESHOLD = 80      # General string similarity threshold (0–100)
VENDOR_THRESHOLD  = 70      # Lower threshold for vendor names (OCR errors common)


def similarity(a: str, b: str) -> float:
    """General-purpose similarity score using ratio (character-level)."""
    if not a or not b:
        return 0.0
    return fuzz.ratio(a.strip(), b.strip())


def partial_similarity(a: str, b: str) -> float:
    """Partial ratio — useful when one string is a substring of the other."""
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a.strip(), b.strip())


def token_similarity(a: str, b: str) -> float:
    """Token sort ratio — word order insensitive (good for vendor names)."""
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a.strip(), b.strip())


def is_match(a: str, b: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Returns True if any of the three scores exceeds threshold."""
    scores = [similarity(a, b), partial_similarity(a, b), token_similarity(a, b)]
    return max(scores) >= threshold


def best_score(a: str, b: str) -> float:
    """Returns the highest score across all three methods."""
    if not a or not b:
        return 0.0
    return max(similarity(a, b), partial_similarity(a, b), token_similarity(a, b))
