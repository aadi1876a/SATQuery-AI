"""
backend/models/vqa_captioning_grounding/utils.py
=============================================================================
SatQuery AI — P2 Utilities
=============================================================================
Shared helpers: device detection, model cache, sanitizers, path builders.
"""

import os
import re
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(_CURRENT_DIR, "outputs")

DOMAIN_PROMPT = "In this satellite remote sensing image: {q}"

# ---------------------------------------------------------------------------
# Model Cache (prevents reloading large models on every request)
# ---------------------------------------------------------------------------
import threading

_CACHED_MODELS: dict = {}
_DEVICE: str = None
_CACHE_LOCK = threading.Lock()

def generate_run_id() -> str:
    import time
    import hashlib
    ts = str(time.time())
    h = hashlib.md5(ts.encode()).hexdigest()[:6]
    return f"{int(float(ts))}_{h}"



def get_cached(key: str):
    """Returns a cached model tuple, or None if not yet loaded."""
    with _CACHE_LOCK:
        return _CACHED_MODELS.get(key)


def set_cached(key: str, value):
    """Caches a model tuple under the given key."""
    with _CACHE_LOCK:
        _CACHED_MODELS[key] = value


# ---------------------------------------------------------------------------
# Output Path Helpers
# ---------------------------------------------------------------------------

def get_modality_dirs(modality: str) -> dict:
    """
    Returns the canonical output directories for a given modality.

    Args:
        modality: 'optical' or 'sar' (or Modality enum or string representation)

    Returns:
        dict with keys: vqa, captioning, masks, overlays, bboxes, logs, reports
    """
    if hasattr(modality, "value"):
        modality = modality.value
    elif isinstance(modality, str) and "." in modality:
        modality = modality.split(".")[-1]
    modality = str(modality).lower().strip()
    if modality not in ["optical", "sar"]:
        modality = "optical"

    base = os.path.join(OUTPUTS_DIR, modality)
    return {
        "vqa":       os.path.join(base, "vqa"),
        "captioning": os.path.join(base, "captioning"),
        "masks":     os.path.join(base, "grounding", "masks"),
        "overlays":  os.path.join(base, "grounding", "overlays"),
        "bboxes":    os.path.join(base, "grounding", "bboxes"),
    }


def ensure_dirs(modality: str = None) -> dict:
    """Creates output directories for the given modality (or both if None) and returns paths."""
    modalities = ["optical", "sar"] if modality is None else [modality]
    all_dirs = {}
    for m in modalities:
        dirs = get_modality_dirs(m)
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
        all_dirs[m] = dirs
    os.makedirs(os.path.join(OUTPUTS_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUTS_DIR, "reports"), exist_ok=True)
    return all_dirs if modality is None else all_dirs[modality]


# ---------------------------------------------------------------------------
# String / Name Sanitizers
# ---------------------------------------------------------------------------

def sanitize_name(text: str) -> str:
    """Converts arbitrary text into a safe filename component."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', text.strip()).strip('_') or "object"


def safe_plural(word: str) -> str:
    """
    Returns a simple plural form of the word without duplicating trailing 's'.
    Avoids the bug where 'roads' → 'roadss'.
    """
    word = word.strip().lower()
    if not word:
        return word
    # Already plural (ends in s, es, etc.) — return as-is
    if word.endswith(("s", "es")):
        return word
    # Simple English pluralization rules
    if word.endswith(("sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"
