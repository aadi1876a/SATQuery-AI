"""
backend/models/vqa_captioning_grounding/postprocessing.py
=============================================================================
SatQuery AI — P2 Postprocessing
=============================================================================
Handles mask saving, overlay generation, and bbox JSON export.
All outputs are stored under the modality-organized output directory.
"""

import os
import json
import re
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw

from .utils import sanitize_name, ensure_dirs


def save_mask(
    mask_arr: np.ndarray,
    image_id: str,
    label: str,
    idx: int,
    modality: str = "optical",
) -> str:
    """
    Saves a binary mask as a PNG file under the correct modality directory.

    Args:
        mask_arr: Boolean or uint8 numpy array (H, W)
        image_id: Unique image identifier
        label: Object label (e.g. 'building', 'water body')
        idx: Detection index (1-based)
        modality: 'optical' or 'sar'

    Returns:
        Absolute path to the saved mask PNG.

    Raises:
        ValueError if the mask array is empty or all-zero.
    """
    dirs = ensure_dirs(modality)
    masks_dir = dirs["masks"]

    clean_id = sanitize_name(image_id)
    clean_label = sanitize_name(label)
    filename = f"{clean_id}_{clean_label}_{idx:03d}.png"
    filepath = os.path.join(masks_dir, filename)

    binary_mask = np.where(mask_arr > 0, 255, 0).astype(np.uint8)
    Image.fromarray(binary_mask, mode="L").save(filepath, format="PNG")
    return filepath


def validate_mask(mask_path: str) -> dict:
    """
    Validates a saved mask file.

    Returns:
        dict with keys: exists, readable, non_empty, pixel_count, pct_coverage
    """
    result = {"path": mask_path, "exists": False, "readable": False,
               "non_empty": False, "pixel_count": 0, "pct_coverage": 0.0}
    if not os.path.exists(mask_path):
        return result
    result["exists"] = True
    try:
        img = Image.open(mask_path)
        arr = np.array(img)
        result["readable"] = True
        result["pixel_count"] = int(np.sum(arr > 0))
        total = arr.size
        result["pct_coverage"] = round(100.0 * result["pixel_count"] / max(total, 1), 2)
        result["non_empty"] = result["pixel_count"] > 0
    except Exception as e:
        result["read_error"] = str(e)
    return result


def generate_overlay(
    original_img: Image.Image,
    boxes: List[List[float]],
    masks: List[Optional[np.ndarray]],
    labels: List[str],
    image_id: str,
    modality: str = "optical",
) -> str:
    """
    Generates a color overlay showing detected bounding boxes and segment masks
    on top of the original satellite image.

    Args:
        original_img: The original PIL Image
        boxes: List of [x1, y1, x2, y2] pixel coordinate boxes
        masks: List of boolean/uint8 numpy masks (H, W), or None per box
        labels: List of label strings
        image_id: Used for filename
        modality: 'optical' or 'sar'

    Returns:
        Absolute path to the saved overlay PNG.
    """
    dirs = ensure_dirs(modality)
    overlays_dir = dirs["overlays"]
    clean_id = sanitize_name(image_id)
    filepath = os.path.join(overlays_dir, f"{clean_id}_overlay.png")

    colors_rgba = [
        (255, 50, 50, 110),
        (50, 200, 50, 110),
        (50, 120, 255, 110),
        (255, 180, 0, 110),
        (180, 50, 220, 110),
        (0, 220, 200, 110),
    ]
    borders_rgb = [
        (220, 0, 0),
        (0, 180, 0),
        (0, 80, 220),
        (220, 140, 0),
        (140, 0, 200),
        (0, 180, 160),
    ]

    overlay = original_img.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    w, h = original_img.size

    for i, (box, label) in enumerate(zip(boxes, labels)):
        # Use purple for all detected regions
        c_rgba = (180, 50, 220, 110)
        c_rgb = (140, 0, 200)

        if i < len(masks) and masks[i] is not None:
            m_arr = masks[i]
            if m_arr.shape[:2] == (h, w):
                color_layer = Image.new("RGBA", (w, h), c_rgba)
                mask_pil = Image.fromarray((m_arr > 0).astype(np.uint8) * 255, mode="L")
                overlay.paste(color_layer, (0, 0), mask=mask_pil)

        # Bounding boxes and labels have been removed per user request

    final = overlay.convert("RGB")
    final.save(filepath, format="PNG")
    return filepath


def save_bboxes_json(
    boxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    image_id: str,
    modality: str = "optical",
) -> str:
    """
    Saves bounding box detections as a JSON file.

    Returns:
        Absolute path to the saved JSON file.
    """
    dirs = ensure_dirs(modality)
    bboxes_dir = dirs["bboxes"]
    clean_id = sanitize_name(image_id)
    filepath = os.path.join(bboxes_dir, f"{clean_id}_bboxes.json")

    detections = []
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        detections.append({
            "detection_id": i + 1,
            "label": label,
            "score": round(float(score), 4),
            "pixel_coords": {
                "x1": round(box[0], 2),
                "y1": round(box[1], 2),
                "x2": round(box[2], 2),
                "y2": round(box[3], 2),
            },
            "coordinate_system": "image_pixels",
            "note": "These are pixel coordinates, not geographic lat/lon."
        })

    data = {"image_id": image_id, "modality": modality, "detections": detections}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return filepath
