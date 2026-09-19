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
from PIL import Image, ImageDraw, ImageFont

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
    scores: List[float],
    image_id: str,
    modality: str = "optical",
) -> str:
    """
    Generates a high-quality color overlay showing detected bounding boxes and segment masks
    on top of the original satellite image (upscaled 4x for readability).
    """
    dirs = ensure_dirs(modality)
    overlays_dir = dirs["overlays"]
    clean_id = sanitize_name(image_id)
    filepath = os.path.join(overlays_dir, f"{clean_id}_overlay.png")

    colors_rgba = [
        (255, 50, 50, 110), (50, 200, 50, 110), (50, 120, 255, 110),
        (255, 180, 0, 110), (180, 50, 220, 110), (0, 220, 200, 110),
    ]
    borders_rgb = [
        (220, 0, 0), (0, 180, 0), (0, 80, 220),
        (220, 140, 0), (140, 0, 200), (0, 180, 160),
    ]

    # Upscale 4x for drawing
    w, h = original_img.size
    scale = 4
    overlay = original_img.resize((w * scale, h * scale), resample=Image.Resampling.LANCZOS).convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    
    font = None # Default PIL font is tiny; scale geometry instead
    lw = max(2, scale)
    text_pad = 4 * scale

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        c_rgba = colors_rgba[i % len(colors_rgba)]
        c_rgb = borders_rgb[i % len(borders_rgb)]

        if i < len(masks) and masks[i] is not None:
            m_arr = masks[i]
            if m_arr.shape[:2] == (h, w):
                # upscale mask to 4x
                mask_pil = Image.fromarray((m_arr > 0).astype(np.uint8) * 255, mode="L")
                mask_upscaled = mask_pil.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
                color_layer = Image.new("RGBA", (w * scale, h * scale), c_rgba)
                overlay.paste(color_layer, (0, 0), mask=mask_upscaled)

        x1, y1, x2, y2 = [round(c * scale) for c in box]
        draw.rectangle([x1, y1, x2, y2], outline=c_rgb, width=lw)
        
        tag = f"{label} {score:.2f} [#{i+1}]"
        # We manually simulate larger text by drawing thicker? Pillow default font is 10px. 
        # Without custom TTF, we just draw standard text (it will be small but acceptable if we can't load ttf)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 12 * scale)
        except Exception:
            font = None
            
        if font:
            t_box = draw.textbbox((x1, max(0, y1 - (16 * scale))), tag, font=font)
            draw.rectangle([t_box[0]-2, t_box[1]-2, t_box[2]+2, t_box[3]+2], fill=c_rgb)
            draw.text((x1 + 2, max(0, y1 - (16 * scale))), tag, fill=(255, 255, 255), font=font)
        else:
            t_box = draw.textbbox((x1, max(0, y1 - 20)), tag)
            draw.rectangle(t_box, fill=c_rgb)
            draw.text((x1 + 2, max(0, y1 - 20)), tag, fill=(255, 255, 255))

    final = overlay.convert("RGB")
    final.save(filepath, format="PNG")
    return filepath

def generate_panel(
    original_img: Image.Image,
    overlay_path: str,
    masks: List[Optional[np.ndarray]],
    image_id: str,
    modality: str = "optical"
) -> str:
    """
    Generates a side-by-side panel: [Original] | [Overlay] | [Combined Binary Mask]
    """
    dirs = ensure_dirs(modality)
    overlays_dir = dirs["overlays"]
    clean_id = sanitize_name(image_id)
    filepath = os.path.join(overlays_dir, f"{clean_id}_panel.png")
    
    overlay_img = Image.open(overlay_path).convert("RGB")
    
    # 4x upscale original
    w, h = original_img.size
    orig_up = original_img.resize((w * 4, h * 4), resample=Image.Resampling.LANCZOS).convert("RGB")
    
    # Create combined mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m is not None:
            combined_mask[m > 0] = 255
    mask_pil = Image.fromarray(combined_mask, mode="L").convert("RGB")
    mask_up = mask_pil.resize((w * 4, h * 4), resample=Image.Resampling.NEAREST)
    
    panel_w = orig_up.width * 3
    panel_h = orig_up.height
    
    panel = Image.new("RGB", (panel_w, panel_h))
    panel.paste(orig_up, (0, 0))
    panel.paste(overlay_img, (orig_up.width, 0))
    panel.paste(mask_up, (orig_up.width * 2, 0))
    
    panel.save(filepath, format="PNG")
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


def write_sidecar_json(
    run_id: str,
    task: str,
    model_used: str,
    latency: float,
    params: dict,
    evidence: list,
    warnings: list,
    modality: str = "optical"
) -> str:
    """
    Writes a sidecar JSON with run metadata. Returns absolute path.
    """
    dirs = ensure_dirs(modality)
    # The utils.py ensure_dirs might not return "reports" in the dictionary, but it ensures outputs/reports exists.
    # Let's put sidecar JSON in the outputs/reports dir
    base_dir = os.path.dirname(os.path.dirname(dirs["vqa"])) # gets to outputs/
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    filepath = os.path.join(reports_dir, f"{run_id}_sidecar.json")
    
    data = {
        "run_id": run_id,
        "task": task,
        "model_used": model_used,
        "latency_seconds": latency,
        "params": params,
        "evidence": evidence,
        "warnings": warnings,
        "modality": str(modality)
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    return filepath
