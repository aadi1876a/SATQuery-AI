"""
backend/models/vqa_captioning_grounding/preprocessing.py
=============================================================================
SatQuery AI — P2 Image Preprocessing
=============================================================================
Handles both optical and SAR satellite imagery preprocessing for VLM input.
All functions convert inputs to 8-bit RGB PIL Images.

SAR Note: The pretrained VLMs (BLIP, OWLv2) are NOT native SAR models.
They receive SAR-derived visualizations:
  - Lee speckle filtering
  - dB transform for dynamic range compression
  - CLAHE contrast enhancement
  - False-color mapping (backscatter → RGB channels)
This is clearly documented as SAR visualization input, NOT native SAR understanding.
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple

# Add backend to sys.path for schema import
import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
for p in [_BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from schemas import ImageObject, Modality


# ---------------------------------------------------------------------------
# Low-level band processors
# ---------------------------------------------------------------------------

def percentile_stretch(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """
    Percentile contrast stretching for 16-bit satellite GeoTIFFs.
    Maps the [low, high] percentile range to [0, 255] uint8.
    """
    b_float = band.astype(np.float32)
    p_low, p_high = np.percentile(b_float, low), np.percentile(b_float, high)
    if p_high > p_low:
        stretched = np.clip((b_float - p_low) / (p_high - p_low), 0.0, 1.0) * 255.0
    else:
        stretched = np.clip(b_float, 0, 255)
    return stretched.astype(np.uint8)


def lee_speckle_filter(band: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Lee speckle filter — standard SAR denoising algorithm.
    Reduces multiplicative SAR speckle noise while preserving edges.
    Dramatically improves VLM perception of SAR imagery.
    """
    from scipy.ndimage import uniform_filter
    band = band.astype(np.float32)
    mean = uniform_filter(band, kernel_size)
    mean_sq = uniform_filter(band ** 2, kernel_size)
    var = mean_sq - mean ** 2
    # Noise variance estimate (global)
    noise_var = np.mean(var) / (np.mean(mean) ** 2 + 1e-10)
    # Local SNR weight
    weight = var / (var + noise_var * mean ** 2 + 1e-10)
    filtered = mean + weight * (band - mean)
    return filtered.astype(np.float32)


def clahe_enhance(band_uint8: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances local contrast in SAR images — makes urban/water/vegetation
    boundaries more visible to VLMs without blowing out bright targets.
    """
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        return clahe.apply(band_uint8)
    except ImportError:
        import PIL.ImageOps
        pil_band = Image.fromarray(band_uint8, mode="L")
        return np.array(PIL.ImageOps.equalize(pil_band))


def sar_to_false_color_rgb(band: np.ndarray) -> np.ndarray:
    """
    Converts single-band SAR backscatter to false-color RGB.
    Backscatter intensity ranges map to physically meaningful colors:
      Low backscatter  (water/shadow)     → Blue channel
      Mid backscatter  (vegetation)       → Green channel
      High backscatter (urban/metal)      → Red channel

    This makes SAR backscatter physically interpretable by optical VLMs.
    """
    db_band = np.where(band > 0, 10.0 * np.log10(band + 1e-10), -30.0)
    db_min, db_max = np.percentile(db_band, 2), np.percentile(db_band, 98)
    db_norm = np.clip((db_band - db_min) / (db_max - db_min + 1e-10), 0.0, 1.0)

    r = np.clip(db_norm * 2.0 - 1.0, 0.0, 1.0)         # high backscatter → red
    g = np.clip(1.0 - np.abs(db_norm - 0.5) * 2.0, 0.0, 1.0)  # mid → green
    b = np.clip(1.0 - db_norm * 2.0, 0.0, 1.0)         # low backscatter → blue

    rgb = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
    ], axis=-1)
    return rgb


def preprocess_sar_band(band: np.ndarray, use_false_color: bool = True) -> np.ndarray:
    """
    Full SAR preprocessing pipeline for VLM input:
      Step 1: Lee speckle filter     → reduce noise
      Step 2: Log (dB) transform     → compress dynamic range
      Step 3: CLAHE enhancement      → boost local contrast
      Step 4: False-color mapping    → make backscatter physically visible
               OR
               Grayscale 3-channel   → simpler approach for grounding

    Args:
        band: Raw SAR backscatter array (float32, linear scale)
        use_false_color: If True, output false-color RGB (recommended for VQA/Caption)
                         If False, output grayscale-as-RGB (simpler, for grounding)

    Returns:
        np.ndarray of shape (H, W, 3) uint8 — ready for PIL/VLM

    NOTE: Output is a SAR-derived visualization. The model does NOT receive
    raw SAR backscatter. Results reflect interpretation of the visual pattern.
    """
    # Step 1: Speckle filter
    try:
        band_filtered = lee_speckle_filter(band)
    except Exception:
        band_filtered = band.astype(np.float32)

    if use_false_color:
        # Step 4a: False-color RGB (best for BLIP VQA/captioning)
        rgb = sar_to_false_color_rgb(band_filtered)
    else:
        # Step 2: dB transform
        db = np.where(band_filtered > 0, 10.0 * np.log10(band_filtered + 1e-10), -30.0)
        # Step 3: CLAHE on uint8
        stretched = percentile_stretch(db)
        enhanced = clahe_enhance(stretched)
        # Step 4b: Grayscale → 3 channel (best for OWLv2 grounding)
        rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)

    return rgb


# ---------------------------------------------------------------------------
# Array → RGB converter
# ---------------------------------------------------------------------------

def convert_array_to_rgb(arr: np.ndarray, path: str = "") -> Tuple[Image.Image, str]:
    """Converts any numpy array (2D / 3D / float / int) to 8-bit RGB PIL Image."""
    if arr.ndim == 2:
        stretched = percentile_stretch(arr)
        rgb_arr = np.stack([stretched, stretched, stretched], axis=-1)
    elif arr.ndim == 3:
        # Handle (C, H, W) layout from rasterio/satellite data
        if arr.shape[0] in (1, 2, 3, 4, 12, 13) and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
        channels = arr.shape[-1]
        if channels >= 3:
            r = percentile_stretch(arr[:, :, 0])
            g = percentile_stretch(arr[:, :, 1])
            b = percentile_stretch(arr[:, :, 2])
            rgb_arr = np.stack([r, g, b], axis=-1)
        else:
            stretched = percentile_stretch(arr[:, :, 0])
            rgb_arr = np.stack([stretched, stretched, stretched], axis=-1)
    else:
        raise ValueError(f"Unsupported image dimensions: {arr.ndim} for '{path}'")
    return Image.fromarray(rgb_arr.astype(np.uint8), mode="RGB"), path


def force_load_as_numpy(path: str) -> np.ndarray:
    """Last-resort loader for exotic GeoTIFF formats."""
    with open(path, "rb") as f:
        header = f.read(4)
    if header[:2] in (b"II", b"MM"):
        from PIL import TiffImagePlugin
        img = Image.open(path)
        img.load()
        try:
            arr = np.array(img, dtype=np.float32)
        except TypeError:
            arr = np.array(list(img.getdata()), dtype=np.float32)
            arr = arr.reshape(img.size[1], img.size[0], -1)
        return arr
    else:
        raise ValueError("Not a recognizable TIFF file format.")


# ---------------------------------------------------------------------------
# Main entry point: load any satellite image as RGB
# ---------------------------------------------------------------------------

def load_image_rgb(image_obj: ImageObject, use_false_color_sar: bool = True) -> Tuple[Image.Image, str]:
    """
    Safely loads satellite imagery (PNG, JPG, TIFF, SAR GeoTIFF) and converts
    to standard 8-bit RGB PIL Image for Vision-Language Models.

    Handles:
    - Standard PNG / JPG / RGBA / L modes.
    - 16-bit GeoTIFF (Sentinel-2 optical).
    - SAR GeoTIFF (Sentinel-1 GRDH) with complex/float data modes.
    - Rasterio-based reading for maximum GeoTIFF compatibility.

    SAR images are converted using the full SAR preprocessing pipeline
    (Lee filter → dB → CLAHE → false-color/grayscale).

    Args:
        image_obj: ImageObject from schemas.py
        use_false_color_sar: If True, SAR bands use false-color mapping (VQA/captioning).
                             If False, SAR uses grayscale-as-RGB (grounding).

    Returns:
        (PIL.Image in RGB mode, path used)
    """
    target_path = image_obj.file_path
    if not os.path.exists(target_path):
        if image_obj.thumbnail_path and os.path.exists(image_obj.thumbnail_path):
            target_path = image_obj.thumbnail_path
        else:
            raise FileNotFoundError(
                f"Image not found at '{target_path}' or thumbnail '{image_obj.thumbnail_path}'"
            )

    ext = os.path.splitext(target_path)[1].lower()
    modality = image_obj.modality

    # -----------------------------------------------------------------------
    # Strategy 1: Rasterio (most reliable for GeoTIFF / SAR)
    # -----------------------------------------------------------------------
    if ext in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(target_path) as src:
                bands = src.count
                if modality == Modality.sar or bands == 1:
                    # SAR single-band (Sentinel-1 GRD VV/VH) or forced SAR modality
                    band = src.read(1).astype(np.float32)
                    rgb_arr = preprocess_sar_band(band, use_false_color=use_false_color_sar)
                elif bands >= 3:
                    # Optical multispectral
                    r = src.read(1).astype(np.float32)
                    g = src.read(2).astype(np.float32)
                    b = src.read(3).astype(np.float32)
                    rgb_arr = np.stack([
                        percentile_stretch(r),
                        percentile_stretch(g),
                        percentile_stretch(b),
                    ], axis=-1)
                else:
                    band = src.read(1).astype(np.float32)
                    rgb_arr = preprocess_sar_band(band, use_false_color=use_false_color_sar)
            return Image.fromarray(rgb_arr.astype(np.uint8), mode="RGB"), target_path
        except ImportError:
            pass
        except Exception as rasterio_err:
            print(f"[P2 Preprocessing] Rasterio notice for '{os.path.basename(target_path)}': "
                  f"{rasterio_err}. Trying PIL fallback.")

    # -----------------------------------------------------------------------
    # Strategy 2: PIL-based loading (PNG, JPG, standard formats)
    # -----------------------------------------------------------------------
    try:
        raw_img = Image.open(target_path)
        if modality == Modality.sar:
            # For SAR images in standard image formats (PNG, JPG), run full SAR preprocessing
            gray_arr = np.array(raw_img.convert("L"), dtype=np.float32)
            rgb_arr = preprocess_sar_band(gray_arr, use_false_color=use_false_color_sar)
            return Image.fromarray(rgb_arr.astype(np.uint8), mode="RGB"), target_path

        if raw_img.mode in ("RGB", "RGBA", "L", "P"):
            return raw_img.convert("RGB"), target_path
        try:
            arr = np.array(raw_img)
        except Exception:
            width, height = raw_img.size
            raw_bytes = raw_img.tobytes()
            arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(height, width, -1)
        return convert_array_to_rgb(arr, target_path)
    except Exception as pil_err:
        # -----------------------------------------------------------------------
        # Strategy 3: Raw binary NumPy fallback
        # -----------------------------------------------------------------------
        try:
            arr = force_load_as_numpy(target_path)
            return convert_array_to_rgb(arr, target_path)
        except Exception as numpy_err:
            raise ValueError(
                f"Could not load '{os.path.basename(target_path)}' with any method.\n"
                f"PIL error: {pil_err}\n"
                f"NumPy error: {numpy_err}\n"
                f"Tip: Install rasterio for full GeoTIFF/SAR support: pip install rasterio"
            )
