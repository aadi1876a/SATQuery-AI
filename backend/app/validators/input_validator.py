"""Geospatial Input Validator and Metadata Extractor for SatQuery AI.

P1 Deliverable:
- extract_metadata(): Extracts dimensions, CRS, bounds, acquisition date, and modality.
- check_co_registration(): Validates CRS match and spatial bounds alignment.
- validate(): Validates input images, enforces modality/count constraints, and categorizes input configuration.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False

from backend.app.schemas.contracts import (
    ImageMeta,
    InputConfig,
    Modality,
    ValidationResult,
)

# Supported image formats
SUPPORTED_FORMATS = {"GeoTIFF", "TIFF", "PNG", "JPEG"}

# Regex patterns for extracting dates from tags or filenames
DATE_PATTERNS = [
    # YYYY-MM-DD or YYYY_MM_DD or YYYY/MM/DD with optional time
    re.compile(r"(\d{4})[-_/](\d{2})[-_/](\d{2})(?:[T\s](\d{2})[-_:](\d{2})[-_:](\d{2}))?"),
    # YYYYMMDDTHHMMSS (Sentinel style)
    re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})"),
    # YYYYMMDD compact (e.g. 20240115)
    re.compile(r"(?:^|[^0-9])(20\d{2}|19\d{2})(\d{2})(\d{2})(?:[^0-9]|$)"),
]


def _normalize_date_string(date_str: str) -> Optional[str]:
    """Parse and normalize date strings to ISO YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format."""
    if not date_str:
        return None
    date_str = str(date_str).strip()

    # Try common formats first
    common_formats = [
        "%Y:%m:%d %H:%M:%S",  # TIFF standard tag format
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%dT%H%M%S",
        "%Y%m%d",
    ]
    for fmt in common_formats:
        try:
            dt = datetime.strptime(date_str[:19], fmt)
            if "%H" in fmt:
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

    # Try regex matchers
    for pattern in DATE_PATTERNS:
        match = pattern.search(date_str)
        if match:
            groups = match.groups()
            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
            if 1970 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                if len(groups) >= 6 and groups[3] is not None:
                    hour, minute, sec = int(groups[3]), int(groups[4]), int(groups[5])
                    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{sec:02d}"
                return f"{year:04d}-{month:02d}-{day:02d}"

    return None


def _extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract acquisition date from common remote sensing filename conventions."""
    return _normalize_date_string(filename)


def _detect_format(filepath: str, raw_format: Optional[str] = None) -> str:
    """Detect human-readable standard format name."""
    ext = os.path.splitext(filepath)[1].lower()
    if raw_format:
        rf = raw_format.upper()
        if "GTIF" in rf or "GEOTIFF" in rf:
            return "GeoTIFF"
        if "TIFF" in rf or "TIF" in rf:
            return "TIFF"
        if "PNG" in rf:
            return "PNG"
        if "JPEG" in rf or "JPG" in rf:
            return "JPEG"

    if ext in (".tif", ".tiff"):
        return "GeoTIFF"
    if ext == ".png":
        return "PNG"
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    return "UNKNOWN"


def _infer_modality_rasterio(src) -> Modality:
    """Infer whether a raster dataset is OPTICAL, SAR, or UNKNOWN."""
    try:
        count = src.count
        descriptions = [d.upper() for d in (src.descriptions or []) if d]
        tags = {k.upper(): str(v).upper() for k, v in src.tags().items()}

        # 1. Check explicit metadata hints (descriptions or tags)
        sar_keywords = ["VV", "VH", "HH", "HV", "SAR", "BACKSCATTER", "SIGMA0", "GAMMA0", "BETA0"]
        optical_keywords = ["RED", "GREEN", "BLUE", "NIR", "SWIR", "RGB", "TRUE COLOR", "MULTISPECTRAL"]

        text_evidence = " ".join(descriptions) + " " + " ".join(tags.values())
        if any(kw in text_evidence for kw in sar_keywords):
            return Modality.SAR
        if any(kw in text_evidence for kw in optical_keywords):
            return Modality.OPTICAL

        # 2. Heuristic by band count
        if count >= 3:
            # Multispectral / RGB satellite imagery typically has >= 3 bands
            return Modality.OPTICAL
        if count in (1, 2):
            # SAR sensors (Sentinel-1, TerraSAR-X, etc.) typically supply 1 (single-pol) or 2 bands (dual-pol)
            # Inspect pixel value distribution for confirmation
            try:
                # Read a small downsampled window from the first band
                out_h = min(src.height, 64)
                out_w = min(src.width, 64)
                sample = src.read(1, out_shape=(out_h, out_w)).astype(np.float32)
                valid = sample[np.isfinite(sample)]
                if len(valid) > 0:
                    # SAR backscatter in dB has negative values, or linear has large speckle/skew
                    if np.any(valid < 0):
                        return Modality.SAR
            except Exception:
                pass
            return Modality.SAR

        return Modality.UNKNOWN
    except Exception:
        return Modality.UNKNOWN


def _extract_date_rasterio(src, filename: str) -> Optional[str]:
    """Extract acquisition date from raster tags or filename."""
    try:
        tags = src.tags()
        date_keys = [
            "TIFFTAG_DATETIME",
            "DATETIME",
            "ACQUISITION_DATE",
            "acquisition_date",
            "DATE_ACQUIRED",
            "TIME",
            "start_time",
            "stop_time",
            "ImgDateTime",
        ]
        for key in date_keys:
            if key in tags and tags[key]:
                parsed = _normalize_date_string(tags[key])
                if parsed:
                    return parsed
    except Exception:
        pass

    # Fallback: parse from filename
    return _extract_date_from_filename(filename)


def extract_metadata(filepath: str) -> ImageMeta:
    """
    Open a GeoTIFF/TIFF/PNG/JPEG and extract format, dimensions, CRS,
    bounds, acquisition date, and band count using rasterio.
    Modality detection: infer optical vs SAR primarily from band count
    (SAR is typically 1-2 bands; multispectral optical is higher),
    supplemented by pixel value distribution if ambiguous.
    Must never crash on malformed/missing metadata — degrade to
    modality=UNKNOWN, crs=None rather than raising.
    """
    filename = os.path.basename(filepath)

    if not os.path.isfile(filepath):
        return ImageMeta(
            filename=filename,
            format="UNKNOWN",
            modality=Modality.UNKNOWN,
            width=0,
            height=0,
            crs=None,
            acquisition_date=None,
            bounds=None,
        )

    # Attempt to open with rasterio first
    if HAS_RASTERIO:
        try:
            with rasterio.open(filepath) as src:
                raw_format = src.driver
                detected_format = _detect_format(filepath, raw_format)
                width = int(src.width)
                height = int(src.height)

                # CRS extraction
                crs_str: Optional[str] = None
                if src.crs is not None:
                    try:
                        epsg = src.crs.to_epsg()
                        if epsg:
                            crs_str = f"EPSG:{epsg}"
                        else:
                            crs_str = src.crs.to_string()
                    except Exception:
                        crs_str = str(src.crs)

                # Bounds extraction [min_lon, min_lat, max_lon, max_lat]
                bounds_list: Optional[List[float]] = None
                if src.bounds is not None and crs_str is not None:
                    try:
                        if src.crs.is_geographic:
                            bounds_list = [
                                float(src.bounds.left),
                                float(src.bounds.bottom),
                                float(src.bounds.right),
                                float(src.bounds.top),
                            ]
                        else:
                            # Reproject projected bounds to WGS84 coordinates
                            min_lon, min_lat, max_lon, max_lat = transform_bounds(
                                src.crs, "EPSG:4326", src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
                            )
                            bounds_list = [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]
                    except Exception:
                        # Fallback to native coordinates if reprojection fails
                        bounds_list = [
                            float(src.bounds.left),
                            float(src.bounds.bottom),
                            float(src.bounds.right),
                            float(src.bounds.top),
                        ]

                acquisition_date = _extract_date_rasterio(src, filename)
                modality = _infer_modality_rasterio(src)

                # If GeoTIFF without geospatial tags, ensure format reflects TIFF
                if detected_format == "GeoTIFF" and crs_str is None and bounds_list is None:
                    detected_format = "TIFF"

                return ImageMeta(
                    filename=filename,
                    format=detected_format,
                    modality=modality,
                    width=width,
                    height=height,
                    crs=crs_str,
                    acquisition_date=acquisition_date,
                    bounds=bounds_list,
                )
        except Exception:
            # Degrade gracefully to fallback parser rather than crashing
            pass

    # Fallback to PIL for standard formats or when rasterio fails
    try:
        with Image.open(filepath) as img:
            detected_format = _detect_format(filepath, img.format)
            width, height = img.size
            bands = len(img.getbands())

            if bands >= 3:
                modality = Modality.OPTICAL
            elif bands in (1, 2):
                modality = Modality.UNKNOWN
            else:
                modality = Modality.UNKNOWN

            acquisition_date = _extract_date_from_filename(filename)

            return ImageMeta(
                filename=filename,
                format=detected_format,
                modality=modality,
                width=width,
                height=height,
                crs=None,
                acquisition_date=acquisition_date,
                bounds=None,
            )
    except Exception:
        # Complete fallback for unreadable/corrupted files
        return ImageMeta(
            filename=filename,
            format="UNKNOWN",
            modality=Modality.UNKNOWN,
            width=0,
            height=0,
            crs=None,
            acquisition_date=_extract_date_from_filename(filename),
            bounds=None,
        )


def check_co_registration(
    meta_a: ImageMeta, meta_b: ImageMeta, tolerance: float = 1e-3
) -> Tuple[bool, str]:
    """
    Compare CRS and bounds between two images. Return (True, "") if they
    match within tolerance, else (False, human-readable reason).
    This is a CRS/bounds comparison, NOT pixel-level image registration —
    that is explicitly out of scope.
    """
    # 1. Verify CRS existence
    if not meta_a.crs and not meta_b.crs:
        return (
            False,
            f"Both images ('{meta_a.filename}' and '{meta_b.filename}') lack Coordinate Reference System (CRS) metadata.",
        )
    if not meta_a.crs:
        return (
            False,
            f"Image '{meta_a.filename}' lacks CRS metadata, required for co-registration verification.",
        )
    if not meta_b.crs:
        return (
            False,
            f"Image '{meta_b.filename}' lacks CRS metadata, required for co-registration verification.",
        )

    # 2. Compare CRS
    # Normalize CRS strings (e.g. EPSG:4326)
    crs_a = meta_a.crs.strip().upper()
    crs_b = meta_b.crs.strip().upper()
    if crs_a != crs_b:
        return (
            False,
            f"CRS mismatch: '{meta_a.filename}' is in {meta_a.crs} while '{meta_b.filename}' is in {meta_b.crs}.",
        )

    # 3. Verify bounds existence
    if meta_a.bounds is None or len(meta_a.bounds) != 4:
        return (
            False,
            f"Image '{meta_a.filename}' has missing or incomplete bounding coordinates.",
        )
    if meta_b.bounds is None or len(meta_b.bounds) != 4:
        return (
            False,
            f"Image '{meta_b.filename}' has missing or incomplete bounding coordinates.",
        )

    # 4. Compare spatial bounds within tolerance
    coord_labels = ["min_lon", "min_lat", "max_lon", "max_lat"]
    for i in range(4):
        val_a = meta_a.bounds[i]
        val_b = meta_b.bounds[i]
        diff = abs(val_a - val_b)
        if diff > tolerance:
            return (
                False,
                f"Spatial bounds mismatch on {coord_labels[i]}: '{meta_a.filename}' has {val_a:.6f} whereas "
                f"'{meta_b.filename}' has {val_b:.6f} (difference {diff:.6f} exceeds tolerance {tolerance}).",
            )

    return (True, "")


def validate(filepaths: List[str], query: str) -> ValidationResult:
    """
    Main entry point. Sequence:
      1. Check image count is 1 or 2 — reject otherwise with clear reason
      2. Extract metadata for each image
      3. Check format is supported (GeoTIFF/TIFF always; PNG/JPEG only
         acceptable for benchmark-style inputs)
      4. If 1 image: return is_valid=True, detected_config=SINGLE
      5. If 2 images: determine cross_modal_pair (one optical + one SAR,
         check co-registration) vs bi_temporal_pair (same modality,
         different acquisition_date, check co-registration)
      6. If ambiguous or mismatched, return is_valid=False with a
         specific explanation — never a stack trace, never a silent pass
    """
    # 1. Check image count
    if len(filepaths) == 0:
        return ValidationResult(
            is_valid=False,
            reason="No input images provided. Please provide 1 image (for single-image VQA/captioning) "
            "or 2 images (for bi-temporal change detection or optical-SAR fusion).",
            detected_config=InputConfig.INVALID,
        )

    if len(filepaths) > 2:
        return ValidationResult(
            is_valid=False,
            reason=f"Expected 1 or 2 images, but received {len(filepaths)}. The system supports single-image "
            "analysis, bi-temporal pairs, or optical-SAR pairs.",
            detected_config=InputConfig.INVALID,
        )

    # 2. Extract metadata and verify file existence
    images_meta: List[ImageMeta] = []
    for path in filepaths:
        if not os.path.exists(path):
            return ValidationResult(
                is_valid=False,
                reason=f"Image file not found: '{path}'. Please check that the file path is correct.",
                detected_config=InputConfig.INVALID,
            )
        meta = extract_metadata(path)
        images_meta.append(meta)

    # 3. Check format support
    for meta in images_meta:
        if meta.format not in SUPPORTED_FORMATS:
            return ValidationResult(
                is_valid=False,
                reason=f"File '{meta.filename}' has unsupported format '{meta.format}'. "
                f"Supported formats are {', '.join(sorted(SUPPORTED_FORMATS))}.",
                images_meta=images_meta,
                detected_config=InputConfig.INVALID,
            )

    # 4. Handle 1 image
    if len(images_meta) == 1:
        return ValidationResult(
            is_valid=True,
            reason=None,
            images_meta=images_meta,
            detected_config=InputConfig.SINGLE,
        )

    # 5. Handle 2 images
    meta_a, meta_b = images_meta[0], images_meta[1]

    # Case 5.1: Cross-Modal Pair (One Optical + One SAR)
    is_optical_sar = (meta_a.modality == Modality.OPTICAL and meta_b.modality == Modality.SAR) or (
        meta_a.modality == Modality.SAR and meta_b.modality == Modality.OPTICAL
    )

    if is_optical_sar:
        is_coreg, coreg_msg = check_co_registration(meta_a, meta_b)
        if not is_coreg:
            return ValidationResult(
                is_valid=False,
                reason=f"Optical-SAR cross-modal pair failed co-registration: {coreg_msg}",
                images_meta=images_meta,
                detected_config=InputConfig.INVALID,
            )
        return ValidationResult(
            is_valid=True,
            reason=None,
            images_meta=images_meta,
            detected_config=InputConfig.CROSS_MODAL_PAIR,
        )

    # Case 5.2: Bi-Temporal Pair (Same modality: both optical or both SAR)
    is_same_modality = (
        meta_a.modality == meta_b.modality and meta_a.modality in (Modality.OPTICAL, Modality.SAR)
    )

    if is_same_modality:
        # Check acquisition dates
        if not meta_a.acquisition_date or not meta_b.acquisition_date:
            missing = []
            if not meta_a.acquisition_date:
                missing.append(meta_a.filename)
            if not meta_b.acquisition_date:
                missing.append(meta_b.filename)
            return ValidationResult(
                is_valid=False,
                reason=f"Bi-temporal change detection pair requires acquisition dates for both images. "
                f"Missing acquisition date in: {', '.join(missing)}.",
                images_meta=images_meta,
                detected_config=InputConfig.INVALID,
            )

        if meta_a.acquisition_date == meta_b.acquisition_date:
            return ValidationResult(
                is_valid=False,
                reason=f"Bi-temporal change detection requires images from different acquisition dates, "
                f"but both '{meta_a.filename}' and '{meta_b.filename}' have identical date '{meta_a.acquisition_date}'.",
                images_meta=images_meta,
                detected_config=InputConfig.INVALID,
            )

        # Check co-registration
        is_coreg, coreg_msg = check_co_registration(meta_a, meta_b)
        if not is_coreg:
            return ValidationResult(
                is_valid=False,
                reason=f"Bi-temporal pair failed co-registration: {coreg_msg}",
                images_meta=images_meta,
                detected_config=InputConfig.INVALID,
            )

        return ValidationResult(
            is_valid=True,
            reason=None,
            images_meta=images_meta,
            detected_config=InputConfig.BI_TEMPORAL_PAIR,
        )

    # Case 5.3: Ambiguous or invalid modality combination
    return ValidationResult(
        is_valid=False,
        reason=f"Cannot determine valid pair configuration. Detected modalities: '{meta_a.filename}' is "
        f"{meta_a.modality.value}, '{meta_b.filename}' is {meta_b.modality.value}. Two-image queries must be either "
        f"a cross-modal pair (one optical, one SAR) or a bi-temporal pair (same modality with distinct acquisition dates).",
        images_meta=images_meta,
        detected_config=InputConfig.INVALID,
    )
