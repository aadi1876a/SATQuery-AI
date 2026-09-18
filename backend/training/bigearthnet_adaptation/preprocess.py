"""BigEarthNet Preprocessing Pipeline for SatQuery AI (P1 -> P2 handoff).

Prepares a clean, filtered, normalized subset of BigEarthNet (Sentinel-1 SAR + Sentinel-2 Optical)
with paired natural-language text annotations formatted as JSONL for LoRA fine-tuning of VLMs.
Supports official BigEarthNet-MM metadata.parquet as well as raw directory structures.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False

try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:  # pragma: no cover
    HAS_PYARROW = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False


def load_bigearthnet_index(metadata_path: str) -> Any:
    """
    Load the official BigEarthNet metadata index.
    Columns: patch_id, labels, split, country, s1_name, s2v1_name,
             contains_seasonal_snow, contains_cloud_or_shadow
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for load_bigearthnet_index. Please install pandas.")
    return pd.read_parquet(metadata_path)


def filter_clean_patches(df: Any) -> Any:
    """
    Remove patches unsuitable for fine-tuning:
    - Drop rows where contains_cloud_or_shadow is True (corrupts optical signal)
    - Drop rows where contains_seasonal_snow is True (skews land-cover appearance)
    """
    mask = pd.Series(True, index=df.index)
    if "contains_cloud_or_shadow" in df.columns:
        mask = mask & (~df["contains_cloud_or_shadow"].fillna(False))
    if "contains_seasonal_snow" in df.columns:
        mask = mask & (~df["contains_seasonal_snow"].fillna(False))
    return df[mask]


def get_split(df: Any, split: str, subset_size: int = 3000) -> Any:
    """
    Use the dataset's OWN 'split' column (train/validation/test) rather than
    re-splitting randomly — BigEarthNet's split is curated to avoid geographic
    leakage between train and test. Sample down to subset_size for a
    hackathon-feasible fine-tuning run.
    """
    if "split" not in df.columns:
        return df.sample(n=min(subset_size, len(df)), random_state=42)
    subset = df[df["split"] == split]
    return subset.sample(n=min(subset_size, len(subset)), random_state=42)


def build_image_text_pairs(df: Any, s1_dir: str, s2_dir: str, output_path: str) -> None:
    """
    For each row, resolve the actual S1/SAR and S2/optical file paths using
    s1_name and s2v1_name, and write out image-text pairs for P2's LoRA
    fine-tuning script. Text is derived from the 'labels' list, e.g.:
        "Satellite image showing: arable land, mixed forest, pastures."
    Output format: JSONL, one record per patch:
        {"patch_id": ..., "s1_path": ..., "s2_path": ..., "text": ..., "labels": [...]}
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    s1_path_obj = Path(s1_dir) if s1_dir and os.path.exists(s1_dir) else None
    s2_path_obj = Path(s2_dir) if s2_dir and os.path.exists(s2_dir) else None

    records = []
    for _, row in df.iterrows():
        patch_id = str(row.get("patch_id", ""))
        labels = row.get("labels", [])
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        elif not isinstance(labels, list):
            labels = [str(labels)] if labels else []

        s1_name = str(row.get("s1_name", "")) if pd.notna(row.get("s1_name")) else None
        s2v1_name = str(row.get("s2v1_name", "")) if pd.notna(row.get("s2v1_name")) else None

        s2_path = None
        if s2_path_obj and s2v1_name:
            candidate = s2_path_obj / s2v1_name
            if candidate.exists():
                s2_path = str(candidate)
            else:
                matches = list(s2_path_obj.glob(f"**/{s2v1_name}"))
                if matches:
                    s2_path = str(matches[0])

        s1_path = None
        if s1_path_obj and s1_name:
            candidate = s1_path_obj / s1_name
            if candidate.exists():
                s1_path = str(candidate)
            else:
                matches = list(s1_path_obj.glob(f"**/{s1_name}"))
                if matches:
                    s1_path = str(matches[0])

        if labels:
            text = f"Satellite image showing: {', '.join(labels)}."
        else:
            text = "Satellite remote sensing image showing terrain."

        records.append(
            {
                "patch_id": patch_id,
                "s1_path": s1_path,
                "s2_path": s2_path,
                "text": text,
                "labels": labels,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _read_s2_rgb(path: str) -> Optional[np.ndarray]:
    """Read a Sentinel-2 GeoTIFF and extract/composite RGB channels (H, W, 3)."""
    if not os.path.exists(path):
        return None

    if HAS_RASTERIO and (path.endswith(".tif") or path.endswith(".tiff")):
        try:
            with rasterio.open(path) as src:
                count = src.count
                if count >= 10:
                    # Standard 10-band or 12-band BigEarthNet S2 patch
                    # Band 1: B02 (Blue), Band 2: B03 (Green), Band 3: B04 (Red)
                    b_blue = src.read(1)
                    b_green = src.read(2)
                    b_red = src.read(3)
                    return np.stack([b_red, b_green, b_blue], axis=-1)
                elif count >= 3:
                    # Generic RGB or multispectral
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                    return np.stack([r, g, b], axis=-1)
                elif count == 1:
                    arr = src.read(1)
                    return arr
        except Exception:
            pass

    try:
        with Image.open(path) as img:
            return np.array(img)
    except Exception:
        return None


def _read_s1_sar(path: str) -> Optional[np.ndarray]:
    """Read a Sentinel-1 SAR GeoTIFF (H, W, C) with VV and VH polarizations."""
    if not os.path.exists(path):
        return None

    if HAS_RASTERIO and (path.endswith(".tif") or path.endswith(".tiff")):
        try:
            with rasterio.open(path) as src:
                arr = src.read()
                if arr.shape[0] == 1:
                    return arr[0]
                return np.transpose(arr, (1, 2, 0))
        except Exception:
            pass

    try:
        with Image.open(path) as img:
            return np.array(img)
    except Exception:
        return None


def _normalize_and_resize(
    img_arr: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    is_sar: bool = False,
) -> Image.Image:
    """Normalize array values to 8-bit [0, 255] and resize to target resolution."""
    if is_sar:
        # Clip SAR backscatter (typically dB [-35, 0] or amplitude) and scale to 0-255
        valid = img_arr[np.isfinite(img_arr)]
        if len(valid) > 0:
            p2 = np.percentile(valid, 2)
            p98 = np.percentile(valid, 98)
            clipped = np.clip(img_arr, p2, p98)
            norm = (
                (clipped - p2) / (p98 - p2 + 1e-6) * 255.0
                if p98 > p2
                else np.zeros_like(img_arr)
            )
            uint8_arr = np.uint8(norm)
        else:
            uint8_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)
    else:
        # Optical reflectance: Sentinel-2 surface reflectance (typically 0-10000)
        if img_arr.dtype == np.uint8:
            uint8_arr = img_arr
        else:
            valid = img_arr[np.isfinite(img_arr)]
            if len(valid) > 0:
                p98 = np.percentile(valid, 98)
                if p98 > 255:
                    scale = 255.0 / max(p98, 1000.0)
                    uint8_arr = np.uint8(np.clip(img_arr * scale, 0, 255))
                else:
                    uint8_arr = np.uint8(np.clip(img_arr, 0, 255))
            else:
                uint8_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)

    # Convert to PIL Image
    if uint8_arr.ndim == 2:
        pil_img = Image.fromarray(uint8_arr, mode="L")
    elif uint8_arr.ndim == 3:
        if uint8_arr.shape[2] == 1:
            pil_img = Image.fromarray(uint8_arr[:, :, 0], mode="L")
        elif uint8_arr.shape[2] >= 3:
            pil_img = Image.fromarray(uint8_arr[:, :, :3], mode="RGB")
        else:
            # 2-channel (VV, VH) -> compose false-color RGB: (VV, VH, (VV+VH)/2)
            ch1 = uint8_arr[:, :, 0]
            ch2 = uint8_arr[:, :, 1]
            ch3 = np.uint8(np.clip((ch1.astype(float) + ch2.astype(float)) / 2.0, 0, 255))
            pil_img = Image.fromarray(np.stack([ch1, ch2, ch3], axis=-1), mode="RGB")
    else:
        pil_img = Image.new("RGB", target_size, color="black")

    # Resize to target resolution for VLM backbone
    return pil_img.resize(target_size, Image.Resampling.BILINEAR)


def _build_caption_from_labels(labels: List[str], country: str = "", patch_name: str = "") -> str:
    """Generate an evidence-grounded natural-language caption from BigEarthNet class labels."""
    if not labels:
        if country:
            return f"A remote sensing satellite observation over {country} showing landscape terrain."
        return "A remote sensing satellite patch showing ground landscape and land cover."

    joined_labels = ", ".join(labels)
    country_suffix = f" in {country}" if country else ""
    templates = [
        f"A satellite observation showing terrain characterized by {joined_labels}{country_suffix}.",
        f"High-resolution remote sensing patch capturing land cover consisting of {joined_labels}{country_suffix}.",
        f"Aerial satellite imagery depicting {joined_labels}{country_suffix}.",
    ]
    idx = abs(hash(patch_name)) % len(templates)
    return templates[idx]


def discover_patches(raw_data_dir: str) -> List[Dict[str, Any]]:
    """
    Discover optical patches, SAR patches, and metadata/text in raw_data_dir.
    Supports official BigEarthNet-MM metadata.parquet as well as raw directory structures.
    """
    raw_path = Path(raw_data_dir)
    discovered: List[Dict[str, Any]] = []

    # 1. Check for metadata.parquet (BigEarthNet-MM parquet schema)
    parquet_files = list(raw_path.glob("**/metadata.parquet"))
    if parquet_files and HAS_PYARROW:
        pq_path = parquet_files[0]
        base_dir = pq_path.parent
        table = pq.read_table(
            str(pq_path),
            columns=["patch_id", "labels", "split", "country", "s1_name", "s2v1_name", "contains_cloud_or_shadow"],
        )

        # Index available on-disk files for fast lookup
        s2_disk_map: Dict[str, str] = {}
        s1_disk_map: Dict[str, str] = {}

        # Scan for S2 and S1 files on disk (supports both single-file multi-band GeoTIFFs and band-separated folders)
        for path in base_dir.rglob("*.tif"):
            stem = path.stem
            folder_name = path.parent.name
            if "S2" in stem or "BigEarthNet-S2" in str(path):
                s2_disk_map[stem] = str(path)
                s2_disk_map[folder_name] = str(path.parent)
            else:
                s1_disk_map[stem] = str(path)
                s1_disk_map[folder_name] = str(path.parent)

        for row in table.to_pylist():
            patch_id = row["patch_id"]
            if patch_id in s2_disk_map:
                s2_file = s2_disk_map[patch_id]
                s1_name = row.get("s1_name")
                s1_file = s1_disk_map.get(s1_name) if s1_name else None
                labels = row.get("labels", [])
                country = row.get("country", "")
                split = row.get("split", "train")
                cloud = row.get("contains_cloud_or_shadow", False)

                optical_path = None
                if os.path.isfile(s2_file):
                    optical_path = s2_file
                else:
                    b4 = list(Path(s2_file).glob("*_B04.tif")) or list(Path(s2_file).glob("*_B4.tif"))
                    b3 = list(Path(s2_file).glob("*_B03.tif")) or list(Path(s2_file).glob("*_B3.tif"))
                    b2 = list(Path(s2_file).glob("*_B02.tif")) or list(Path(s2_file).glob("*_B2.tif"))
                    if b4 and b3 and b2:
                        optical_path = (str(b4[0]), str(b3[0]), str(b2[0]))

                sar_path = None
                if s1_file:
                    if os.path.isfile(s1_file):
                        sar_path = s1_file
                    else:
                        s1_tifs = list(Path(s1_file).glob("*.tif"))
                        if s1_tifs:
                            sar_path = str(s1_tifs[0])

                if optical_path:
                    discovered.append(
                        {
                            "patch_id": patch_id,
                            "optical": optical_path,
                            "sar": sar_path,
                            "labels": labels,
                            "country": country,
                            "split": split,
                            "contains_cloud": cloud,
                            "caption": _build_caption_from_labels(labels, country, patch_id),
                        }
                    )

        if discovered:
            return discovered

    # 2. Check for BigEarthNet labels metadata JSON files
    label_files = list(raw_path.glob("**/*labels_metadata.json")) + list(
        raw_path.glob("**/*labels.json")
    )

    if label_files:
        for lbl_path in label_files:
            patch_dir = lbl_path.parent
            patch_name = patch_dir.name
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                labels = meta.get("labels", [])
            except Exception:
                labels = []

            # Check for optical RGB bands (B04, B03, B02) or composite
            b4 = list(patch_dir.glob("*_B04.tif")) or list(patch_dir.glob("*_B4.tif"))
            b3 = list(patch_dir.glob("*_B03.tif")) or list(patch_dir.glob("*_B3.tif"))
            b2 = list(patch_dir.glob("*_B02.tif")) or list(patch_dir.glob("*_B2.tif"))
            composite = (
                list(patch_dir.glob("*_RGB.tif"))
                or list(patch_dir.glob("*_composite.tif"))
                or list(patch_dir.glob("*.tif"))
            )

            optical_path = None
            if b4 and b3 and b2:
                optical_path = (str(b4[0]), str(b3[0]), str(b2[0]))
            elif composite:
                optical_path = str(composite[0])

            sar_path = None
            sar_candidates = list(raw_path.glob(f"**/*S1*/*{patch_name.replace('S2_', 'S1_')}*"))
            if sar_candidates:
                sar_tifs = list(sar_candidates[0].glob("*.tif"))
                if sar_tifs:
                    sar_path = str(sar_tifs[0])

            if optical_path:
                discovered.append(
                    {
                        "patch_id": patch_name,
                        "optical": optical_path,
                        "sar": sar_path,
                        "labels": labels,
                        "caption": _build_caption_from_labels(labels, "", patch_name),
                    }
                )

    # 3. Fallback: discover any images directly
    if not discovered:
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
            for img_path in raw_path.glob(f"**/{ext}"):
                p_name = img_path.stem
                discovered.append(
                    {
                        "patch_id": p_name,
                        "optical": str(img_path),
                        "sar": None,
                        "labels": ["Satellite Land Cover"],
                        "caption": f"Satellite remote sensing image showing land cover ({p_name}).",
                    }
                )

    return discovered


def preprocess_bigearthnet(
    raw_data_dir: str,
    output_dir: str,
    subset_size: int = 3000,
    target_size: Tuple[int, int] = (224, 224),
    val_split: float = 0.2,
    filter_clouds: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    1. Load BigEarthNet raw data (co-registered Sentinel-1 SAR + Sentinel-2 optical patches + text annotations)
    2. Filter to a manageable subset (a few thousand patches is enough for LoRA fine-tuning)
    3. Resize/normalize patches to the input resolution P2's chosen VLM expects (configurable)
    4. Pair each image with its text annotation in a clean, documented JSONL format
    5. Write output to output_dir in a structure P2's fine-tuning script can load directly
    """
    random.seed(seed)
    np.random.seed(seed)

    raw_path = Path(raw_data_dir)
    out_path = Path(output_dir)
    images_out = out_path / "images"
    sar_out = out_path / "sar"

    images_out.mkdir(parents=True, exist_ok=True)
    sar_out.mkdir(parents=True, exist_ok=True)

    # Discover raw patches
    patches = discover_patches(str(raw_path))
    if not patches:
        raise ValueError(
            f"No valid BigEarthNet patches or images found in '{raw_data_dir}'. "
            f"Please verify raw dataset directory path."
        )

    # Optional quality filter: filter out heavy cloud or shadow if flag is present
    if filter_clouds and any("contains_cloud" in p for p in patches):
        cloud_free = [p for p in patches if not p.get("contains_cloud", False)]
        if len(cloud_free) >= subset_size:
            patches = cloud_free

    # Filter to subset size
    if len(patches) > subset_size:
        selected_patches = random.sample(patches, subset_size)
    else:
        selected_patches = patches

    processed_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(selected_patches):
        patch_id = item.get("patch_id", f"patch_{idx:06d}")
        optical_src = item.get("optical")
        sar_src = item.get("sar")
        labels = item.get("labels", [])
        country = item.get("country", "")
        split = item.get("split", "train")
        caption = item.get("caption") or _build_caption_from_labels(labels, country, patch_id)

        # Process optical image
        optical_rel_path = None
        if optical_src:
            if isinstance(optical_src, tuple) and len(optical_src) == 3:
                r = _read_s2_rgb(optical_src[0])
                g = _read_s2_rgb(optical_src[1])
                b = _read_s2_rgb(optical_src[2])
                if r is not None and g is not None and b is not None:
                    stacked = np.stack([r, g, b], axis=-1)
                    pil_img = _normalize_and_resize(stacked, target_size=target_size, is_sar=False)
                else:
                    continue
            else:
                arr = _read_s2_rgb(str(optical_src))
                if arr is not None:
                    pil_img = _normalize_and_resize(arr, target_size=target_size, is_sar=False)
                else:
                    continue

            filename = f"{patch_id}.png"
            dest = images_out / filename
            pil_img.save(dest, format="PNG", optimize=True)
            optical_rel_path = f"images/{filename}"

        # Process SAR image if available
        sar_rel_path = None
        if sar_src:
            sar_arr = _read_s1_sar(str(sar_src))
            if sar_arr is not None:
                sar_pil = _normalize_and_resize(sar_arr, target_size=target_size, is_sar=True)
                sar_fn = f"{patch_id}_sar.png"
                sar_dest = sar_out / sar_fn
                sar_pil.save(sar_dest, format="PNG", optimize=True)
                sar_rel_path = f"sar/{sar_fn}"

        record = {
            "id": patch_id,
            "image_path": optical_rel_path,
            "sar_path": sar_rel_path,
            "text": caption,
            "labels": labels,
            "country": country,
            "split": split,
        }
        processed_records.append(record)

    if not processed_records:
        raise RuntimeError("Failed to process any patches from the raw directory.")

    # Train / Val Split (strict adherence to BigEarthNet dataset split)
    train_records = [r for r in processed_records if r.get("split") == "train"]
    val_records = [r for r in processed_records if r.get("split") in ("validation", "test")]
    
    if not train_records and not val_records:
        raise ValueError("Missing or invalid 'split' in metadata. Manual splitting is prohibited.")

    # Write train.jsonl, val.jsonl, and dataset.jsonl
    train_jsonl = out_path / "train.jsonl"
    val_jsonl = out_path / "val.jsonl"
    full_jsonl = out_path / "dataset.jsonl"

    for path, data in [(train_jsonl, train_records), (val_jsonl, val_records), (full_jsonl, processed_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write dataset_info.json (Documentation for P2's fine-tuning script)
    info = {
        "dataset_name": "BigEarthNet-LoRA-Subset",
        "total_samples": len(processed_records),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "target_resolution": list(target_size),
        "image_format": "PNG",
        "channels": 3,
        "optical_dir": "images/",
        "sar_dir": "sar/" if any(r["sar_path"] for r in processed_records) else None,
        "jsonl_files": {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "all": "dataset.jsonl",
        },
        "schema_description": {
            "id": "Unique BigEarthNet patch identifier",
            "image_path": "Relative path to normalized optical RGB image",
            "sar_path": "Relative path to normalized SAR dual-pol image (or null)",
            "text": "Evidence-grounded caption for VLM text instruction / fine-tuning",
            "labels": "List of CORINE Land Cover categorical labels",
            "country": "Geographic country where patch was observed",
            "split": "Original dataset split (train/validation/test)",
        },
    }

    with open(out_path / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return info


def generate_mock_bigearthnet_data(
    raw_data_dir: str, num_patches: int = 10, patch_size: Tuple[int, int] = (120, 120)
) -> None:
    """Generate synthetic BigEarthNet patch structure for offline testing and verification."""
    os.makedirs(raw_data_dir, exist_ok=True)
    sample_labels = [
        ["Continuous urban fabric", "Industrial or commercial units"],
        ["Non-irrigated arable land", "Pastures"],
        ["Broad-leaved forest", "Mixed forest"],
        ["Coniferous forest", "Transitional woodland, shrub"],
        ["Water bodies", "Inland marshes"],
    ]

    for i in range(num_patches):
        p_name = f"S2A_MSIL2A_20230512T101031_patch_{i:04d}"
        p_dir = os.path.join(raw_data_dir, p_name)
        os.makedirs(p_dir, exist_ok=True)

        lbl = sample_labels[i % len(sample_labels)]
        with open(os.path.join(p_dir, f"{p_name}_labels_metadata.json"), "w") as f:
            json.dump({"labels": lbl, "split": "train" if i < int(num_patches*0.75) else "validation", "acquisition_time": "2023-05-12T10:10:31Z"}, f)

        # RGB Bands
        for band in ("B04", "B03", "B02"):
            band_arr = np.random.randint(200, 3000, size=patch_size, dtype=np.uint16)
            if HAS_RASTERIO:
                with rasterio.open(
                    os.path.join(p_dir, f"{p_name}_{band}.tif"),
                    "w",
                    driver="GTiff",
                    height=patch_size[0],
                    width=patch_size[1],
                    count=1,
                    dtype=np.uint16,
                ) as dst:
                    dst.write(band_arr, 1)
            else:
                img = Image.fromarray(np.uint8(band_arr / 3000 * 255))
                img.save(os.path.join(p_dir, f"{p_name}_{band}.tif"))

    if HAS_PYARROW:
        import pyarrow as pa
        import pyarrow.parquet as pq
        arrays = [
            pa.array([f"S2A_MSIL2A_20230512T101031_patch_{i:04d}" for i in range(num_patches)], type=pa.string()),
            pa.array([[str(l) for l in sample_labels[i % len(sample_labels)]] for i in range(num_patches)], type=pa.list_(pa.string())),
            pa.array(["train" if i < int(num_patches*0.75) else "validation" for i in range(num_patches)], type=pa.string()),
            pa.array(["Austria"] * num_patches, type=pa.string()),
            pa.array([None] * num_patches, type=pa.string()),
            pa.array([f"S2A_MSIL2A_20230512T101031_patch_{i:04d}"] * num_patches, type=pa.string()),
            pa.array([False] * num_patches, type=pa.bool_())
        ]
        names = ["patch_id", "labels", "split", "country", "s1_name", "s2v1_name", "contains_cloud_or_shadow"]
        table = pa.Table.from_arrays(arrays, names=names)
        pq.write_table(table, os.path.join(raw_data_dir, "metadata.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BigEarthNet for LoRA fine-tuning (P1 -> P2)")
    parser.add_argument("--raw_data_dir", "--raw-dir", type=str, required=True, help="Path to raw BigEarthNet directory")
    parser.add_argument("--output_dir", "--output-dir", type=str, required=True, help="Destination directory for processed subset")
    parser.add_argument("--subset_size", "--subset-size", type=int, default=3000, help="Number of patches to sample")
    parser.add_argument("--resolution", type=int, default=224, help="Target image resolution (square: W=H)")
    parser.add_argument("--filter_clouds", "--filter-clouds", action="store_true", default=True, help="Filter out cloud-contaminated patches")
    parser.add_argument("--generate_mock", action="store_true", help="Generate mock BigEarthNet data first")

    args = parser.parse_args()

    if args.generate_mock:
        print(f"Generating mock data in {args.raw_data_dir}...")
        generate_mock_bigearthnet_data(args.raw_data_dir, num_patches=20)

    print(f"Preprocessing BigEarthNet from {args.raw_data_dir} to {args.output_dir}...")
    summary = preprocess_bigearthnet(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        target_size=(args.resolution, args.resolution),
    )
    print("Preprocess completed successfully:")
    print(json.dumps(summary, indent=2))
