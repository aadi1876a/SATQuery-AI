"""Comprehensive Test Suite for SatQuery AI Input Validator and Geospatial Extractor.

Tests all P1 deliverables:
- Schema contracts compliance
- extract_metadata() functionality & edge cases
- check_co_registration() tolerance and CRS comparison
- validate() classification (SINGLE, CROSS_MODAL_PAIR, BI_TEMPORAL_PAIR, INVALID)
- Error handling and human-readable rejection messages
- BigEarthNet preprocessing pipeline
"""

import json
import os
import shutil
import tempfile
import pytest

from backend.app.schemas.contracts import (
    ImageMeta,
    InputConfig,
    Modality,
    ValidationResult,
)
from backend.app.validators.input_validator import (
    check_co_registration,
    extract_metadata,
    validate,
)
from backend.tests.generate_sample_data import generate_all_samples
from backend.training.bigearthnet_adaptation.preprocess import (
    generate_mock_bigearthnet_data,
    preprocess_bigearthnet,
)


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory):
    """Generate session-scoped synthetic sample data."""
    data_dir = tmp_path_factory.mktemp("sample_data")
    generate_all_samples(str(data_dir))
    return str(data_dir)


# =========================================================================
# 1. Schema & Contract Tests
# =========================================================================
def test_schema_contract_instantiation():
    """Verify ImageMeta and ValidationResult strictly adhere to contract."""
    meta = ImageMeta(
        filename="test.tif",
        format="GeoTIFF",
        modality=Modality.OPTICAL,
        width=256,
        height=256,
        crs="EPSG:4326",
        acquisition_date="2024-01-15",
        bounds=[72.85, 19.05, 72.95, 19.15],
    )
    assert meta.filename == "test.tif"
    assert meta.modality == Modality.OPTICAL
    assert meta.crs == "EPSG:4326"

    res = ValidationResult(
        is_valid=True,
        reason=None,
        images_meta=[meta],
        detected_config=InputConfig.SINGLE,
    )
    assert res.is_valid is True
    assert res.reason is None
    assert len(res.images_meta) == 1
    assert res.detected_config == InputConfig.SINGLE


# =========================================================================
# 2. Metadata Extraction Tests
# =========================================================================
def test_extract_metadata_optical(sample_data_dir):
    """Extract metadata from standard multispectral optical GeoTIFF."""
    path = os.path.join(sample_data_dir, "optical_t1_20240115.tif")
    meta = extract_metadata(path)

    assert meta.filename == "optical_t1_20240115.tif"
    assert meta.format == "GeoTIFF"
    assert meta.modality == Modality.OPTICAL
    assert meta.width == 128
    assert meta.height == 128
    assert meta.crs == "EPSG:4326"
    assert meta.acquisition_date is not None
    assert "2024-01-15" in meta.acquisition_date
    assert meta.bounds is not None
    assert len(meta.bounds) == 4
    assert pytest.approx(meta.bounds[0], abs=1e-3) == 72.85
    assert pytest.approx(meta.bounds[1], abs=1e-3) == 19.05


def test_extract_metadata_sar(sample_data_dir):
    """Extract metadata from dual-pol SAR GeoTIFF."""
    path = os.path.join(sample_data_dir, "sar_s1_20240115.tif")
    meta = extract_metadata(path)

    assert meta.filename == "sar_s1_20240115.tif"
    assert meta.format == "GeoTIFF"
    assert meta.modality == Modality.SAR
    assert meta.crs == "EPSG:4326"
    assert meta.bounds is not None


def test_extract_metadata_missing_crs(sample_data_dir):
    """Degrade gracefully on GeoTIFF missing CRS metadata."""
    path = os.path.join(sample_data_dir, "optical_missing_crs.tif")
    meta = extract_metadata(path)

    assert meta.filename == "optical_missing_crs.tif"
    # Never raise, degrades to crs=None
    assert meta.crs is None
    assert meta.bounds is None


def test_extract_metadata_benchmark_png(sample_data_dir):
    """Extract metadata from benchmark PNG image."""
    path = os.path.join(sample_data_dir, "benchmark_optical_20240201.png")
    meta = extract_metadata(path)

    assert meta.format == "PNG"
    assert meta.modality == Modality.OPTICAL
    assert meta.width == 128
    assert meta.height == 128
    assert meta.crs is None
    assert meta.bounds is None
    assert meta.acquisition_date == "2024-02-01"


def test_extract_metadata_corrupted_file(sample_data_dir):
    """Corrupted or truncated file must never crash the pipeline."""
    path = os.path.join(sample_data_dir, "corrupt_truncated.tif")
    meta = extract_metadata(path)

    assert meta.filename == "corrupt_truncated.tif"
    assert meta.format in ("UNKNOWN", "TIFF", "GeoTIFF")
    assert meta.crs is None


def test_extract_metadata_nonexistent_file():
    """Nonexistent file path must not raise."""
    meta = extract_metadata("/tmp/non_existent_image_12345.tif")
    assert meta.filename == "non_existent_image_12345.tif"
    assert meta.format == "UNKNOWN"
    assert meta.modality == Modality.UNKNOWN


# =========================================================================
# 3. Co-registration Tests
# =========================================================================
def test_check_co_registration_success(sample_data_dir):
    """Identical CRS and bounds within tolerance pass co-registration."""
    meta_a = extract_metadata(os.path.join(sample_data_dir, "optical_t1_20240115.tif"))
    meta_b = extract_metadata(os.path.join(sample_data_dir, "optical_t2_20240615.tif"))

    ok, reason = check_co_registration(meta_a, meta_b, tolerance=1e-3)
    assert ok is True
    assert reason == ""


def test_check_co_registration_mismatched_bounds(sample_data_dir):
    """Mismatched spatial bounds fail with a clear human-readable reason."""
    meta_a = extract_metadata(os.path.join(sample_data_dir, "optical_t1_20240115.tif"))
    meta_b = extract_metadata(os.path.join(sample_data_dir, "optical_mismatched_bounds.tif"))

    ok, reason = check_co_registration(meta_a, meta_b, tolerance=1e-3)
    assert ok is False
    assert "Spatial bounds mismatch" in reason
    assert "tolerance" in reason


def test_check_co_registration_mismatched_crs(sample_data_dir):
    """Mismatched CRS fails with clear explanation."""
    meta_a = extract_metadata(os.path.join(sample_data_dir, "optical_t1_20240115.tif"))
    meta_b = extract_metadata(os.path.join(sample_data_dir, "optical_utm_crs.tif"))

    ok, reason = check_co_registration(meta_a, meta_b)
    assert ok is False
    assert "CRS mismatch" in reason


def test_check_co_registration_missing_crs(sample_data_dir):
    """Missing CRS fails with clear explanation."""
    meta_a = extract_metadata(os.path.join(sample_data_dir, "optical_t1_20240115.tif"))
    meta_b = extract_metadata(os.path.join(sample_data_dir, "optical_missing_crs.tif"))

    ok, reason = check_co_registration(meta_a, meta_b)
    assert ok is False
    assert "lacks CRS metadata" in reason or "lack Coordinate Reference System" in reason


# =========================================================================
# 4. Pipeline Validation (validate()) Tests
# =========================================================================
def test_validate_single_image(sample_data_dir):
    """Single image classifies as InputConfig.SINGLE."""
    path = os.path.join(sample_data_dir, "optical_t1_20240115.tif")
    res = validate([path], query="What type of land cover is visible?")

    assert res.is_valid is True
    assert res.reason is None
    assert len(res.images_meta) == 1
    assert res.detected_config == InputConfig.SINGLE


def test_validate_cross_modal_pair(sample_data_dir):
    """Optical + SAR co-registered pair classifies as InputConfig.CROSS_MODAL_PAIR."""
    opt_path = os.path.join(sample_data_dir, "optical_t1_20240115.tif")
    sar_path = os.path.join(sample_data_dir, "sar_s1_20240115.tif")
    res = validate([opt_path, sar_path], query="Fuse optical and SAR to identify flooded regions.")

    assert res.is_valid is True
    assert res.reason is None
    assert len(res.images_meta) == 2
    assert res.detected_config == InputConfig.CROSS_MODAL_PAIR


def test_validate_bi_temporal_pair(sample_data_dir):
    """Optical T1 + Optical T2 with distinct dates classifies as InputConfig.BI_TEMPORAL_PAIR."""
    t1_path = os.path.join(sample_data_dir, "optical_t1_20240115.tif")
    t2_path = os.path.join(sample_data_dir, "optical_t2_20240615.tif")
    res = validate([t1_path, t2_path], query="What changes occurred between January and June?")

    assert res.is_valid is True
    assert res.reason is None
    assert len(res.images_meta) == 2
    assert res.detected_config == InputConfig.BI_TEMPORAL_PAIR


def test_validate_reject_zero_images():
    """Empty image list is rejected with clear reason."""
    res = validate([], query="Find rivers.")
    assert res.is_valid is False
    assert res.reason is not None
    assert "No input images provided" in res.reason
    assert res.detected_config == InputConfig.INVALID


def test_validate_reject_more_than_two_images(sample_data_dir):
    """More than 2 images are rejected with clear reason."""
    t1 = os.path.join(sample_data_dir, "optical_t1_20240115.tif")
    t2 = os.path.join(sample_data_dir, "optical_t2_20240615.tif")
    s1 = os.path.join(sample_data_dir, "sar_s1_20240115.tif")
    res = validate([t1, t2, s1], query="Analyze all three.")

    assert res.is_valid is False
    assert "Expected 1 or 2 images, but received 3" in res.reason
    assert res.detected_config == InputConfig.INVALID


def test_validate_reject_nonexistent_file():
    """Non-existent file path rejected with human-readable reason."""
    res = validate(["/tmp/does_not_exist_satquery.tif"], query="Is there a building?")
    assert res.is_valid is False
    assert "Image file not found" in res.reason
    assert res.detected_config == InputConfig.INVALID


def test_validate_reject_unsupported_format(sample_data_dir):
    """Unsupported format (.txt) rejected with human-readable reason."""
    path = os.path.join(sample_data_dir, "invalid_format.txt")
    res = validate([path], query="Analyze text file.")
    assert res.is_valid is False
    assert "unsupported format" in res.reason.lower()
    assert res.detected_config == InputConfig.INVALID


def test_validate_reject_bi_temporal_same_date(sample_data_dir):
    """Same-modality pair with identical dates rejected with clear reason."""
    t1 = os.path.join(sample_data_dir, "optical_t1_20240115.tif")
    res = validate([t1, t1], query="Detect change.")
    assert res.is_valid is False
    assert "different acquisition dates" in res.reason
    assert res.detected_config == InputConfig.INVALID


def test_validate_reject_non_co_registered_pair(sample_data_dir):
    """Pair with mismatched bounds rejected with detailed co-registration reason."""
    t1 = os.path.join(sample_data_dir, "optical_t1_20240115.tif")
    mismatched = os.path.join(sample_data_dir, "optical_mismatched_bounds.tif")
    res = validate([t1, mismatched], query="Detect change.")

    assert res.is_valid is False
    assert "co-registration" in res.reason.lower()
    assert res.detected_config == InputConfig.INVALID


# =========================================================================
# 5. BigEarthNet Preprocessing Pipeline Tests
# =========================================================================
def test_bigearthnet_preprocessing_pipeline(tmp_path):
    """Test full BigEarthNet preprocessing pipeline from mock data to output directory."""
    raw_dir = tmp_path / "raw_bigearthnet"
    out_dir = tmp_path / "processed_subset"

    # 1. Generate synthetic raw data
    generate_mock_bigearthnet_data(str(raw_dir), num_patches=6, patch_size=(64, 64))

    # 2. Run preprocessing
    info = preprocess_bigearthnet(
        raw_data_dir=str(raw_dir),
        output_dir=str(out_dir),
        subset_size=4,
        target_size=(128, 128),
        val_split=0.25,
    )

    # 3. Verify outputs
    assert info["total_samples"] == 4
    assert info["train_samples"] + info["val_samples"] == 4
    assert info["train_samples"] > 0
    assert info["val_samples"] > 0
    assert (out_dir / "images").exists()
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "val.jsonl").exists()
    assert (out_dir / "dataset_info.json").exists()

    # 4. Check JSONL content structure
    with open(out_dir / "train.jsonl", "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
        assert len(lines) == info["train_samples"]
        first = lines[0]
        assert "id" in first
        assert "image_path" in first
        assert "text" in first
        # Verify optical image exists on disk
        img_full_path = out_dir / first["image_path"]
        assert img_full_path.exists()


def test_bigearthnet_modular_functions(tmp_path):
    """Test load_bigearthnet_index, filter_clean_patches, get_split, and build_image_text_pairs."""
    from backend.training.bigearthnet_adaptation.preprocess import (
        load_bigearthnet_index,
        filter_clean_patches,
        get_split,
        build_image_text_pairs,
    )

    raw_dir = tmp_path / "raw_parquet_test"
    generate_mock_bigearthnet_data(str(raw_dir), num_patches=6, patch_size=(32, 32))
    pq_path = raw_dir / "metadata.parquet"

    # Test load
    df = load_bigearthnet_index(str(pq_path))
    assert len(df) == 6
    assert "patch_id" in df.columns
    assert "labels" in df.columns
    assert "split" in df.columns

    # Test filter
    clean_df = filter_clean_patches(df)
    assert len(clean_df) == 6

    # Test split
    train_df = get_split(clean_df, split="train", subset_size=2)
    assert len(train_df) == 2
    assert all(train_df["split"] == "train")

    # Test build_image_text_pairs
    out_jsonl = tmp_path / "output_pairs.jsonl"
    build_image_text_pairs(train_df, s1_dir="", s2_dir=str(raw_dir), output_path=str(out_jsonl))
    assert out_jsonl.exists()

    with open(out_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 2
    assert "patch_id" in records[0]
    assert "text" in records[0]
    assert "labels" in records[0]
    assert isinstance(records[0]["labels"], list)
