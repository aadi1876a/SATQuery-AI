"""
backend/tests/test_p3.py
Pytest Test Suite for SatQuery AI P3 Change Detection Specialist Model.
"""

import os
import sys
import pytest
from PIL import Image, ImageDraw

# Ensure repository root is in sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from schemas import ToolInput, ImageObject, TaskType, Modality, ToolOutput
from backend.models.change_detection.inference import call_change_model


@pytest.fixture(scope="module")
def sample_image_pair(tmp_path_factory):
    """
    Pytest fixture to generate synthetic T1 and T2 satellite images for testing.
    """
    tmp_dir = tmp_path_factory.mktemp("sample_data")
    w, h = 400, 400

    # Image T1 (2021)
    t1 = Image.new("RGB", (w, h), color=(50, 140, 60))  # Green field
    draw1 = ImageDraw.Draw(t1)
    draw1.rectangle([50, 50, 150, 100], fill=(30, 90, 180))  # River patch
    t1_path = os.path.join(tmp_dir, "satellite_T1_2021.png")
    t1.save(t1_path)

    # Image T2 (2023) with 2 changes
    t2 = Image.new("RGB", (w, h), color=(50, 140, 60))
    draw2 = ImageDraw.Draw(t2)
    draw2.rectangle([50, 50, 150, 100], fill=(30, 90, 180))
    # Change 1: New Industrial Building (North-East)
    draw2.rectangle([250, 60, 360, 170], fill=(220, 220, 220))
    # Change 2: Water Flooding (South-West)
    draw2.ellipse([60, 250, 180, 360], fill=(20, 80, 190))

    t2_path = os.path.join(tmp_dir, "satellite_T2_2023.png")
    t2.save(t2_path)

    img_t1 = ImageObject(
        image_id="TEST_T1",
        file_path=t1_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[0.0, 0.0, 1.0, 1.0],
        acquisition_date="2021-06-01",
        width=w,
        height=h
    )

    img_t2 = ImageObject(
        image_id="TEST_T2",
        file_path=t2_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[0.0, 0.0, 1.0, 1.0],
        acquisition_date="2023-06-01",
        width=w,
        height=h
    )

    return img_t1, img_t2


def test_call_change_model_success_with_two_images(sample_image_pair):
    """
    Test successful change detection run with 2 valid images.
    Returns status == 'success' and populated spatial_evidence.
    """
    img_t1, img_t2 = sample_image_pair

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="What new buildings or water bodies appeared between 2021 and 2023?",
        images=[img_t1, img_t2]
    )

    output: ToolOutput = call_change_model(tool_input)

    assert output.status == "success"
    assert output.error_message is None
    assert output.confidence is not None and output.confidence > 0.5
    assert output.spatial_evidence is not None and len(output.spatial_evidence) >= 2

    # Check for both mask and bbox types in spatial_evidence
    evidence_types = [ev.type for ev in output.spatial_evidence]
    assert "mask" in evidence_types
    assert "bbox" in evidence_types


def test_call_change_model_single_image_error(sample_image_pair):
    """
    Test providing only 1 image returns status == 'error' with clear error_message.
    """
    img_t1, _ = sample_image_pair

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="Detect changes",
        images=[img_t1]
    )

    output: ToolOutput = call_change_model(tool_input)

    assert output.status == "error"
    assert output.error_message is not None
    assert "at least 2 images" in output.error_message.lower() or "requires" in output.error_message.lower()


def test_call_change_model_nonexistent_filepath_error(sample_image_pair):
    """
    Test providing nonexistent file path returns status == 'error'.
    """
    img_t1, _ = sample_image_pair

    nonexistent_img = ImageObject(
        image_id="NONEXISTENT",
        file_path="sample_data/nonexistent_image_12345.png",
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[0.0, 0.0, 1.0, 1.0],
        acquisition_date="2021-01-01",
        width=400,
        height=400
    )

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="Detect changes",
        images=[img_t1, nonexistent_img]
    )

    output: ToolOutput = call_change_model(tool_input)

    assert output.status == "error"
    assert output.error_message is not None
    assert "not found" in output.error_message.lower() or "failed" in output.error_message.lower()


def test_call_change_model_nonempty_model_used(sample_image_pair):
    """
    Test that ToolOutput.model_used is a non-empty string.
    """
    img_t1, img_t2 = sample_image_pair

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="Summarize changes",
        images=[img_t1, img_t2]
    )

    output: ToolOutput = call_change_model(tool_input)

    assert isinstance(output.model_used, str)
    assert len(output.model_used) > 0
    assert "P3" in output.model_used


def test_call_change_model_text_answer_vlm_or_fallback(sample_image_pair):
    """
    Test that text_answer is non-empty and generates meaningful region description.
    """
    img_t1, img_t2 = sample_image_pair

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="Count all changes",
        images=[img_t1, img_t2]
    )

    output: ToolOutput = call_change_model(tool_input)

    assert output.text_answer is not None
    assert len(output.text_answer) > 20
    # Ensure it's not empty or equal to simple default placeholder
    assert output.text_answer != "Detected 0 distinct changed region(s)"


def test_call_change_model_json_input_output(sample_image_pair):
    """
    Test that call_change_model_json accepts raw JSON string and returns structured JSON string response.
    """
    from backend.models.change_detection.inference import call_change_model_json
    import json

    img_t1, img_t2 = sample_image_pair

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="What changed between T1 and T2?",
        images=[img_t1, img_t2]
    )

    json_input_str = tool_input.model_dump_json()

    json_output_str = call_change_model_json(json_input_str)

    assert isinstance(json_output_str, str)
    parsed_output = json.loads(json_output_str)

    assert parsed_output["status"] == "success"
    assert "text_answer" in parsed_output and parsed_output["text_answer"] is not None
    assert "spatial_evidence" in parsed_output and isinstance(parsed_output["spatial_evidence"], list)
    assert "model_used" in parsed_output and "P3" in parsed_output["model_used"]


def test_call_change_model_multi_domain_building_and_water(sample_image_pair):
    """
    Test multi-domain change detection accuracy for both building construction and water inundation.
    """
    img_t1, img_t2 = sample_image_pair

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="Identify all new building construction and water level changes between 2021 and 2023.",
        images=[img_t1, img_t2]
    )

    output: ToolOutput = call_change_model(tool_input)

    assert output.status == "success"
    assert output.confidence >= 0.85
    assert len(output.spatial_evidence) >= 3  # Mask + at least 2 bboxes

    # Verify structured answer description contains multi-domain change types
    answer_text = output.text_answer
    assert "Bi-Temporal" in answer_text
    assert "Building" in answer_text or "Structure" in answer_text
    assert "Water" in answer_text or "Flooding" in answer_text


def test_call_change_model_high_confidence_scoring(sample_image_pair):
    """
    Test that SNR-based confidence scoring yields high confidence (>0.85) on valid change images.
    """
    img_t1, img_t2 = sample_image_pair

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="What changed?",
        images=[img_t1, img_t2]
    )

    output: ToolOutput = call_change_model(tool_input)

    assert output.status == "success"
    assert output.confidence is not None
    assert 0.85 <= output.confidence <= 0.98


def test_call_change_model_identical_images_no_changes(sample_image_pair):
    """
    Test providing identical images (T1 vs T1) yields zero bounding boxes,
    high confidence (>=0.90), and explicit 'no changes' text answer.
    """
    img_t1, _ = sample_image_pair

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="What changed between T1 and T1?",
        images=[img_t1, img_t1]  # Identical images
    )

    output: ToolOutput = call_change_model(tool_input)

    assert output.status == "success"
    assert output.confidence is not None and output.confidence >= 0.90

    # Ensure no bounding boxes are returned for identical images
    bboxes = [ev for ev in output.spatial_evidence if ev.type == "bbox"]
    assert len(bboxes) == 0

    # Ensure text answer explicitly indicates no changes detected
    assert output.text_answer is not None
    assert "no" in output.text_answer.lower()
    assert "change" in output.text_answer.lower()



