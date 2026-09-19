"""
test_controller.py
Run with: pytest tests/test_controller.py -v
Confirms the controller correctly routes all 5 representative query types.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import ImageObject
from agent.controller import run_query


def make_image(image_id="img1", modality="optical"):
    return ImageObject(
        image_id=image_id,
        file_path=f"/data/{image_id}.tif",
        modality=modality,
        format="GeoTIFF",
        bands=4,
        resolution_m=10,
        crs="EPSG:32643",
        bbox=[72.81, 21.15, 72.84, 21.18],
        width=512,
        height=512,
    )


def test_vqa_single_image():
    resp = run_query("What is the dominant land cover here?", [make_image()])
    assert resp.status == "success"
    assert resp.execution_trace.task_detected == "vqa"


def test_grounding_single_image():
    resp = run_query("Highlight the water body in this image.", [make_image()])
    assert resp.status == "success"
    assert resp.execution_trace.task_detected == "grounding"
    assert len(resp.visual_evidence) > 0


def test_change_vqa_needs_two_images():
    resp = run_query("What changed between these two dates?", [make_image("a"), make_image("b")])
    assert resp.status == "success"
    assert resp.execution_trace.task_detected == "change_vqa"


def test_change_vqa_rejects_single_image():
    resp = run_query("What changed between these two dates?", [make_image("a")])
    assert resp.status == "rejected"
    assert "requires 2 image" in resp.reason


def test_fusion_analysis():
    imgs = [make_image("a", "optical"), make_image("b", "sar")]
    resp = run_query("Use the optical and SAR images together to find built-up regions.", imgs)
    assert resp.status == "success"
    assert resp.execution_trace.task_detected == "fusion_analysis"
