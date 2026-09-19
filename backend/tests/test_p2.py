"""
backend/tests/test_p2.py
=============================================================================
SatQuery AI — Person 2 (P2) Automated Verification Test Suite
=============================================================================
Tests:
  1. Schemas & Data Contracts (Pydantic v2 validation)
  2. Preprocessing & SAR Pipeline (Lee filter, dB, CLAHE, false-color)
  3. Visual Question Answering (BLIP VQA)
  4. Image Captioning (BLIP Captioning)
  5. Grounding & Segmentation (OWLv2 + SAM)
  6. Unified P5 Dispatcher (run() routing and validation)

Run with:
  python backend/tests/test_p2.py
  or
  pytest backend/tests/test_p2.py -v -s
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend and root are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURRENT_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for p in [_PROJECT_ROOT, _BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.schemas import (
    Modality,
    ImageObject,
    TaskType,
    ToolInput,
    ToolOutput,
    SpatialEvidence,
)
from backend.models.vqa_captioning_grounding import (
    run,
    call_vqa_model,
    call_caption_model,
    call_grounding_model,
)
from backend.models.vqa_captioning_grounding.preprocessing import (
    load_image_rgb,
    preprocess_sar_band,
    lee_speckle_filter,
    percentile_stretch,
)
from backend.models.vqa_captioning_grounding.postprocessing import validate_mask
from backend.models.vqa_captioning_grounding.utils import OUTPUTS_DIR


def create_test_satellite_patch(filepath: str) -> str:
    """Generates a synthetic satellite scene with distinct features: river, road, buildings."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img = Image.new("RGB", (512, 512), color=(75, 120, 65))  # Green vegetation
    draw = ImageDraw.Draw(img)

    # Blue river
    draw.line([(0, 180), (180, 240), (360, 310), (512, 360)], fill=(30, 85, 160), width=48)
    # Gray asphalt road
    draw.line([(70, 0), (90, 512)], fill=(110, 110, 110), width=12)
    # Rectangular buildings (light roof)
    for b in [
        (130, 70, 210, 140),
        (240, 80, 310, 150),
        (140, 340, 220, 410),
        (260, 350, 330, 420),
    ]:
        draw.rectangle(b, fill=(220, 210, 195), outline=(50, 50, 50), width=2)

    img.save(filepath)
    return filepath


class TestP2Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare sample image objects for testing."""
        cls.test_dir = os.path.join(OUTPUTS_DIR, "test_images", "optical")
        cls.img_path = os.path.join(cls.test_dir, "test_patch.png")
        create_test_satellite_patch(cls.img_path)

        cls.image_obj = ImageObject(
            image_id="test_patch_01",
            file_path=os.path.abspath(cls.img_path),
            modality=Modality.optical,
            format="png",
            bands=3,
            resolution_m=10.0,
            crs="EPSG:4326",
            bbox=[77.10, 28.50, 77.25, 28.65],
            width=512,
            height=512,
        )

        # Prepare SAR test image
        from backend.models.vqa_captioning_grounding.create_sample_sar import save_sar_test_patch
        sar_paths = save_sar_test_patch()
        cls.sar_img_path = sar_paths["tif"] if os.path.exists(sar_paths["tif"]) else sar_paths["png"]
        cls.sar_image_obj = ImageObject(
            image_id="test_sar_01",
            file_path=os.path.abspath(cls.sar_img_path),
            modality=Modality.sar,
            format=os.path.splitext(cls.sar_img_path)[1].lstrip(".") or "tif",
            bands=1,
            resolution_m=10.0,
            crs="EPSG:4326",
            bbox=[77.10, 28.50, 77.25, 28.65],
            width=512,
            height=512,
        )

    def test_01_schemas_validation(self):
        """Test schemas contract validity and Pydantic serialization."""
        self.assertEqual(self.image_obj.modality, Modality.optical)
        self.assertEqual(self.image_obj.width, 512)

        tool_in = ToolInput(
            task=TaskType.vqa,
            query="Are there buildings visible?",
            images=[self.image_obj],
        )
        self.assertEqual(tool_in.task, TaskType.vqa)

        dummy_output = ToolOutput(
            status="success",
            text_answer="Yes",
            spatial_evidence=[
                SpatialEvidence(type="bbox", coords=[10, 10, 50, 50], label="building")
            ],
            confidence=0.92,
            model_used="test-model",
        )
        self.assertEqual(dummy_output.status, "success")
        self.assertEqual(len(dummy_output.spatial_evidence), 1)

    def test_02_preprocessing_and_sar(self):
        """Test optical image loading and SAR preprocessing pipeline."""
        # 1. Optical loading
        pil_img, path = load_image_rgb(self.image_obj)
        self.assertEqual(pil_img.size, (512, 512))
        self.assertEqual(pil_img.mode, "RGB")

        # 2. Synthetic SAR preprocessing (single-band float32 simulation)
        np.random.seed(42)
        raw_sar = np.random.exponential(scale=10.0, size=(256, 256)).astype(np.float32)
        filtered = lee_speckle_filter(raw_sar, kernel_size=5)
        self.assertEqual(filtered.shape, (256, 256))

        # Test false color SAR conversion
        false_color = preprocess_sar_band(raw_sar, use_false_color=True)
        self.assertEqual(false_color.shape, (256, 256, 3))
        self.assertEqual(false_color.dtype, np.uint8)

        # Test grayscale SAR conversion
        gray_sar = preprocess_sar_band(raw_sar, use_false_color=False)
        self.assertEqual(gray_sar.shape, (256, 256, 3))
        self.assertEqual(gray_sar.dtype, np.uint8)

    def test_03_vqa_model(self):
        """Test BLIP Visual Question Answering inference."""
        print("\n--- Running VQA Model Test ---")
        tool_in = ToolInput(
            task=TaskType.vqa,
            query="What is visible in this satellite remote sensing image?",
            images=[self.image_obj],
        )
        output = call_vqa_model(tool_in)
        self.assertEqual(output.status, "success", f"VQA failed: {output.error_message}")
        self.assertIsNotNone(output.text_answer)
        self.assertTrue(len(output.text_answer) > 0)
        print(f"[VQA Test Result] Model: {output.model_used}")
        print(f"[VQA Test Result] Answer: {output.text_answer}")
        print(f"[VQA Test Result] Confidence: {output.confidence}")

    def test_04_captioning_model(self):
        """Test BLIP Image Captioning inference."""
        print("\n--- Running Captioning Model Test ---")
        tool_in = ToolInput(
            task=TaskType.captioning,
            query="A satellite aerial view showing",
            images=[self.image_obj],
        )
        output = call_caption_model(tool_in)
        self.assertEqual(output.status, "success", f"Captioning failed: {output.error_message}")
        self.assertIsNotNone(output.text_answer)
        self.assertTrue(len(output.text_answer) > 0)
        print(f"[Captioning Test Result] Model: {output.model_used}")
        print(f"[Captioning Test Result] Caption: {output.text_answer}")
        print(f"[Captioning Test Result] Confidence: {output.confidence}")

    def test_05_grounding_model(self):
        """Test OWLv2 Open-Vocabulary Grounding + SAM Segmentation inference."""
        print("\n--- Running Grounding + SAM Model Test ---")
        tool_in = ToolInput(
            task=TaskType.grounding,
            query="buildings",
            images=[self.image_obj],
            params={"threshold": 0.005, "max_detections": 3},
        )
        output = call_grounding_model(tool_in)
        self.assertEqual(output.status, "success", f"Grounding failed: {output.error_message}")
        print(f"[Grounding Test Result] Model: {output.model_used}")
        print(f"[Grounding Test Result] Text: {output.text_answer}")
        print(f"[Grounding Test Result] Evidence Count: {len(output.spatial_evidence)}")
        if output.raw_output_path:
            self.assertTrue(os.path.exists(output.raw_output_path))
            print(f"[Grounding Test Result] Overlay saved: {output.raw_output_path}")

        # If detections occurred, verify mask files
        for ev in output.spatial_evidence:
            if ev.type == "mask" and ev.mask_path:
                v = validate_mask(ev.mask_path)
                self.assertTrue(v["exists"])
                self.assertTrue(v["readable"])

    def test_06_dispatcher_and_error_handling(self):
        """Test unified P5 run() entry point and task routing."""
        # Test valid routing through run()
        tool_in = ToolInput(
            task=TaskType.captioning,
            query="",
            images=[self.image_obj],
        )
        out = run(tool_in)
        self.assertEqual(out.status, "success")

        # Test invalid task rejection (P3/P4 task routed to P2)
        invalid_in = ToolInput(
            task=TaskType.change_vqa,
            query="Has vegetation changed?",
            images=[self.image_obj],
        )
        err_out = run(invalid_in)
        self.assertEqual(err_out.status, "error")
        self.assertIn("not handled by P2", err_out.error_message)

    def test_07_sar_multimodal_pipeline(self):
        """Test full SAR multimodal pipeline: VQA, Captioning, Grounding."""
        print("\n--- Running SAR Multimodal Pipeline Tests ---")

        # 1. SAR VQA
        vqa_in = ToolInput(
            task=TaskType.vqa,
            query="What structures or features are visible in this radar image?",
            images=[self.sar_image_obj],
        )
        vqa_out = call_vqa_model(vqa_in)
        self.assertEqual(vqa_out.status, "success")
        self.assertIsNotNone(vqa_out.text_answer)
        self.assertIn("Note: SAR image was converted", vqa_out.text_answer)
        print(f"[SAR VQA] Answer: {vqa_out.text_answer}")

        # 2. SAR Captioning
        cap_in = ToolInput(
            task=TaskType.captioning,
            query="A synthetic aperture radar SAR satellite view showing",
            images=[self.sar_image_obj],
        )
        cap_out = call_caption_model(cap_in)
        self.assertEqual(cap_out.status, "success")
        self.assertIsNotNone(cap_out.text_answer)
        self.assertIn("Note: SAR image converted", cap_out.text_answer)
        print(f"[SAR Captioning] Caption: {cap_out.text_answer}")

        # 3. SAR Grounding
        grd_in = ToolInput(
            task=TaskType.grounding,
            query="structures",
            images=[self.sar_image_obj],
            params={"threshold": 0.005, "max_detections": 3},
        )
        grd_out = call_grounding_model(grd_in)
        self.assertEqual(grd_out.status, "success")
        self.assertIsNotNone(grd_out.text_answer)
        print(f"[SAR Grounding] Result: {grd_out.text_answer}")
        if grd_out.raw_output_path:
            self.assertTrue(os.path.exists(grd_out.raw_output_path))
            print(f"[SAR Grounding] Overlay saved: {grd_out.raw_output_path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
