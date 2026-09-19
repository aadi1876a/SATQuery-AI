"""
backend/tests/test_p2.py
=============================================================================
SatQuery AI — Person 2 (P2) Automated Verification Test Suite
=============================================================================
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image, ImageDraw

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURRENT_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for p in [_PROJECT_ROOT, _BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.schemas import Modality, ImageObject, TaskType, ToolInput, ToolOutput, SpatialEvidence
from backend.models.vqa_captioning_grounding import run, call_vqa_model, call_caption_model, call_grounding_model
from backend.models.vqa_captioning_grounding.utils import OUTPUTS_DIR


def create_test_satellite_patch(filepath: str) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img = Image.new("RGB", (512, 512), color=(75, 120, 65))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 180), (180, 240), (360, 310), (512, 360)], fill=(30, 85, 160), width=48)
    draw.line([(70, 0), (90, 512)], fill=(110, 110, 110), width=12)
    for b in [(130, 70, 210, 140), (240, 80, 310, 150), (140, 340, 220, 410), (260, 350, 330, 420)]:
        draw.rectangle(b, fill=(220, 210, 195), outline=(50, 50, 50), width=2)
    img.save(filepath)
    return filepath


class TestP2Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def test_01_schemas_validation(self):
        tool_in = ToolInput(task=TaskType.vqa, query="Are there buildings visible?", images=[self.image_obj])
        self.assertEqual(tool_in.task, TaskType.vqa)

        dummy_output = ToolOutput(
            status="success", text_answer="Yes", spatial_evidence=[SpatialEvidence(type="bbox", coords=[10, 10, 50, 50], label="building")],
            confidence=0.92, model_used="test-model"
        )
        self.assertEqual(dummy_output.status, "success")

    @patch("backend.models.vqa_captioning_grounding.vqa._get_vqa_pipeline")
    def test_03_vqa_model(self, mock_vqa_pipeline):
        mock_processor = MagicMock()
        mock_processor.return_value.to.return_value = {"input_ids": "mock"}
        mock_model = MagicMock()
        mock_out = MagicMock()
        mock_out.sequences = [[1, 2, 3]]
        mock_model.generate.return_value = mock_out
        mock_processor.decode.return_value = "buildings are visible"
        mock_vqa_pipeline.return_value = (mock_processor, mock_model, "mock-vqa-model", 0.0)
        
        tool_in = ToolInput(task=TaskType.vqa, query="What is visible?", images=[self.image_obj])
        output = call_vqa_model(tool_in)
        self.assertEqual(output.status, "success")
        self.assertIn("Buildings are visible", output.text_answer)
        self.assertIsNotNone(output.raw_output_path)
        self.assertFalse(os.path.isabs(output.raw_output_path), "Sidecar path must be relative")

    @patch("backend.models.vqa_captioning_grounding.captioning._get_caption_pipeline")
    def test_04_captioning_model(self, mock_cap_pipeline):
        mock_processor = MagicMock()
        mock_processor.return_value.to.return_value = {"input_ids": "mock"}
        mock_model = MagicMock()
        mock_out = MagicMock()
        mock_out.sequences = [[1, 2, 3]]
        del mock_out.sequences_scores
        del mock_out.scores
        mock_model.generate.return_value = mock_out
        mock_processor.decode.return_value = "a satellite image of buildings"
        mock_cap_pipeline.return_value = (mock_processor, mock_model, "mock-cap-model", 0.0)

        tool_in = ToolInput(task=TaskType.captioning, query="", images=[self.image_obj])
        output = call_caption_model(tool_in)
        self.assertEqual(output.status, "success")
        self.assertTrue(len(output.text_answer) > 0)
        self.assertFalse(os.path.isabs(output.raw_output_path), "Sidecar path must be relative")

    @patch("backend.models.vqa_captioning_grounding.grounding._get_sam_pipeline")
    @patch("backend.models.vqa_captioning_grounding.grounding._get_grounding_pipeline")
    def test_05_grounding_model(self, mock_grd_pipeline, mock_sam_pipeline):
        mock_proc = MagicMock()
        mock_img_proc = MagicMock()
        mock_model = MagicMock()
        mock_grd_pipeline.return_value = (mock_proc, mock_img_proc, mock_model, 0.0)

        import torch
        mock_results = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]
        mock_img_proc.post_process_object_detection.return_value = mock_results

        mock_sam_pipeline.side_effect = Exception("SAM is offline")

        tool_in = ToolInput(task=TaskType.grounding, query="buildings", images=[self.image_obj], params={"threshold": 0.01})
        output = call_grounding_model(tool_in)
        
        self.assertEqual(output.status, "success")
        self.assertTrue(len(output.spatial_evidence) > 0)
        self.assertEqual(output.spatial_evidence[0].type, "bbox")
        if output.raw_output_path:
            self.assertFalse(os.path.isabs(output.raw_output_path), "Overlay path must be relative")

    @patch("backend.models.vqa_captioning_grounding.inference.run_vqa")
    def test_06_dispatcher(self, mock_vqa):
        mock_vqa.return_value = ToolOutput(status="success", text_answer="mocked", spatial_evidence=[], confidence=1.0, model_used="m")
        tool_in = ToolInput(task=TaskType.vqa, query="test", images=[self.image_obj])
        out = run(tool_in)
        self.assertEqual(out.status, "success")
        mock_vqa.assert_called_once()

    @patch("backend.models.vqa_captioning_grounding.inference.run_grounding")
    def test_cross_check(self, mock_grounding):
        mock_grounding.return_value = ToolOutput(status="success", text_answer="Detected", spatial_evidence=[SpatialEvidence(type="bbox", coords=[0,0,1,1], label="X")], confidence=1.0, model_used="m")
        tool_in = ToolInput(task=TaskType.vqa, query="Is there a building?", images=[self.image_obj], params={"verify_with_grounding": True})
        out = run(tool_in)
        self.assertEqual(out.text_answer, "Yes, building is present.")
        mock_grounding.assert_called_once()


if __name__ == "__main__":
    unittest.main()
