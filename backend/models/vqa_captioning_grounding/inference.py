"""
backend/models/vqa_captioning_grounding/inference.py
=============================================================================
SatQuery AI — Person 2 (P2): VLM, VQA, Captioning & Segment-wise Grounding
=============================================================================
Unified P2 entry point for Person 5 (P5).

PUBLIC API:
  run(tool_input: ToolInput) -> ToolOutput          [P5 integration point]
  call_vqa_model(tool_input: ToolInput) -> ToolOutput
  call_caption_model(tool_input: ToolInput) -> ToolOutput
  call_grounding_model(tool_input: ToolInput) -> ToolOutput

CLI (independent testing without P1 or P5):
  python inference.py --task vqa      [--image path] [--query "question"]
  python inference.py --task captioning [--image path]
  python inference.py --task grounding  [--image path] [--query "buildings"]
  python inference.py --task all      [--image path]

P2 handles:
  TaskType.vqa        → BLIP VQA (Salesforce/blip-vqa-base)
  TaskType.captioning → BLIP Caption (Salesforce/blip-image-captioning-base)
  TaskType.grounding  → OWLv2 (google/owlv2-base-patch16-ensemble) + SAM (facebook/sam-vit-base)

P2 does NOT handle:
  TaskType.change_vqa    → P3 (change detection)
  TaskType.fusion_analysis → P4 (fusion)
"""

import os
import sys
import argparse
from typing import List

# ---------------------------------------------------------------------------
# Path bootstrap — makes 'backend.schemas' importable from any working dir
# ---------------------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_CURRENT_DIR)
_BACKEND_DIR = os.path.dirname(_MODELS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for path in [_PROJECT_ROOT, _BACKEND_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.schemas import ImageObject, ToolInput, ToolOutput, SpatialEvidence, TaskType, Modality

# ---------------------------------------------------------------------------
# Import modular P2 components
# (All preprocessing, model loading, and postprocessing live in submodules)
# ---------------------------------------------------------------------------
from .vqa import run_vqa, VQA_MODEL_ID
from .captioning import run_captioning, CAPTION_MODEL_ID
from .grounding import run_grounding, GROUNDING_MODEL_ID, SAM_MODEL_ID, COMBINED_MODEL_ID
from .preprocessing import load_image_rgb
from .postprocessing import save_mask, generate_overlay
from .utils import get_device, ensure_dirs, OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Backward-compatible aliases (used by existing test_p2.py and __init__.py)
# ---------------------------------------------------------------------------
def call_vqa_model(tool_input: ToolInput) -> ToolOutput:
    """Backward-compatible alias for run_vqa()."""
    return run_vqa(tool_input)


def call_caption_model(tool_input: ToolInput) -> ToolOutput:
    """Backward-compatible alias for run_captioning()."""
    return run_captioning(tool_input)


def call_grounding_model(tool_input: ToolInput) -> ToolOutput:
    """Backward-compatible alias for run_grounding()."""
    return run_grounding(tool_input)


# Also expose _save_mask and _generate_overlay for existing unit test imports
def _save_mask(mask_arr, image_id: str, label: str, idx: int) -> str:
    """Backward-compatible alias for save_mask() (used in test_p2.py)."""
    return save_mask(mask_arr, image_id, label, idx, modality="optical")


def _generate_overlay(original_img, boxes, masks, labels, image_id: str) -> str:
    """Backward-compatible alias for generate_overlay() (used in test_p2.py)."""
    return generate_overlay(original_img, boxes, masks, labels, image_id, modality="optical")


# ---------------------------------------------------------------------------
# Unified P5 Entry Point
# ---------------------------------------------------------------------------

def run(tool_input: ToolInput) -> ToolOutput:
    """
    Unified P5 entry point for Person 2 (P2).

    Routes to the correct P2 model based on task type:
      vqa        → BLIP VQA
      captioning → BLIP Captioning
      grounding  → OWLv2 + SAM Segmentation

    P5 does not need to know which VLM, grounding, or segmentation model
    is used internally. Those details are encapsulated here.

    Usage:
        from models.vqa_captioning_grounding.inference import run
        output = run(tool_input)
        assert isinstance(output, ToolOutput)
    """
    if tool_input.task == TaskType.vqa:
        return run_vqa(tool_input)
    elif tool_input.task == TaskType.captioning:
        return run_captioning(tool_input)
    elif tool_input.task == TaskType.grounding:
        return run_grounding(tool_input)
    else:
        return ToolOutput(
            status="error",
            text_answer=None,
            spatial_evidence=[],
            confidence=None,
            model_used="P2-Dispatcher",
            error_message=(
                f"Task '{tool_input.task}' is not handled by P2. "
                f"P2 handles: vqa, captioning, grounding. "
                f"Change analysis belongs to P3; Fusion analysis belongs to P4."
            )
        )


# ---------------------------------------------------------------------------
# CLI Standalone Runner
# ---------------------------------------------------------------------------

def _create_sample_patch(filepath: str) -> str:
    """Generates a simple satellite-scene-like test image for CLI testing."""
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img = Image.new("RGB", (512, 512), color=(80, 115, 60))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 200), (200, 250), (400, 300), (512, 350)], fill=(30, 80, 150), width=50)
    draw.line([(80, 0), (100, 512)], fill=(120, 120, 120), width=10)
    for b in [(130, 80, 200, 140), (240, 90, 300, 150), (140, 340, 210, 400), (250, 350, 320, 410)]:
        draw.rectangle(b, fill=(210, 200, 190), outline=(40, 40, 40), width=2)
    img.save(filepath)
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2 Model Tester (Independent of P1/P5)")
    parser.add_argument("--task", type=str, default="all",
                        choices=["vqa", "captioning", "grounding", "all"],
                        help="Task to run: vqa, captioning, grounding, or all")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to satellite image (PNG, JPG, TIFF)")
    parser.add_argument("--query", type=str, default=None,
                        help="Question for VQA or target for Grounding")
    parser.add_argument("--modality", type=str, default="optical",
                        choices=["optical", "sar"],
                        help="Image modality: optical or sar")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Grounding detection confidence threshold")
    args = parser.parse_args()

    img_path = args.image
    if not img_path or not os.path.exists(img_path):
        sample_path = os.path.join(OUTPUTS_DIR, "test_images", "optical", "cli_sample_patch.png")
        img_path = _create_sample_patch(sample_path)
        print(f"[P2 CLI] No image provided. Using generated sample: {img_path}")
        modality = Modality.optical
    else:
        modality = Modality.sar if args.modality == "sar" else Modality.optical

    from PIL import Image
    w, h = Image.open(img_path).size
    
    # Use the image filename (without extension) as the unique image ID to avoid overwriting outputs
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    img_obj = ImageObject(
        image_id=base_name,
        file_path=os.path.abspath(img_path),
        modality=modality,
        format=os.path.splitext(img_path)[1].lstrip(".") or "jpg",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[0.0, 0.0, 1.0, 1.0],
        width=w,
        height=h,
    )

    tasks = ["vqa", "captioning", "grounding"] if args.task == "all" else [args.task]

    for t in tasks:
        print("\n" + "=" * 65)
        if t == "vqa":
            q = args.query or "What objects or natural features are visible?"
            print(f" TESTING VQA | Question: '{q}'")
            print("=" * 65)
            res = run_vqa(ToolInput(task=TaskType.vqa, query=q, images=[img_obj]))
        elif t == "captioning":
            print(" TESTING CAPTIONING")
            print("=" * 65)
            res = run_captioning(ToolInput(task=TaskType.captioning, query="", images=[img_obj]))
        elif t == "grounding":
            target = args.query or "buildings"
            print(f" TESTING GROUNDING | Target: '{target}'")
            print("=" * 65)
            res = run_grounding(ToolInput(
                task=TaskType.grounding, query=target, images=[img_obj],
                params={"threshold": args.threshold}
            ))

        print(f" Status        : {res.status}")
        print(f" Model Used    : {res.model_used}")
        print(f" Answer/Output : {res.text_answer}")
        print(f" Confidence    : {res.confidence}")
        print(f" Evidence Count: {len(res.spatial_evidence)}")
        if res.raw_output_path:
            print(f" Visual Overlay: {res.raw_output_path}")
        if res.error_message:
            print(f" Error Message : {res.error_message}")
        print("=" * 65)
