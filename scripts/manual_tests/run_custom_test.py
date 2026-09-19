"""
run_custom_test.py
SatQuery AI - Easy Custom Satellite Image Tester for P3 Specialist Model.

Use this script to test YOUR OWN real satellite image pair (Before & After)!
Usage:
  python run_custom_test.py --t1 path/to/before.png --t2 path/to/after.png --query "What changed?"
"""

import sys
import os
import argparse
from PIL import Image
from backend.app.schemas.schemas import ToolInput, ImageObject, TaskType, Modality
from p3_model import P3ChangeDetectionEngine



def run_custom_satellite_test(img_t1_path: str, img_t2_path: str, query: str = "Detect all changes between T1 and T2"):
    print("=" * 75)
    print(" 📡 SatQuery AI - Running P3 Model on Real Satellite Image Pair")
    print("=" * 75)

    if not os.path.exists(img_t1_path):
        print(f"❌ Error: T1 image file not found at: {img_t1_path}")
        return
    if not os.path.exists(img_t2_path):
        print(f"❌ Error: T2 image file not found at: {img_t2_path}")
        return

    print(f"Image T1 (Before): {img_t1_path}")
    print(f"Image T2 (After):  {img_t2_path}")
    print(f"User Query:        '{query}'\n")

    pil_1 = Image.open(img_t1_path)
    pil_2 = Image.open(img_t2_path)

    img_t1 = ImageObject(
        image_id="CUSTOM_T1",
        file_path=os.path.abspath(img_t1_path),
        modality=Modality.optical,
        format=img_t1_path.split(".")[-1],
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[0.0, 0.0, 1.0, 1.0],
        acquisition_date="Before",
        width=pil_1.width,
        height=pil_1.height
    )

    img_t2 = ImageObject(
        image_id="CUSTOM_T2",
        file_path=os.path.abspath(img_t2_path),
        modality=Modality.optical,
        format=img_t2_path.split(".")[-1],
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[0.0, 0.0, 1.0, 1.0],
        acquisition_date="After",
        width=pil_2.width,
        height=pil_2.height
    )


    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query=query,
        images=[img_t1, img_t2]
    )

    # Execute P3 Engine
    engine = P3ChangeDetectionEngine(output_dir="outputs")
    output = engine.run(tool_input)

    print("-" * 75)
    print(f"STATUS:              {output.status}")
    print(f"MODEL USED:          {output.model_used}")
    print(f"CONFIDENCE:          {output.confidence}")
    print(f"💬 VQA TEXT ANSWER:\n  {output.text_answer}\n")
    print(f"🖼️ RED OVERLAY IMAGE: {output.raw_output_path}")

    bbox_evidences = [e for e in output.spatial_evidence if e.type == "bbox"]
    mask_evidences = [e for e in output.spatial_evidence if e.type == "mask"]

    if mask_evidences:
        print(f"🖼️ BINARY MASK IMAGE: {mask_evidences[0].mask_path}")

    print(f"\nEXTRACTED BOUNDING BOXES ({len(bbox_evidences)} changes detected):")
    for idx, ev in enumerate(bbox_evidences, 1):
        print(f"  [{idx}] {ev.label}: coords={ev.coords}")

    print("=" * 75)
    print(" ✅ CUSTOM SATELLITE IMAGE TEST COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test P3 Change Detection Model on Custom Images")
    parser.add_argument("--t1", type=str, default="sample_data/satellite_T5_2023.png", help="Path to T1 image")
    parser.add_argument("--t2", type=str, default="sample_data/satellite_T6_2025.png", help="Path to T2 image")
    parser.add_argument("--query", type=str, default="What changes occurred between T1 and T2?", help="VQA Question")

    args = parser.parse_args()
    run_custom_satellite_test(args.t1, args.t2, args.query)
