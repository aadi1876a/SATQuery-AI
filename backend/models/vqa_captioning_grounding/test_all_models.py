"""
backend/models/vqa_captioning_grounding/test_all_models.py
=============================================================================
SatQuery AI — Person 2 (P2): Multi-Modal Comprehensive Model Test Suite
=============================================================================
Allows testing the three P2 models (BLIP VQA, BLIP Captioning, OWLv2+SAM Grounding)
both INDIVIDUALLY and TOGETHER on both OPTICAL and SAR imagery.

Usage Examples:
  # 1. Test individual models on optical imagery:
  python test_all_models.py --modality optical --task vqa
  python test_all_models.py --modality optical --task captioning
  python test_all_models.py --modality optical --task grounding

  # 2. Test individual models on SAR imagery:
  python test_all_models.py --modality sar --task vqa
  python test_all_models.py --modality sar --task captioning
  python test_all_models.py --modality sar --task grounding

  # 3. Test all three models together on a specific modality:
  python test_all_models.py --modality optical --task all
  python test_all_models.py --modality sar --task all

  # 4. Test all three models together on BOTH optical and SAR (Full Evaluation):
  python test_all_models.py --modality all --task all
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any
from PIL import Image

# Setup paths
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_CURRENT_DIR)
_BACKEND_DIR = os.path.dirname(_MODELS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for p in [_PROJECT_ROOT, _BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.schemas import ImageObject, ToolInput, ToolOutput, TaskType, Modality
from backend.models.vqa_captioning_grounding.vqa import run_vqa
from backend.models.vqa_captioning_grounding.captioning import run_captioning
from backend.models.vqa_captioning_grounding.grounding import run_grounding
from backend.models.vqa_captioning_grounding.inference import run
from backend.models.vqa_captioning_grounding.utils import OUTPUTS_DIR, ensure_dirs
from backend.models.vqa_captioning_grounding.create_sample_sar import save_sar_test_patch


def get_or_create_test_images() -> Dict[str, ImageObject]:
    """Ensures test optical and SAR images exist and returns ImageObject references."""
    # 1. Optical test image
    opt_dir = os.path.join(OUTPUTS_DIR, "test_images", "optical")
    os.makedirs(opt_dir, exist_ok=True)
    opt_path = os.path.join(opt_dir, "test_patch.png")
    if not os.path.exists(opt_path):
        from backend.tests.test_p2 import create_test_satellite_patch
        create_test_satellite_patch(opt_path)

    w_opt, h_opt = Image.open(opt_path).size
    optical_obj = ImageObject(
        image_id="optical_scene_eval",
        file_path=os.path.abspath(opt_path),
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.10, 28.50, 77.25, 28.65],
        width=w_opt,
        height=h_opt,
    )

    # 2. SAR test image
    sar_dir = os.path.join(OUTPUTS_DIR, "test_images", "sar")
    sar_png = os.path.join(sar_dir, "sar_sample_patch.png")
    sar_tif = os.path.join(sar_dir, "sar_sample_patch.tif")
    if not os.path.exists(sar_png) or not os.path.exists(sar_tif):
        save_sar_test_patch(sar_dir)

    target_sar = sar_tif if os.path.exists(sar_tif) else sar_png
    w_sar, h_sar = Image.open(sar_png).size
    sar_obj = ImageObject(
        image_id="sar_scene_eval",
        file_path=os.path.abspath(target_sar),
        modality=Modality.sar,
        format=os.path.splitext(target_sar)[1].lstrip(".") or "tif",
        bands=1,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.10, 28.50, 77.25, 28.65],
        width=w_sar,
        height=h_sar,
    )

    return {"optical": optical_obj, "sar": sar_obj}


def test_single_task(
    task: str,
    modality: str,
    image_obj: ImageObject,
    query: str = None,
    threshold: float = 0.01,
) -> Dict[str, Any]:
    """Runs an individual model test and records execution telemetry."""
    print(f"\n{'='*70}")
    print(f"  EXECUTING: Task [{task.upper()}] on Modality [{modality.upper()}]")
    print(f"  Image Target: {image_obj.file_path}")
    print(f"{'='*70}")

    t0 = time.time()
    if task == "vqa":
        q = query or (
            "What objects or natural features are visible in this scene?"
            if modality == "optical"
            else "What high backscatter structures or water features are visible?"
        )
        print(f"  Query / Question : '{q}'")
        tool_in = ToolInput(task=TaskType.vqa, query=q, images=[image_obj])
        result = run_vqa(tool_in)

    elif task == "captioning":
        q = query or (
            "A high-resolution satellite aerial view showing"
            if modality == "optical"
            else "A synthetic aperture radar SAR satellite view showing"
        )
        print(f"  Prompt Prefix    : '{q}'")
        tool_in = ToolInput(task=TaskType.captioning, query=q, images=[image_obj])
        result = run_captioning(tool_in)

    elif task == "grounding":
        target = query or ("buildings" if modality == "optical" else "structures")
        print(f"  Grounding Target : '{target}' (threshold={threshold})")
        tool_in = ToolInput(
            task=TaskType.grounding,
            query=target,
            images=[image_obj],
            params={"threshold": threshold, "max_detections": 4},
        )
        result = run_grounding(tool_in)

    else:
        raise ValueError(f"Unknown task: {task}")

    latency = round(time.time() - t0, 3)

    print(f"\n  [RESULT SUMMARY]")
    print(f"  Status          : {result.status}")
    print(f"  Model Used      : {result.model_used}")
    print(f"  Latency         : {latency}s")
    print(f"  Confidence      : {result.confidence}")
    print(f"  Answer / Text   : {result.text_answer}")
    print(f"  Spatial Evidence: {len(result.spatial_evidence)} item(s)")
    if result.raw_output_path:
        print(f"  Overlay Path    : {result.raw_output_path}")
    if result.error_message:
        print(f"  Error Message   : {result.error_message}")
    print(f"{'='*70}")

    return {
        "task": task,
        "modality": modality,
        "status": result.status,
        "latency_sec": latency,
        "confidence": result.confidence,
        "model_used": result.model_used,
        "text_answer": result.text_answer,
        "evidence_count": len(result.spatial_evidence),
        "overlay_path": result.raw_output_path,
        "error_message": result.error_message,
    }


def run_full_evaluation(
    modalities_to_run: List[str],
    tasks_to_run: List[str],
    custom_query: str = None,
    threshold: float = 0.01,
) -> List[Dict[str, Any]]:
    """Runs all specified tasks and modalities, printing an end-of-run summary table."""
    images = get_or_create_test_images()
    results = []

    print("\n" + "#" * 70)
    print("  SatQuery AI Person 2 — Multi-Modal Model Evaluation Harness")
    print(f"  Modalities : {', '.join(modalities_to_run)}")
    print(f"  Tasks      : {', '.join(tasks_to_run)}")
    print("#" * 70)

    start_all = time.time()

    for mod in modalities_to_run:
        img_obj = images[mod]
        for t in tasks_to_run:
            res = test_single_task(
                task=t,
                modality=mod,
                image_obj=img_obj,
                query=custom_query,
                threshold=threshold,
            )
            results.append(res)

    total_time = round(time.time() - start_all, 2)

    # Print Final Markdown-Style Table
    print("\n\n" + "=" * 95)
    print("                      FINAL EVALUATION RESULTS TABLE")
    print("=" * 95)
    header = f"{'Modality':<9} | {'Task':<11} | {'Status':<7} | {'Latency':<8} | {'Conf':<6} | {'Evidence':<8} | {'Model'}"
    print(header)
    print("-" * 95)
    for r in results:
        conf_str = f"{r['confidence']:.3f}" if r["confidence"] is not None else "N/A"
        row = (
            f"{r['modality']:<9} | "
            f"{r['task']:<11} | "
            f"{r['status']:<7} | "
            f"{r['latency_sec']:<6}s | "
            f"{conf_str:<6} | "
            f"{r['evidence_count']:<8} | "
            f"{r['model_used'].split('/')[-1]}"
        )
        print(row)
    print("=" * 95)
    print(f"Total Multi-Modal Evaluation Duration: {total_time}s")

    # Save summary report to JSON
    reports_dir = os.path.join(OUTPUTS_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, f"evaluation_report_{int(time.time())}.json")
    with open(report_file, "w") as f:
        json.dump({"total_time_sec": total_time, "results": results}, f, indent=2)
    print(f"Structured JSON Report saved to: {report_file}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2 Model Test Suite (Optical & SAR, Individual & Together)")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["vqa", "captioning", "grounding", "all"],
        help="Task to test: vqa, captioning, grounding, or all (together)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="all",
        choices=["optical", "sar", "all"],
        help="Imagery modality: optical, sar, or all (both)",
    )
    parser.add_argument("--query", type=str, default=None, help="Custom query / prompt for testing")
    parser.add_argument("--threshold", type=float, default=0.01, help="Grounding confidence threshold")
    args = parser.parse_args()

    modalities = ["optical", "sar"] if args.modality == "all" else [args.modality]
    tasks = ["vqa", "captioning", "grounding"] if args.task == "all" else [args.task]

    run_full_evaluation(modalities, tasks, custom_query=args.query, threshold=args.threshold)
