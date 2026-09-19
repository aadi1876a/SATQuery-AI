import os
import time
import json
import shutil
from pathlib import Path

# Add the project root to the PYTHONPATH
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from schemas import ImageObject, Modality
from agent.controller import run_query

import argparse
from backend.app.validators.input_validator import extract_metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/bigearthnet_subset/images/S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_80_58.png")
    parser.add_argument("--query-vqa", type=str, default="What land cover is visible in this satellite image?")
    parser.add_argument("--query-cap", type=str, default="Describe this scene.")
    args = parser.parse_args()
    
    print("===========================================================================")
    print(" 🧠 SatQuery AI - P2 VQA & Captioning End-to-End Verification")
    print("===========================================================================")
    
    # 1. Prepare Verification Folder
    timestamp = int(time.time())
    output_dir = Path(f"outputs/verification_run_p2_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Image
    img_path = args.image
    if not os.path.exists(img_path):
        print(f"Error: image {img_path} not found.")
        sys.exit(1)
    
    # Copy input image for proof
    ext = os.path.splitext(img_path)[1]
    output_img_path = output_dir / f"input_image{ext}"
    shutil.copy(img_path, output_img_path)

    # Use real metadata extraction for robust testing
    meta = extract_metadata(img_path)
    img_obj = ImageObject(
        image_id=meta.filename.split('.')[0],
        file_path=img_path,
        modality=Modality.optical,
        format=meta.format,
        bands=3,
        resolution_m=10.0,
        crs=meta.crs or "EPSG:32634",
        bbox=meta.bounds or [0, 0, 10, 10],
        width=meta.width,
        height=meta.height
    )
    
    # ==========================================
    # RUN 1: VQA
    # ==========================================
    query_vqa = args.query_vqa
    print(f"\n[RUN 1] VQA QUERY: {query_vqa}")
    vqa_resp = run_query(query_vqa, [img_obj])
    
    with open(output_dir / "execution_trace_vqa.json", "w") as f:
        json.dump(vqa_resp.model_dump(), f, indent=2)
        
    print("\n--- VQA RESULTS ---")
    print(f"INPUT IMAGE: {img_path}, Source: Real P1 BigEarthNet Data")
    print(f"QUERY: {query_vqa}")
    print(f"TASK CLASSIFIED AS: {vqa_resp.execution_trace.task_detected}")
    print(f"MODEL USED: {vqa_resp.execution_trace.tools_selected[0] if vqa_resp.execution_trace.tools_selected else 'None'}")
    print(f"ANSWER: {vqa_resp.answer}")
    print(f"CONFIDENCE: {vqa_resp.confidence}")

    # ==========================================
    # RUN 2: CAPTIONING
    # ==========================================
    query_cap = args.query_cap
    print(f"\n[RUN 2] CAPTIONING QUERY: {query_cap}")
    cap_resp = run_query(query_cap, [img_obj])
    
    with open(output_dir / "execution_trace_captioning.json", "w") as f:
        json.dump(cap_resp.model_dump(), f, indent=2)
        
    print("\n--- CAPTIONING RESULTS ---")
    print(f"INPUT IMAGE: {img_path}, Source: Real P1 BigEarthNet Data")
    print(f"QUERY: {query_cap}")
    print(f"TASK CLASSIFIED AS: {cap_resp.execution_trace.task_detected}")
    print(f"MODEL USED: {cap_resp.execution_trace.tools_selected[0] if cap_resp.execution_trace.tools_selected else 'None'}")
    print(f"ANSWER: {cap_resp.answer}")
    print(f"CONFIDENCE: {cap_resp.confidence}")

    # Write Pipeline Log
    with open(output_dir / "pipeline_log.txt", "w") as f:
        f.write("=== VQA PIPELINE LOG ===\n")
        f.write(f"Validation Result: {vqa_resp.execution_trace.input_validation.status}\n")
        f.write(f"Task Classified: {vqa_resp.execution_trace.task_detected}\n")
        f.write(f"Model Selected: {vqa_resp.execution_trace.tools_selected}\n")
        f.write(f"Answer: {vqa_resp.answer}\n\n")
        f.write("=== CAPTIONING PIPELINE LOG ===\n")
        f.write(f"Validation Result: {cap_resp.execution_trace.input_validation.status}\n")
        f.write(f"Task Classified: {cap_resp.execution_trace.task_detected}\n")
        f.write(f"Model Selected: {cap_resp.execution_trace.tools_selected}\n")
        f.write(f"Answer: {cap_resp.answer}\n")
        
    print(f"\nVerification artifacts saved to: {output_dir}")

if __name__ == "__main__":
    main()
