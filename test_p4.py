import os
import sys
import json
import time
import requests
import shutil

sih26_dir = os.path.abspath(".")
if sih26_dir not in sys.path:
    sys.path.insert(0, sih26_dir)

from backend.app.validators.input_validator import validate, extract_metadata
from schemas import Modality

def test_p1_validation():
    print("--- Testing P1 Validator on Real Optical/SAR Pair ---")
    s1_path = "s1_aligned.tif"
    s2_path = "s2.tif"
    
    val_result = validate([s1_path, s2_path], "fuse")
    if not val_result.is_valid:
        print("P1 Validation Failed:", val_result.reason)
        return False
        
    print(f"P1 Validation Successful! Detected Config: {val_result.detected_config.value}")
    
    optical_found = False
    sar_found = False
    for m in val_result.images_meta:
        if m.modality == Modality.optical:
            optical_found = True
        elif m.modality == Modality.sar:
            sar_found = True
            
    if optical_found and sar_found:
        print("P1 correctly identified one OPTICAL and one SAR image.")
        return True
    else:
        print("P1 failed to detect both OPTICAL and SAR modalities.")
        return False

def test_fusion_api():
    print("\n--- Testing P4 Fusion API via P5 Agent ---")
    timestamp = int(time.time())
    run_dir = f"outputs/verification_run_p4_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    s1_path = "s1_aligned.tif"
    s2_path = "s2.tif"
    
    shutil.copy(s1_path, os.path.join(run_dir, "input_sar.tif"))
    shutil.copy(s2_path, os.path.join(run_dir, "input_optical.tif"))
    
    meta1 = extract_metadata(s1_path)
    meta2 = extract_metadata(s2_path)
    
    payload = {
        "query": "Identify the predominant land cover by fusing optical and sar together.",
        "images": [
            {
                "image_id": "s1_sar",
                "file_path": s1_path,
                "modality": meta1.modality.value,
                "format": meta1.format,
                "bands": 1,
                "resolution_m": 10.0,
                "crs": meta1.crs,
                "bbox": meta1.bounds,
                "width": meta1.width,
                "height": meta1.height
            },
            {
                "image_id": "s2_optical",
                "file_path": s2_path,
                "modality": meta2.modality.value,
                "format": meta2.format,
                "bands": 3,
                "resolution_m": 10.0,
                "crs": meta2.crs,
                "bbox": meta2.bounds,
                "width": meta2.width,
                "height": meta2.height
            }
        ]
    }
    
    resp = requests.post("http://localhost:8000/query", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        
        with open(os.path.join(run_dir, "execution_trace.json"), "w") as f:
            json.dump(data, f, indent=2)
            
        print("OPTICAL IMAGE: s2.tif, SAR IMAGE: s1.tif, Source: real co-registered data")
        print("QUERY: Identify the predominant land cover by fusing optical and sar together.")
        print(f"TASK CLASSIFIED AS: {data.get('execution_trace', {}).get('task_detected')}")
        print(f"MODEL USED: {data.get('execution_trace', {}).get('tools_selected', [''])[0]}")
        print(f"ANSWER: {data.get('answer')}")
        print(f"CONFIDENCE: {data.get('confidence')}")
        
        with open(os.path.join(run_dir, "pipeline_log.txt"), "w") as f:
            f.write("OPTICAL IMAGE: s2.tif, SAR IMAGE: s1.tif, Source: real co-registered data\n")
            f.write("QUERY: Identify the predominant land cover by fusing optical and sar together.\n")
            f.write(f"TASK CLASSIFIED AS: {data.get('execution_trace', {}).get('task_detected')}\n")
            f.write(f"MODEL USED: {data.get('execution_trace', {}).get('tools_selected', [''])[0]}\n")
            f.write(f"ANSWER: {data.get('answer')}\n")
            f.write(f"CONFIDENCE: {data.get('confidence')}\n")
            
        print(f"Check {run_dir} for output.")
    else:
        print(f"Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    if test_p1_validation():
        test_fusion_api()
