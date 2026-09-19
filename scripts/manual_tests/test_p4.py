import os
import sys
import json
import time
import requests
import shutil

sih26_dir = os.path.abspath(".")
if sih26_dir not in sys.path:
    sys.path.insert(0, sih26_dir)

import rasterio
import numpy as np
from PIL import Image
from backend.app.validators.input_validator import validate, extract_metadata
from schemas import Modality

def save_as_png(tif_path: str, png_path: str, is_sar: bool = False):
    with rasterio.open(tif_path) as src:
        if is_sar:
            data = src.read(1)
            # Normalize to 0-255
            vmin, vmax = data.min(), data.max()
            if vmax > vmin:
                data = (data - vmin) / (vmax - vmin) * 255.0
            img = Image.fromarray(data.astype(np.uint8), mode='L')
            img.save(png_path)
        else:
            bands = min(3, src.count)
            data = src.read(list(range(1, bands + 1)))
            data = np.transpose(data, (1, 2, 0)) # (C, H, W) -> (H, W, C)
            vmin, vmax = data.min(), data.max()
            if vmax > vmin:
                data = (data - vmin) / (vmax - vmin) * 255.0
            
            if bands == 1:
                img = Image.fromarray(data[:, :, 0].astype(np.uint8), mode='L')
            else:
                img = Image.fromarray(data.astype(np.uint8), mode='RGB')
            img.save(png_path)

def test_p1_validation():
    print("--- Testing P1 Validator on Real Optical/SAR Pair ---")
    s1_path = "samples/fusion/real_s1.tif"
    s2_path = "samples/fusion/real_s2.tif"
    
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

def test_p1_reject_invalid_pair():
    print("--- Testing P1 Validator on INVALID (non-overlapping) Pair ---")
    s1_path = "data/bigearthnet_raw/BigEarthNet-S1/test/S1A_IW_GRDH_1SDV_20170826T163327_34TEQ_32_67.tif"
    s2_path = "samples/fusion/real_s2.tif" # Different location entirely
    
    val_result = validate([s1_path, s2_path], "fuse")
    if val_result.is_valid:
        print("FAIL: P1 Validator allowed an invalid pair!")
        return False
    else:
        print("P1 correctly rejected the invalid pair. Reason:", val_result.reason)
        return True

def test_fusion_api():
    print("\n--- Testing P4 Fusion API via P5 Agent ---")
    timestamp = int(time.time())
    run_dir = f"outputs/p4/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    s1_path = "samples/fusion/real_s1.tif"
    s2_path = "samples/fusion/real_s2.tif"
    
    save_as_png(s1_path, os.path.join(run_dir, "input_sar.png"), is_sar=True)
    save_as_png(s2_path, os.path.join(run_dir, "input_optical.png"), is_sar=False)
    
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
        
        with open(os.path.join(run_dir, "response.json"), "w") as f:
            json.dump(data, f, indent=2)
            
        print("OPTICAL IMAGE: real_s2.tif, SAR IMAGE: real_s1.tif, Source: real co-registered data")
        print("QUERY: Identify the predominant land cover by fusing optical and sar together.")
        print(f"TASK CLASSIFIED AS: {data.get('execution_trace', {}).get('task_detected')}")
        print(f"MODEL USED: {data.get('execution_trace', {}).get('tools_selected', [''])[0]}")
        print(f"ANSWER: {data.get('answer')}")
        print(f"CONFIDENCE: {data.get('confidence')}")
        
        with open(os.path.join(run_dir, "pipeline_log.txt"), "w") as f:
            f.write("OPTICAL IMAGE: real_s2.tif, SAR IMAGE: real_s1.tif, Source: real co-registered data\n")
            f.write("QUERY: Identify the predominant land cover by fusing optical and sar together.\n")
            f.write(f"TASK CLASSIFIED AS: {data.get('execution_trace', {}).get('task_detected')}\n")
            f.write(f"MODEL USED: {data.get('execution_trace', {}).get('tools_selected', [''])[0]}\n")
            f.write(f"ANSWER: {data.get('answer')}\n")
            f.write(f"CONFIDENCE: {data.get('confidence')}\n")
            
        print(f"Check {run_dir} for output.")
    else:
        print(f"Error {resp.status_code}: {resp.text}")

def test_fusion_api_second_pair():
    print("\n--- Testing P4 Fusion API with SECOND Pair ---")
    timestamp = int(time.time())
    run_dir = f"outputs/p4/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    s1_path = "data/bigearthnet_raw/BigEarthNet-S1/test/S1A_IW_GRDH_1SDV_20170802T163350_34TCR_44_41.tif"
    s2_path = "data/bigearthnet_raw/BigEarthNet-S2/test/S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_44_41.tif"
    
    save_as_png(s1_path, os.path.join(run_dir, "input_sar.png"), is_sar=True)
    save_as_png(s2_path, os.path.join(run_dir, "input_optical.png"), is_sar=False)
    
    meta1 = extract_metadata(s1_path)
    meta2 = extract_metadata(s2_path)
    
    payload = {
        "query": "Identify the predominant land cover by fusing optical and sar together.",
        "images": [
            {
                "image_id": "s1_sar2",
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
                "image_id": "s2_optical2",
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
        
        with open(os.path.join(run_dir, "response.json"), "w") as f:
            json.dump(data, f, indent=2)
        print("OPTICAL IMAGE: 44_41, SAR IMAGE: 44_41, Source: real co-registered data")
        print(f"ANSWER: {data.get('answer')}")
        print(f"CONFIDENCE: {data.get('confidence')}")
    else:
        print(f"Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    passed_p1 = test_p1_validation()
    passed_reject = test_p1_reject_invalid_pair()
    if passed_p1 and passed_reject:
        test_fusion_api()
        test_fusion_api_second_pair()
