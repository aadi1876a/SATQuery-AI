import os
import sys
import json
import time
import requests
import random
import glob
import shutil

sih26_dir = os.path.abspath(".")
if sih26_dir not in sys.path:
    sys.path.insert(0, sih26_dir)

import rasterio
import numpy as np
from PIL import Image
from backend.app.validators.input_validator import extract_metadata
from schemas import Modality

def save_as_png(tif_path: str, png_path: str, is_sar: bool = False):
    with rasterio.open(tif_path) as src:
        if is_sar:
            data = src.read(1)
            vmin, vmax = data.min(), data.max()
            if vmax > vmin:
                data = (data - vmin) / (vmax - vmin) * 255.0
            img = Image.fromarray(data.astype(np.uint8), mode='L')
            img.save(png_path)
        else:
            bands = min(3, src.count)
            data = src.read(list(range(1, bands + 1)))
            data = np.transpose(data, (1, 2, 0))
            vmin, vmax = data.min(), data.max()
            if vmax > vmin:
                data = (data - vmin) / (vmax - vmin) * 255.0
            
            if bands == 1:
                img = Image.fromarray(data[:, :, 0].astype(np.uint8), mode='L')
            else:
                img = Image.fromarray(data.astype(np.uint8), mode='RGB')
            img.save(png_path)

def get_random_pairs(num_pairs=3):
    s1_dir = "data/bigearthnet_raw/BigEarthNet-S1/test"
    s2_dir = "data/bigearthnet_raw/BigEarthNet-S2/test"
    
    s1_files = glob.glob(os.path.join(s1_dir, "*.tif"))
    pairs = []
    
    for s1_path in s1_files:
        basename = os.path.basename(s1_path)
        parts = basename.split("_")
        if len(parts) >= 3:
            patch_id = "_".join(parts[-3:])
            s2_pattern = os.path.join(s2_dir, f"*{patch_id}")
            s2_matches = glob.glob(s2_pattern)
            if s2_matches:
                pairs.append((s1_path, s2_matches[0], patch_id))
                
    random.shuffle(pairs)
    return pairs[:num_pairs]

def test_fusion_random():
    pairs = get_random_pairs(5)
    if not pairs:
        print("No matched pairs found in BigEarthNet directories!")
        return
        
    print(f"Found {len(pairs)} random matched S1/S2 pairs for Fusion testing.")
    
    for i, (s1_path, s2_path, patch_id) in enumerate(pairs):
        print(f"\n--- Testing P4 Fusion API with Random Pair {i+1}/5 ---")
        print(f"Patch ID: {patch_id}")
        
        timestamp = int(time.time())
        run_dir = f"outputs/p4/run_{timestamp}_{patch_id.replace('.tif', '')}"
        os.makedirs(run_dir, exist_ok=True)
        
        # Save images as PNGs for viewing
        save_as_png(s1_path, os.path.join(run_dir, "input_sar.png"), is_sar=True)
        save_as_png(s2_path, os.path.join(run_dir, "input_optical.png"), is_sar=False)
        
        meta1 = extract_metadata(s1_path)
        meta2 = extract_metadata(s2_path)
        
        payload = {
            "query": "Identify the predominant land cover by fusing optical and sar together.",
            "images": [
                {
                    "image_id": f"sar_{patch_id}",
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
                    "image_id": f"optical_{patch_id}",
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
            print(f"ANSWER: {data.get('answer')}")
            print(f"CONFIDENCE: {data.get('confidence')}")
            print(f"Saved run data to: {run_dir}")
            
            with open(os.path.join(run_dir, "response.json"), "w") as f:
                json.dump(data, f, indent=2)
        else:
            print(f"Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    test_fusion_random()
