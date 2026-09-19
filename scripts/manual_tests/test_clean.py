import os
import shutil
import json
import time
import random
import glob
import argparse
import requests
from pathlib import Path
from PIL import Image, ImageDraw
from backend.app.schemas.schemas import ImageObject, Modality
from backend.app.validators.input_validator import extract_metadata

# Get all images for random selection
ALL_IMAGES = glob.glob("data/bigearthnet_subset/images/*.png")

def create_image_object(filepath: str) -> dict:
    meta = extract_metadata(filepath)
    return ImageObject(
        image_id=meta.filename.split('.')[0],
        file_path=filepath,
        modality=Modality.optical,
        format=meta.format,
        bands=3,
        resolution_m=10.0,
        crs=meta.crs or "EPSG:32634",
        bbox=meta.bounds or [0, 0, 10, 10],
        width=meta.width,
        height=meta.height
    ).model_dump()

def run_p2_test(query: str):
    print("\n--- Running P2 Test (Single Image VQA) ---")
    img_path = random.choice(ALL_IMAGES)
    print(f"Selected Image: {img_path}")
    
    # Setup Output Dir
    timestamp = int(time.time())
    out_dir = Path(f"outputs/p2/run_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy Image
    ext = os.path.splitext(img_path)[1]
    saved_img_path = out_dir / f"input_image{ext}"
    shutil.copy(img_path, saved_img_path)
    
    # Run API
    img_obj = create_image_object(img_path)
    payload = {"query": query, "images": [img_obj]}
    
    resp = requests.post("http://localhost:8000/query", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Success! Answer: {data.get('answer')}")
        
        # Draw bounding boxes if present (Grounding)
        visual_evidence = data.get("visual_evidence", [])
        if visual_evidence:
            try:
                img = Image.open(saved_img_path)
                draw = ImageDraw.Draw(img)
                for item in visual_evidence:
                    if item.get("type") == "bbox":
                        coords = item.get("coords")
                        if len(coords) == 4:
                            # Draw rectangle: [xmin, ymin, xmax, ymax]
                            draw.rectangle(coords, outline="red", width=3)
                            # Draw label if available
                            label = item.get("label")
                            if label:
                                draw.text((coords[0], coords[1] - 10), label, fill="red")
                
                marked_img_path = out_dir / f"marked_image{ext}"
                img.save(marked_img_path)
                print(f"Grounding overlay saved to {marked_img_path}")
            except Exception as e:
                print(f"Could not draw bounding box: {e}")
                
        # Save JSON
        with open(out_dir / "response.json", "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Check {out_dir} for the saved image and JSON response.")
    else:
        print(f"Error {resp.status_code}: {resp.text}")

def run_p3_test(query: str):
    print("\n--- Running P3 Test (Bi-Temporal Change Detection) ---")
    img1_path, img2_path = random.sample(ALL_IMAGES, 2)
    print(f"Selected T1 Image: {img1_path}")
    print(f"Selected T2 Image: {img2_path}")
    
    # Setup Output Dir
    timestamp = int(time.time())
    out_dir = Path(f"outputs/p3/run_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy Images
    ext1 = os.path.splitext(img1_path)[1]
    ext2 = os.path.splitext(img2_path)[1]
    shutil.copy(img1_path, out_dir / f"input_image_t1{ext1}")
    shutil.copy(img2_path, out_dir / f"input_image_t2{ext2}")
    
    # Run API
    img1_obj = create_image_object(img1_path)
    img2_obj = create_image_object(img2_path)
    payload = {"query": query, "images": [img1_obj, img2_obj]}
    
    resp = requests.post("http://localhost:8000/query", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Success! Answer: {data.get('answer')}")
        
        # Save JSON
        with open(out_dir / "response.json", "w") as f:
            json.dump(data, f, indent=2)
            
        # P3 generates an output mask image in its own 'outputs/' folder by default.
        # We need to find it from the visual_evidence and move it into our clean folder.
        for ev in data.get("visual_evidence", []):
            if ev.get("mask_path") and os.path.exists(ev.get("mask_path")):
                target_path = out_dir / os.path.basename(ev["mask_path"])
                try:
                    shutil.copy(ev["mask_path"], target_path)
                    print(f"Mask saved to: {target_path}")
                except shutil.SameFileError:
                    print(f"Mask already perfectly aligned in: {target_path}")
                
        print(f"Check {out_dir} for the saved images and JSON response.")
    else:
        print(f"Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean API test script")
    parser.add_argument("--query-p2", type=str, default="What land cover is visible in this satellite image?", help="Query to run for the P2 (single image) test")
    parser.add_argument("--query-p3", type=str, default="What changed between these two dates?", help="Query to run for the P3 (two images) test")
    
    args = parser.parse_args()
    
    run_p2_test(args.query_p2)
    run_p3_test(args.query_p3)
