import os
import sys
import glob
import random
from PIL import Image

sih26_dir = os.path.abspath(".")
if sih26_dir not in sys.path:
    sys.path.insert(0, sih26_dir)

from backend.app.schemas.schemas import ToolInput, ImageObject, TaskType, Modality
from backend.models.change_detection.inference import call_change_model

# Get all S2 images
s2_images = glob.glob("data/bigearthnet_subset/images/S2*.png")
if len(s2_images) < 2:
    print("Not enough images in data to run verification!")
    sys.exit(1)

t1_dst, t2_dst = random.sample(s2_images, 2)

im1 = Image.open(t1_dst)
im2 = Image.open(t2_dst)
print(f"T1 image ({os.path.basename(t1_dst)}) size: {im1.size}")
print(f"T2 image ({os.path.basename(t2_dst)}) size: {im2.size}")

img_t1 = ImageObject(
    image_id=os.path.basename(t1_dst),
    file_path=t1_dst,
    modality=Modality.optical,
    format="png",
    bands=3,
    resolution_m=10.0,
    crs="EPSG:32634",
    bbox=[0, 0, 1, 1], # Dummy bbox
    acquisition_date="Before (T1)",
    width=im1.width,
    height=im1.height
)

img_t2 = ImageObject(
    image_id=os.path.basename(t2_dst),
    file_path=t2_dst,
    modality=Modality.optical,
    format="png",
    bands=3,
    resolution_m=10.0,
    crs="EPSG:32634",
    bbox=[0, 0, 1, 1], # Dummy bbox
    acquisition_date="After (T2)",
    width=im2.width,
    height=im2.height
)

tool_input = ToolInput(
    task=TaskType.change_vqa,
    query="What infrastructure, land cover, or building changes occurred between these two satellite images?",
    images=[img_t1, img_t2]
)

output = call_change_model(tool_input)

print("\n================ P3 TOOL OUTPUT RESULTS ================")
print("Status:", output.status)
print("Model Used:", output.model_used)
print("Confidence Score:", output.confidence)
print("Visual Overlay Saved at:", output.raw_output_path)
print("Spatial Evidence Items Detected:", len(output.spatial_evidence))
print("\n--- GENERATED TEXT ANSWER ---")
print(output.text_answer)
print("\n--- DETECTED SPATIAL EVIDENCE ---")
for ev in output.spatial_evidence:
    print(f"Type: {ev.type} | Label: {ev.label} | Coords: {ev.coords} | Mask: {ev.mask_path}")

# Copy T1 and T2 images to the output run folder
import shutil
if output.raw_output_path:
    run_dir = os.path.dirname(output.raw_output_path)
    shutil.copy(t1_dst, os.path.join(run_dir, "input_T1.png"))
    shutil.copy(t2_dst, os.path.join(run_dir, "input_T2.png"))
    print(f"Copied input images to: {run_dir}")
