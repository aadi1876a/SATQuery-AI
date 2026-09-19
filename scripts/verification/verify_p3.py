import shutil
import os
import sys
from PIL import Image

brain_dir = r"C:\Users\kamda_k\.gemini\antigravity\brain\b682aa20-f47f-4598-82e4-56bfb07c37df"
t1_src = os.path.join(brain_dir, "media__1789758046193.jpg")
t2_src = os.path.join(brain_dir, "media__1789758054743.jpg")

target_dir = os.path.abspath("sih26/sample_data")
os.makedirs(target_dir, exist_ok=True)
t1_dst = os.path.join(target_dir, "nirma_T1.jpg")
t2_dst = os.path.join(target_dir, "nirma_T2.jpg")

shutil.copy(t1_src, t1_dst)
shutil.copy(t2_src, t2_dst)

im1 = Image.open(t1_dst)
im2 = Image.open(t2_dst)
print(f"T1 image size: {im1.size}")
print(f"T2 image size: {im2.size}")

sih26_dir = os.path.abspath("sih26")
if sih26_dir not in sys.path:
    sys.path.insert(0, sih26_dir)

from schemas import ToolInput, ImageObject, TaskType, Modality
from backend.models.change_detection.inference import call_change_model

img_t1 = ImageObject(
    image_id="NIRMA_T1",
    file_path=t1_dst,
    modality=Modality.optical,
    format="jpg",
    bands=3,
    resolution_m=0.5,
    crs="EPSG:4326",
    bbox=[72.548, 23.128, 72.558, 23.138],
    acquisition_date="Before (T1)",
    width=im1.width,
    height=im1.height
)

img_t2 = ImageObject(
    image_id="NIRMA_T2",
    file_path=t2_dst,
    modality=Modality.optical,
    format="jpg",
    bands=3,
    resolution_m=0.5,
    crs="EPSG:4326",
    bbox=[72.548, 23.128, 72.558, 23.138],
    acquisition_date="After (T2)",
    width=im2.width,
    height=im2.height
)

tool_input = ToolInput(
    task=TaskType.change_vqa,
    query="What infrastructure, land cover, or building changes occurred between these two satellite images of Nirma University?",
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
