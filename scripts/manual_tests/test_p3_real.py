import os
import json
from PIL import Image
from schemas import ToolInput, ImageObject, TaskType, Modality
from p3_model import P3ChangeDetectionEngine

def run_test_on_preprocessed_data():
    dataset_path = "data/bigearthnet_subset/dataset.jsonl"
    base_dir = "data/bigearthnet_subset"
    
    with open(dataset_path, "r") as f:
        lines = f.readlines()
        
    if len(lines) < 2:
        print("Not enough images in dataset.")
        return
        
    data1 = json.loads(lines[0])
    data2 = json.loads(lines[1])
    
    t1_path = os.path.join(base_dir, data1["image_path"])
    t2_path = os.path.join(base_dir, data2["image_path"])
    
    print(f"Testing with image 1: {t1_path}")
    print(f"Testing with image 2: {t2_path}")
    
    img1 = ImageObject(
        image_id=data1["id"],
        file_path=t1_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.1000, 28.6000, 77.1100, 28.6100],  # Dummy bbox
        acquisition_date="2020-01-01",
        width=224,
        height=224
    )

    img2 = ImageObject(
        image_id=data2["id"],
        file_path=t2_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.1000, 28.6000, 77.1100, 28.6100],  # Dummy bbox
        acquisition_date="2021-01-01",
        width=224,
        height=224
    )

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="What has changed between these two images?",
        images=[img1, img2]
    )

    out_dir = "outputs/bigearthnet_p3_test"
    os.makedirs(out_dir, exist_ok=True)
    engine = P3ChangeDetectionEngine(output_dir=out_dir)
    output = engine.run(tool_input)

    print(f"Status: {output.status}")
    print(f"Text Answer:\n  {output.text_answer}")
    print(f"Overlay Image Output: {output.raw_output_path}")

    bbox_evidences = [ev for ev in output.spatial_evidence if ev.type == "bbox"]
    print(f"Detected Bounding Boxes Count: {len(bbox_evidences)}")
    for idx, ev in enumerate(bbox_evidences, 1):
        print(f"  Box #{idx} ({ev.label}): normalized coords = {ev.coords}")
        
if __name__ == "__main__":
    run_test_on_preprocessed_data()
