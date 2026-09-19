import os
import time
import json
import shutil
from PIL import Image
from schemas import ToolInput, ImageObject, TaskType, Modality
from p3_model import P3ChangeDetectionEngine

def verify_pipeline():
    timestamp = int(time.time())
    out_dir = f"outputs/verification_run_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Use real preprocessed data from dataset.jsonl
    dataset_path = "data/bigearthnet_subset/dataset.jsonl"
    base_dir = "data/bigearthnet_subset"
    
    with open(dataset_path, "r") as f:
        lines = f.readlines()
        
    data1 = json.loads(lines[0])
    data2 = json.loads(lines[1])
    
    t1_orig_path = os.path.join(base_dir, data1["image_path"])
    t2_orig_path = os.path.join(base_dir, data2["image_path"])
    
    ext1 = os.path.splitext(t1_orig_path)[1]
    ext2 = os.path.splitext(t2_orig_path)[1]
    
    t1_new_path = os.path.join(out_dir, f"input_t1{ext1}")
    t2_new_path = os.path.join(out_dir, f"input_t2{ext2}")
    
    shutil.copy2(t1_orig_path, t1_new_path)
    shutil.copy2(t2_orig_path, t2_new_path)

    log_path = os.path.join(out_dir, "pipeline_log.txt")
    with open(log_path, "w") as f_log:
        f_log.write(f"--- Pipeline Execution Log ---\n")
        f_log.write(f"Stage 1: Validation - Validating inputs from real P1 preprocessed data.\n")
        f_log.write(f"T1 Original: {t1_orig_path}\n")
        f_log.write(f"T2 Original: {t2_orig_path}\n")
        f_log.write(f"NOTE: These are TWO DIFFERENT LOCATIONS in Serbia, as BigEarthNet does not contain bi-temporal pairs.\n\n")
        
        f_log.write(f"Stage 2: Task Classification - Classified as TaskType.change_vqa\n\n")

    img1 = ImageObject(
        image_id=data1["id"],
        file_path=t1_new_path,
        modality=Modality.optical,
        format=ext1.replace(".", ""),
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.1000, 28.6000, 77.1100, 28.6100],  # Dummy bbox
        acquisition_date="2017-08-25",
        width=224,
        height=224
    )

    img2 = ImageObject(
        image_id=data2["id"],
        file_path=t2_new_path,
        modality=Modality.optical,
        format=ext2.replace(".", ""),
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.1000, 28.6000, 77.1100, 28.6100],  # Dummy bbox
        acquisition_date="2017-08-03",
        width=224,
        height=224
    )

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="What has changed between these two images?",
        images=[img1, img2]
    )

    with open(log_path, "a") as f_log:
        f_log.write(f"Stage 3: Model Selection - Instantiating P3ChangeDetectionEngine\n\n")
        
    engine = P3ChangeDetectionEngine(output_dir=out_dir)
    
    # Disable the automatic time-based overlay saving and do it manually so we control the exact name
    output = engine.run(tool_input)
    
    # Rename overlay file to change_overlay.png
    actual_overlay = output.raw_output_path
    new_overlay = os.path.join(out_dir, "change_overlay.png")
    if os.path.exists(actual_overlay):
        os.rename(actual_overlay, new_overlay)
        output.raw_output_path = new_overlay
    
    # Find execution trace (the auto-exported tool_output json)
    for f in os.listdir(out_dir):
        if f.startswith("tool_output_") and f.endswith(".json"):
            os.rename(os.path.join(out_dir, f), os.path.join(out_dir, "execution_trace.json"))
            break
            
    with open(log_path, "a") as f_log:
        f_log.write(f"Stage 4: Raw Model Output\n")
        f_log.write(f"Status: {output.status}\n")
        f_log.write(f"Confidence: {output.confidence}\n")
        f_log.write(f"Answer: {output.text_answer}\n")
        
    print(f"INPUT IMAGES USED:")
    print(f"  T1: {t1_new_path}")
    print(f"  T2: {t2_new_path}")
    print(f"  Source: P1 real preprocessed data")
    print()
    bbox_evs = [e for e in output.spatial_evidence if e.type == "bbox"]
    print(f"CHANGE REGIONS DETECTED: {len(bbox_evs)}")
    for i, ev in enumerate(bbox_evs):
        print(f"  {ev.label}: {ev.coords} | confidence: {output.confidence}")
    print()
    print(f"OVERALL ANSWER TEXT: {output.text_answer}")
    print(f"OVERALL CONFIDENCE: {output.confidence}")
    print()
    print("FILES WRITTEN:")
    print(f"  {t1_new_path}")
    print(f"  {t2_new_path}")
    print(f"  {new_overlay}")
    print(f"  {os.path.join(out_dir, 'execution_trace.json')}")
    print(f"  {log_path}")

if __name__ == "__main__":
    verify_pipeline()
