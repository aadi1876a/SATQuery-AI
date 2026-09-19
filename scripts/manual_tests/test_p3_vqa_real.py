import os
import json
from PIL import Image
from backend.app.schemas.schemas import ToolInput, ImageObject, TaskType, Modality
from p3_model import P3ChangeDetectionEngine

def test_vqa_query(engine: P3ChangeDetectionEngine, img1: ImageObject, img2: ImageObject, query: str, query_type_name: str):
    print("\n" + "-" * 70)
    print(f" ❓ QUERY ({query_type_name}): '{query}'")
    print("-" * 70)

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query=query,
        images=[img1, img2]
    )

    output = engine.run(tool_input)

    print(f"Status: {output.status}")
    print(f"Model Used: {output.model_used}")
    print(f"Confidence: {output.confidence}")
    print(f"💬 VQA Text Answer:\n  {output.text_answer}")
    assert output.status == "success"
    assert output.text_answer is not None and len(output.text_answer) > 20
    print(f"✅ VQA Test for '{query_type_name}' PASSED!")


def run_test_vqa_on_real_data():
    dataset_path = "data/bigearthnet_subset/dataset.jsonl"
    base_dir = "data/bigearthnet_subset"
    
    print("=" * 75)
    print(" 🧠 SatQuery AI - P3 Temporal VQA Engine Test Suite (Real P1 Data)")
    print("=" * 75)
    
    import random
    with open(dataset_path, "r") as f:
        lines = f.readlines()
        
    if len(lines) < 2:
        print("Not enough images in dataset.")
        return
        
    sampled_lines = random.sample(lines, 2)
    data1 = json.loads(sampled_lines[0])
    data2 = json.loads(sampled_lines[1])
    
    t1_path = os.path.join(base_dir, data1["image_path"])
    t2_path = os.path.join(base_dir, data2["image_path"])
    
    print(f"Loading real P1 image T1: {t1_path}")
    print(f"Loading real P1 image T2: {t2_path}")
    
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
    
    engine = P3ChangeDetectionEngine(output_dir="outputs/vqa_real_test")
    
    # Test Queries
    queries = [
        ("Summarize all changes between these two dates.", "Summary"),
        ("How many regions show changes?", "Counting"),
        ("Where are the changes located?", "Location"),
        ("What type of land cover change occurred?", "Land Cover"),
    ]
    
    for query, q_type in queries:
        test_vqa_query(engine, img1, img2, query, q_type)

    print("\n" + "=" * 75)
    print(" 🎉 ALL VQA TESTS ON REAL P1 DATA PASSED!")
    print("=" * 75)

if __name__ == "__main__":
    run_test_vqa_on_real_data()
