"""
test_p3_vqa.py
Comprehensive test suite for P3 Temporal VQA System.

Tests query intent parsing, spatial-spectral reasoning, and natural language
answer generation across multiple query types (Counting, Location, Land Cover, Summary).
"""

from schemas import ToolInput, ImageObject, TaskType, Modality
from p3_model import P3ChangeDetectionEngine
from test_t3_t4 import create_satellite_t3_t4_images


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


def main():
    print("=" * 75)
    print(" 🧠 SatQuery AI - P3 Temporal VQA Engine Test Suite")
    print("=" * 75)

    # Prepare Satellite T3 and T4 test images
    t3_path, t4_path = create_satellite_t3_t4_images("sample_data")

    img_t3 = ImageObject(
        image_id="SAT_T3",
        file_path=t3_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=5.0,
        crs="EPSG:4326",
        bbox=[72.85, 19.05, 72.86, 19.06],
        acquisition_date="2022-06-15",
        width=600,
        height=600
    )

    img_t4 = ImageObject(
        image_id="SAT_T4",
        file_path=t4_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=5.0,
        crs="EPSG:4326",
        bbox=[72.85, 19.05, 72.86, 19.06],
        acquisition_date="2024-09-10",
        width=600,
        height=600
    )

    engine = P3ChangeDetectionEngine(output_dir="outputs")

    # 1. Counting Intent
    test_vqa_query(
        engine, img_t3, img_t4,
        "How many new structures and fuel tanks were constructed between 2022 and 2024?",
        "COUNTING INTENT"
    )

    # 2. Location Intent
    test_vqa_query(
        engine, img_t3, img_t4,
        "Where are the spatial changes located in the satellite image?",
        "LOCATION INTENT"
    )

    # 3. Land Cover / Disaster Intent
    test_vqa_query(
        engine, img_t3, img_t4,
        "What type of land cover modifications or coastal erosion occurred?",
        "LAND COVER INTENT"
    )

    # 4. General Temporal VQA Summary
    test_vqa_query(
        engine, img_t3, img_t4,
        "Summarize all changes that happened between image T3 and T4.",
        "GENERAL SUMMARY INTENT"
    )

    print("\n" + "=" * 75)
    print(" 🎉 ALL TEMPORAL VQA ENGINE TESTS PASSED PERFECTLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
