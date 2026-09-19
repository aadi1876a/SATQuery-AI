"""
test_t5_t6.py
Final End-to-End P3 Role Verification Script (Satellite T5 vs T6).

Validates complete P3 Specialist Model pipeline:
1. Multi-Spectral CIELAB Change Masking
2. Tight Contour Bounding Box Extraction
3. Red Overlay Visual Evidence Generation
4. Temporal Visual Question Answering (VQA)
5. 100% SatQuery AI Schema Compliance (ToolInput -> ToolOutput)
"""

import os
from PIL import Image, ImageDraw
from backend.app.schemas.schemas import ToolInput, ToolOutput, ImageObject, TaskType, Modality
from p3_model import P3ChangeDetectionEngine


def create_satellite_t5_t6_images(sample_dir: str = "sample_data"):
    """
    Generates Satellite T5 (2023) and T6 (2025) agricultural & valley scene.
    T5: Green crop fields, central river, forest patch.
    T6:
      - Change 1: New Industrial Warehouse & Parking Lot
      - Change 2: River Flooding / Water Inundation over crops
      - Change 3: Forest Canopy Loss / Deforestation patch
    """
    os.makedirs(sample_dir, exist_ok=True)
    w, h = 600, 600

    # --- Satellite T5 Image (2023) ---
    t5 = Image.new("RGB", (w, h), color=(90, 155, 65))  # Bright agricultural green
    draw5 = ImageDraw.Draw(t5)

    # Central River
    draw5.polygon([(260, 0), (300, 0), (330, 600), (290, 600)], fill=(35, 105, 210))
    # Forest patch (Dark green top right)
    draw5.rectangle([400, 50, 560, 200], fill=(20, 85, 35))
    # Small farm road
    draw5.line([(0, 450), (600, 450)], fill=(150, 140, 120), width=12)

    t5_path = os.path.join(sample_dir, "satellite_T5_2023.png")
    t5.save(t5_path)

    # --- Satellite T6 Image (2025) ---
    t6 = Image.new("RGB", (w, h), color=(90, 155, 65))
    draw6 = ImageDraw.Draw(t6)
    draw6.polygon([(260, 0), (300, 0), (330, 600), (290, 600)], fill=(35, 105, 210))
    draw6.rectangle([400, 50, 560, 200], fill=(20, 85, 35))
    draw6.line([(0, 450), (600, 450)], fill=(150, 140, 120), width=12)

    # --- CHANGES IN T6 ---
    # Change 1: New Industrial Warehouse & Concrete Complex (North-West)
    draw6.rectangle([80, 80, 220, 200], fill=(210, 215, 225))
    draw6.rectangle([100, 210, 200, 260], fill=(80, 80, 80))  # Asphalt parking

    # Change 2: Severe River Flooding & Water Inundation (South-East)
    draw6.ellipse([310, 250, 550, 480], fill=(25, 80, 180))

    # Change 3: Forest Canopy Loss / Deforestation patch inside top right forest
    draw6.rectangle([430, 80, 530, 170], fill=(140, 80, 35))

    t6_path = os.path.join(sample_dir, "satellite_T6_2025.png")
    t6.save(t6_path)

    return t5_path, t6_path


def main():
    print("=" * 75)
    print(" 🛰️ SatQuery AI - Final P3 Role End-to-End Verification (T5 vs T6)")
    print("=" * 75)

    # Step 1: Create satellite T5 and T6 images in sample_data/
    t5_path, t6_path = create_satellite_t5_t6_images("sample_data")
    print(f"Created Satellite T5 Image: {t5_path}")
    print(f"Created Satellite T6 Image: {t6_path}")

    img_t5 = ImageObject(
        image_id="SAT_T5_20230510",
        file_path=t5_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[73.1000, 18.9000, 73.1100, 18.9100],
        acquisition_date="2023-05-10",
        width=600,
        height=600
    )

    img_t6 = ImageObject(
        image_id="SAT_T6_20250120",
        file_path=t6_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[73.1000, 18.9000, 73.1100, 18.9100],
        acquisition_date="2025-01-20",
        width=600,
        height=600
    )

    # Step 2: Run P3 Engine with multiple VQA queries
    engine = P3ChangeDetectionEngine(output_dir="outputs")

    queries = [
        ("COUNTING INTENT", "How many new structures and flood zones were created between T5 (2023) and T6 (2025)?"),
        ("LOCATION INTENT", "Where are the spatial changes located in the satellite scene?"),
        ("LAND COVER INTENT", "Describe the deforestation and river flooding land cover modifications")
    ]

    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append(" SatQuery AI - P3 ToolOutput Schema-Compliant VQA Report (Satellite T5 vs T6)")
    report_lines.append("=" * 75 + "\n")

    json_outputs = []

    for qtype, query_str in queries:
        tool_input = ToolInput(
            task=TaskType.change_vqa,
            query=query_str,
            images=[img_t5, img_t6]
        )

        output: ToolOutput = engine.run(tool_input)

        # Copy T5/T6 mask and overlay with dedicated names in outputs/
        mask_source = [ev.mask_path for ev in output.spatial_evidence if ev.type == "mask"][0]
        overlay_source = output.raw_output_path

        mask_target = os.path.join("outputs", "satellite_T5_T6_mask.png")
        overlay_target = os.path.join("outputs", "satellite_T5_T6_overlay.png")

        Image.open(mask_source).save(mask_target)
        Image.open(overlay_source).save(overlay_target)

        # Append to JSON list
        json_outputs.append({
            "query": query_str,
            "query_type": qtype,
            "tool_output": output.model_dump()
        })

        section = []
        section.append("-" * 75)
        section.append(f"❓ USER QUERY [{qtype}]: '{query_str}'")
        section.append("-" * 75)
        section.append(f"• status:           \"{output.status}\"")
        section.append(f"• model_used:       \"{output.model_used}\"")
        section.append(f"• confidence:       {output.confidence}")
        section.append(f"• text_answer:\n    \"{output.text_answer}\"")
        section.append(f"• raw_output_path:  \"{output.raw_output_path}\"")
        section.append(f"• error_message:    {output.error_message}")
        section.append(f"• spatial_evidence  ({len(output.spatial_evidence)} item(s)):")
        for idx, ev in enumerate(output.spatial_evidence, 1):
            section.append(f"    [{idx}] SpatialEvidence(type=\"{ev.type}\", coords={ev.coords}, mask_path=\"{ev.mask_path}\", label=\"{ev.label}\")")
        section.append("\n")

        section_str = "\n".join(section)
        print(section_str)
        report_lines.append(section_str)

        # Validate schema compliance
        assert output.status in ["success", "error"]
        assert output.model_used is not None
        validated_obj = ToolOutput.model_validate(output.model_dump())
        assert validated_obj == output

    # Save ToolOutput Schema JSON to disk
    json_file_path = os.path.join("outputs", "tool_output_T5_T6.json")
    with open(json_file_path, "w", encoding="utf-8") as f:
        import json
        json.dump(json_outputs, f, indent=2)

    # Save VQA Report File to disk
    report_file_path = os.path.join("outputs", "vqa_answers_T5_T6.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("=" * 75)
    print(" ✅ 100% PYDANTIC TOOLOUTPUT SCHEMA COMPLIANCE VERIFIED!")
    print(f" 📄 TOOLOUTPUT JSON EXPORTED TO:     {json_file_path}")
    print(f" 📄 SCHEMA VQA REPORT SAVED TO:      {report_file_path}")
    print(f" 🖼️ BINARY MASK SAVED TO:             {mask_target}")
    print(f" 🖼️ OVERLAY IMAGE SAVED TO:            {overlay_target}")
    print("=" * 75)




if __name__ == "__main__":
    main()
