"""
test_t3_t4.py
Satellite T3 & T4 Test Suite for P3 Change Detection Engine.

Generates satellite_T3_2022.png (T3 Before) and satellite_T4_2024.png (T4 After)
in sample_data/, executes P3 engine, and saves:
- sample_data/satellite_T3_T4_mask.png (Binary Change Mask)
- sample_data/satellite_T3_T4_overlay.png (T4 with Red Bounding Boxes & Change Highlights)
"""

import os
from PIL import Image, ImageDraw
from schemas import ToolInput, ImageObject, TaskType, Modality
from p3_model import P3ChangeDetectionEngine


def create_satellite_t3_t4_images(sample_dir: str = "sample_data"):
    """
    Generates Satellite T3 (Before) and T4 (After) coastal port scene.
    T3: Coastal bay, harbor line, green hinterland, small pier.
    T4:
      - Change 1: Industrial dock extension built into bay
      - Change 2: 3 fuel storage tanks added near harbor
      - Change 3: Coastal land erosion patch near cliffs
    """
    os.makedirs(sample_dir, exist_ok=True)
    w, h = 600, 600

    # --- Satellite T3 Image (2022) ---
    t3 = Image.new("RGB", (w, h), color=(40, 110, 60))  # Green hinterland
    draw3 = ImageDraw.Draw(t3)

    # Ocean / Coastal Bay (Blue water occupying right side)
    draw3.polygon([(300, 0), (600, 0), (600, 600), (350, 600)], fill=(30, 90, 190))

    # Port pier / coastline
    draw3.rectangle([250, 200, 310, 240], fill=(160, 160, 160))
    # Small coastal road
    draw3.line([(100, 0), (100, 600)], fill=(120, 120, 120), width=15)

    t3_path = os.path.join(sample_dir, "satellite_T3_2022.png")
    t3.save(t3_path)

    # --- Satellite T4 Image (2024) ---
    t4 = Image.new("RGB", (w, h), color=(40, 110, 60))
    draw4 = ImageDraw.Draw(t4)
    draw4.polygon([(300, 0), (600, 0), (600, 600), (350, 600)], fill=(30, 90, 190))
    draw4.rectangle([250, 200, 310, 240], fill=(160, 160, 160))
    draw4.line([(100, 0), (100, 600)], fill=(120, 120, 120), width=15)

    # --- NEW CHANGES IN T4 ---
    # Change 1: New Industrial Port Terminal & Dock Extension into Bay
    draw4.rectangle([320, 80, 480, 220], fill=(70, 75, 85))

    # Change 2: 3 Circular Fuel Storage Tanks (White circles)
    draw4.ellipse([50, 360, 110, 420], fill=(240, 240, 240))
    draw4.ellipse([120, 360, 180, 420], fill=(240, 240, 240))
    draw4.ellipse([85, 430, 145, 490], fill=(240, 240, 240))

    # Change 3: Coastal Erosion / Landslide Patch (Brown soil)
    draw4.polygon([(330, 420), (450, 400), (490, 520), (350, 540)], fill=(140, 75, 30))

    t4_path = os.path.join(sample_dir, "satellite_T4_2024.png")
    t4.save(t4_path)

    return t3_path, t4_path


def main():
    print("=" * 75)
    print(" 📡 SatQuery AI - P3 Specialist Model (Satellite T3 vs T4 Test)")
    print("=" * 75)

    # Step 1: Create satellite T3 and T4 images
    t3_path, t4_path = create_satellite_t3_t4_images("sample_data")
    print(f"Created Satellite T3 Image: {t3_path}")
    print(f"Created Satellite T4 Image: {t4_path}")

    img_t3 = ImageObject(
        image_id="SAT_T3_20220615",
        file_path=t3_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=5.0,
        crs="EPSG:4326",
        bbox=[72.8500, 19.0500, 72.8600, 19.0600],
        acquisition_date="2022-06-15",
        width=600,
        height=600
    )

    img_t4 = ImageObject(
        image_id="SAT_T4_20240910",
        file_path=t4_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=5.0,
        crs="EPSG:4326",
        bbox=[72.8500, 19.0500, 72.8600, 19.0600],
        acquisition_date="2024-09-10",
        width=600,
        height=600
    )

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="Detect new industrial port dock, fuel tank additions, and coastal erosion between T3 (2022) and T4 (2024)",
        images=[img_t3, img_t4]
    )

    # Step 2: Run P3 Change Detection Engine saving strictly into outputs/
    engine = P3ChangeDetectionEngine(output_dir="outputs")
    output = engine.run(tool_input)

    # Step 3: Save explicit, clean named copies strictly in outputs/
    mask_source = [ev.mask_path for ev in output.spatial_evidence if ev.type == "mask"][0]
    overlay_source = output.raw_output_path

    mask_target = os.path.join("outputs", "satellite_T3_T4_mask.png")
    overlay_target = os.path.join("outputs", "satellite_T3_T4_overlay.png")

    Image.open(mask_source).save(mask_target)
    Image.open(overlay_source).save(overlay_target)

    print("\n--- P3 Engine Execution Result ---")
    print(f"Status: {output.status}")
    print(f"Model Used: {output.model_used}")
    print(f"Confidence Score: {output.confidence}")
    print(f"Text Answer:\n  {output.text_answer}\n")

    print(f"✅ Binary Change Mask saved strictly to: {mask_target}")
    print(f"✅ Change Overlay Image saved strictly to: {overlay_target}")


    bbox_evidences = [ev for ev in output.spatial_evidence if ev.type == "bbox"]
    print(f"\nExtracted Bounding Boxes ({len(bbox_evidences)} changes detected):")
    for idx, ev in enumerate(bbox_evidences, 1):
        print(f"  [{idx}] Label: '{ev.label}' | Coords [ymin, xmin, ymax, xmax]: {ev.coords}")


if __name__ == "__main__":
    main()
