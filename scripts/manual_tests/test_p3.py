"""
test_p3.py
Test suite to verify P3 Specialist Model (Change Detection & Temporal Analysis)
end-to-end against SatQuery AI schemas.
"""

import os
from PIL import Image, ImageDraw
from backend.app.schemas.schemas import ToolInput, ImageObject, TaskType, Modality
from p3_model import P3ChangeDetectionEngine


def create_sample_bitemporal_images(sample_dir: str = "sample_data"):
    """
    Creates synthetic T1 (Before) and T2 (After) sample satellite images.
    T1: Green field with road and 1 building.
    T2: Same scene, but with 2 new buildings and deforestation (area changed).
    """
    os.makedirs(sample_dir, exist_ok=True)
    w, h = 512, 512

    # --- T1 Image (Pre-event / Earlier date) ---
    t1_img = Image.new("RGB", (w, h), color=(34, 139, 34))  # Forest green
    draw1 = ImageDraw.Draw(t1_img)
    # Draw gray road
    draw1.rectangle([200, 0, 240, 512], fill=(100, 100, 100))
    # Draw 1 existing building (blue roof)
    draw1.rectangle([100, 100, 160, 160], fill=(70, 130, 180))
    t1_path = os.path.join(sample_dir, "satellite_T1_2021.png")
    t1_img.save(t1_path)

    # --- T2 Image (Post-event / Later date) ---
    t2_img = Image.new("RGB", (w, h), color=(34, 139, 34))
    draw2 = ImageDraw.Draw(t2_img)
    # Draw same road
    draw2.rectangle([200, 0, 240, 512], fill=(100, 100, 100))
    # Draw original building
    draw2.rectangle([100, 100, 160, 160], fill=(70, 130, 180))

    # --- CHANGES IN T2 ---
    # Change 1: New construction (Red building)
    draw2.rectangle([300, 250, 380, 330], fill=(200, 50, 50))
    # Change 2: Another construction (Yellow warehouse)
    draw2.rectangle([50, 350, 140, 420], fill=(220, 200, 50))
    # Change 3: Clearing / Deforestation area (Brown patch)
    draw2.rectangle([320, 50, 450, 180], fill=(139, 69, 19))

    t2_path = os.path.join(sample_dir, "satellite_T2_2023.png")
    t2_img.save(t2_path)

    return t1_path, t2_path


def main():
    print("=" * 70)
    print(" SatQuery AI - P3 Specialist Model Test (Change Detection & Temporal Analysis)")
    print("=" * 70)

    # Step 1: Create test bi-temporal images
    t1_path, t2_path = create_sample_bitemporal_images()
    print(f"Created sample T1 image: {t1_path}")
    print(f"Created sample T2 image: {t2_path}")

    # Step 2: Build ImageObjects (as provided by P1 in team pipeline)
    image_t1 = ImageObject(
        image_id="IMG_T1_20210510",
        file_path=t1_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.1000, 28.6000, 77.1050, 28.6050],
        acquisition_date="2021-05-10",
        width=512,
        height=512,
        thumbnail_path=None
    )

    image_t2 = ImageObject(
        image_id="IMG_T2_20231120",
        file_path=t2_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.1000, 28.6000, 77.1050, 28.6050],
        acquisition_date="2023-11-20",
        width=512,
        height=512,
        thumbnail_path=None
    )

    # Step 3: Create ToolInput (as sent by P5 controller)
    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query="Identify new buildings and land clearing between 2021 and 2023",
        images=[image_t1, image_t2],
        params={"threshold": 0.2, "save_mask": True}
    )

    print("\n--- Input Payload (P5 -> P3) ---")
    print(f"Task: {tool_input.task}")
    print(f"Query: '{tool_input.query}'")
    print(f"Images count: {len(tool_input.images)}")

    # Step 4: Run P3 Engine (saving outputs strictly into outputs/)
    engine = P3ChangeDetectionEngine(output_dir="outputs")
    output = engine.run(tool_input)

    print("\n--- Output Payload (P3 -> P5 Controller) ---")
    print(f"Status: {output.status}")
    print(f"Model Used: {output.model_used}")
    print(f"Confidence: {output.confidence}")
    print(f"Text Answer:\n  {output.text_answer}")
    print(f"Raw Output Overlay Path: {output.raw_output_path}")
    print(f"Spatial Evidence Count: {len(output.spatial_evidence)}")

    for idx, ev in enumerate(output.spatial_evidence, 1):
        print(f"  [{idx}] Type: {ev.type} | Label: '{ev.label}' | Coords: {ev.coords} | Mask: {ev.mask_path}")

    if output.status == "success":
        print("\n✅ P3 Model Test PASSED! Visual outputs saved strictly in 'outputs/'.")
    else:
        print(f"\n❌ P3 Model Test FAILED: {output.error_message}")




if __name__ == "__main__":
    main()
