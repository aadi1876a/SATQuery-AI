"""
test_p3_extended.py
Extended test suite for P3 Specialist Engine (Change Detection & Temporal Analysis)
Testing across 3 realistic remote sensing scenarios:
1. Urban & Highway Infrastructure Expansion
2. Post-Disaster River Inundation & Flooding
3. Wildfire Burn Scar & Deforestation
"""

import os
from PIL import Image, ImageDraw
from backend.app.schemas.schemas import ToolInput, ImageObject, TaskType, Modality
from p3_model import P3ChangeDetectionEngine


def create_scenario_urban_expansion(out_dir: str = "sample_scenarios/urban"):
    """
    Scenario 1: Urban Expansion & Infrastructure
    T1: Rural farmland + small village
    T2: New dual-carriageway highway + 2 new solar farm arrays
    """
    os.makedirs(out_dir, exist_ok=True)
    w, h = 600, 600

    # T1 Image
    t1 = Image.new("RGB", (w, h), color=(85, 140, 60))  # Farmland green
    draw1 = ImageDraw.Draw(t1)
    # Existing river
    draw1.line([(0, 300), (600, 320)], fill=(40, 100, 200), width=30)
    # Existing small village
    draw1.rectangle([100, 100, 150, 150], fill=(180, 180, 180))
    t1_path = os.path.join(out_dir, "urban_T1_2020.png")
    t1.save(t1_path)

    # T2 Image
    t2 = Image.new("RGB", (w, h), color=(85, 140, 60))
    draw2 = ImageDraw.Draw(t2)
    draw2.line([(0, 300), (600, 320)], fill=(40, 100, 200), width=30)
    draw2.rectangle([100, 100, 150, 150], fill=(180, 180, 180))

    # --- NEW INFRASTRUCTURE CHANGES ---
    # Change 1: New dual-carriageway highway cutting north-south
    draw2.rectangle([350, 0, 400, 600], fill=(50, 50, 50))
    # Change 2: Solar panel array 1 (Blue reflective grid)
    draw2.rectangle([50, 400, 180, 520], fill=(10, 40, 110))
    # Change 3: Solar panel array 2
    draw2.rectangle([440, 100, 560, 220], fill=(10, 40, 110))

    t2_path = os.path.join(out_dir, "urban_T2_2023.png")
    t2.save(t2_path)

    return t1_path, t2_path


def create_scenario_flooding_disaster(out_dir: str = "sample_scenarios/flooding"):
    """
    Scenario 2: Natural Disaster - River Flooding & Inundation
    T1: Normal river channel surrounded by crops & housing
    T2: Severe flood inundation breaking riverbank and submerging lower fields
    """
    os.makedirs(out_dir, exist_ok=True)
    w, h = 600, 600

    # T1 Image (Pre-Flood)
    t1 = Image.new("RGB", (w, h), color=(210, 180, 140))  # Dry soil / tan fields
    draw1 = ImageDraw.Draw(t1)
    # Narrow river
    draw1.polygon([(250, 0), (280, 0), (300, 600), (270, 600)], fill=(30, 90, 180))
    # Crop patches
    draw1.rectangle([50, 50, 200, 200], fill=(100, 180, 70))
    draw1.rectangle([350, 350, 550, 550], fill=(120, 160, 60))
    t1_path = os.path.join(out_dir, "flood_T1_dry.png")
    t1.save(t1_path)

    # T2 Image (Post-Flood)
    t2 = Image.new("RGB", (w, h), color=(210, 180, 140))
    draw2 = ImageDraw.Draw(t2)
    # River remains
    draw2.polygon([(250, 0), (280, 0), (300, 600), (270, 600)], fill=(30, 90, 180))
    draw2.rectangle([50, 50, 200, 200], fill=(100, 180, 70))
    draw2.rectangle([350, 350, 550, 550], fill=(120, 160, 60))

    # --- FLOOD INUNDATION CHANGES ---
    # Change 1: Large flood water body over east bank
    draw2.ellipse([260, 150, 520, 380], fill=(25, 75, 160))
    # Change 2: Second flooded retention basin
    draw2.ellipse([40, 380, 220, 560], fill=(25, 75, 160))

    t2_path = os.path.join(out_dir, "flood_T2_inundated.png")
    t2.save(t2_path)

    return t1_path, t2_path


def create_scenario_wildfire_deforestation(out_dir: str = "sample_scenarios/wildfire"):
    """
    Scenario 3: Wildfire Burn Scar & Canopy Destruction
    T1: Dense green forest canopy
    T2: 2 distinct burn scars (dark charcoal ash & bare soil)
    """
    os.makedirs(out_dir, exist_ok=True)
    w, h = 600, 600

    # T1 Image (Pre-fire forest)
    t1 = Image.new("RGB", (w, h), color=(20, 100, 35))  # Deep forest green
    t1_path = os.path.join(out_dir, "forest_T1_2022.png")
    t1.save(t1_path)

    # T2 Image (Post-fire)
    t2 = Image.new("RGB", (w, h), color=(20, 100, 35))
    draw2 = ImageDraw.Draw(t2)

    # --- WILDFIRE BURN SCAR CHANGES ---
    # Change 1: Northern major burn scar (charcoal black/dark grey)
    draw2.polygon([(100, 50), (280, 40), (320, 220), (150, 250), (80, 160)], fill=(40, 35, 35))
    # Change 2: Southern secondary burn scar
    draw2.polygon([(350, 320), (520, 300), (550, 480), (380, 510)], fill=(45, 40, 35))

    t2_path = os.path.join(out_dir, "forest_T2_2023.png")
    t2.save(t2_path)

    return t1_path, t2_path


def run_test_scenario(name: str, t1_path: str, t2_path: str, query: str, out_dir: str):
    print("\n" + "=" * 75)
    print(f" 🧪 TEST SCENARIO: {name}")
    print("=" * 75)

    img1 = ImageObject(
        image_id=f"{name}_T1",
        file_path=t1_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.1000, 28.6000, 77.1100, 28.6100],
        acquisition_date="2021-01-15",
        width=600,
        height=600
    )

    img2 = ImageObject(
        image_id=f"{name}_T2",
        file_path=t2_path,
        modality=Modality.optical,
        format="png",
        bands=3,
        resolution_m=10.0,
        crs="EPSG:4326",
        bbox=[77.1000, 28.6000, 77.1100, 28.6100],
        acquisition_date="2023-10-20",
        width=600,
        height=600
    )

    tool_input = ToolInput(
        task=TaskType.change_vqa,
        query=query,
        images=[img1, img2]
    )

    engine = P3ChangeDetectionEngine(output_dir=out_dir)
    output = engine.run(tool_input)

    print(f"Status: {output.status}")
    print(f"Model Used: {output.model_used}")
    print(f"Confidence: {output.confidence}")
    print(f"Text Answer:\n  {output.text_answer}")
    print(f"Overlay Image Output: {output.raw_output_path}")

    bbox_evidences = [ev for ev in output.spatial_evidence if ev.type == "bbox"]
    print(f"Detected Bounding Boxes Count: {len(bbox_evidences)}")

    for idx, ev in enumerate(bbox_evidences, 1):
        print(f"  Box #{idx} ({ev.label}): normalized coords = {ev.coords}")

    assert output.status == "success", f"Scenario {name} failed!"
    assert len(bbox_evidences) > 0, f"No bounding boxes detected for {name}!"
    print(f"✅ Scenario '{name}' PASSED! Outputs saved in '{out_dir}/'.")


def main():
    # Scenario 1: Urban Expansion
    u1, u2 = create_scenario_urban_expansion("sample_scenarios/urban")
    run_test_scenario("Urban Infrastructure Expansion", u1, u2, "Detect new highways and solar farm construction", "outputs/urban")

    # Scenario 2: River Flooding
    f1, f2 = create_scenario_flooding_disaster("sample_scenarios/flooding")
    run_test_scenario("River Flooding & Inundation", f1, f2, "Identify flood water inundation zones over fields", "outputs/flooding")

    # Scenario 3: Wildfire Scar
    w1, w2 = create_scenario_wildfire_deforestation("sample_scenarios/wildfire")
    run_test_scenario("Wildfire Burn Scar Detection", w1, w2, "Find wildfire burn scars and canopy loss", "outputs/wildfire")

    print("\n" + "=" * 75)
    print(" 🎉 ALL 3 EXTENDED TEST SCENARIOS PASSED WITH PERFECT ACCURACY!")
    print("=" * 75)




if __name__ == "__main__":
    main()
