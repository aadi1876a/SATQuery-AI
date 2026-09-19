import glob
import os
import json
from backend.models.fusion.inference import call_fusion_model
from backend.schemas import ToolInput, ImageObject, Modality, TaskType

opt_files = sorted(glob.glob("input_optical*.png"))
sar_files = sorted(glob.glob("input_sar*.png"))

print("Found the following pairs:")
for opt, sar in zip(opt_files, sar_files):
    print(f" - {opt} & {sar}")
    
print("\nRunning Inference on all pairs...")

for idx, (opt_path, sar_path) in enumerate(zip(opt_files, sar_files)):
    print(f"\n=========================================")
    print(f"TESTING PAIR {idx + 1}: {opt_path} + {sar_path}")
    print(f"=========================================")
    
    opt_image = ImageObject(
        image_id=f"opt_{idx}", file_path=opt_path, modality=Modality.optical,
        format="png", bands=3, resolution_m=10.0, crs="EPSG:4326", bbox=[0,0,0,0], width=224, height=224
    )
    sar_image = ImageObject(
        image_id=f"sar_{idx}", file_path=sar_path, modality=Modality.sar,
        format="png", bands=1, resolution_m=10.0, crs="EPSG:4326", bbox=[0,0,0,0], width=224, height=224
    )

    tool_input = ToolInput(
        task=TaskType.fusion_analysis,
        query="Analyze this",
        images=[opt_image, sar_image]
    )

    result = call_fusion_model(tool_input)
    print(result.model_dump_json(indent=2))
