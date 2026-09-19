import json
from backend.models.fusion.inference import call_fusion_model
from backend.schemas import ToolInput, ImageObject, Modality, TaskType

opt_image = ImageObject(
    image_id="1", file_path="input_optical.png", modality=Modality.optical,
    format="png", bands=3, resolution_m=10.0, crs="EPSG:4326", bbox=[0,0,0,0], width=224, height=224
)
sar_image = ImageObject(
    image_id="2", file_path="input_sar.png", modality=Modality.sar,
    format="png", bands=1, resolution_m=10.0, crs="EPSG:4326", bbox=[0,0,0,0], width=224, height=224
)

tool_input = ToolInput(
    task=TaskType.fusion_analysis,
    query="Analyze this",
    images=[opt_image, sar_image]
)

result = call_fusion_model(tool_input)
print("\n--- INFERENCE RESULT ---")
print(result.model_dump_json(indent=2))
