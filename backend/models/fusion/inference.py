import os
import torch
from torchvision import transforms
from PIL import Image

# Import schemas (P5 / System dependency)
from backend.schemas import ToolInput, ToolOutput, SpatialEvidence, Modality

# Import our P4 model
from backend.models.fusion.fusion_model import OpticalSARFusionModel

# ---------------------------------------------------------
# Inference Configuration
# ---------------------------------------------------------
CHECKPOINT_PATH = "backend/models/fusion/checkpoints/best_fusion_model.pth"
NUM_CLASSES = 10
SAR_CHANNELS = 1

# Dummy land cover labels for the hackathon classification task
LABELS = [
    "Urban", "Agriculture", "Forest", "Water", "Barren", 
    "Wetland", "Snow", "Cloud", "Shadow", "Unknown"
]

def load_image_as_tensor(file_path: str, channels: int) -> torch.Tensor:
    """Loads an image from disk and converts it to a tensor."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image not found at {file_path}")
    
    # Simple transform matching ResNet expectations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # In a real scenario, SAR might be read via rasterio/GDAL (P1 dependency).
    # For now, assuming PIL can open standard formats provided by P1.
    img = Image.open(file_path)
    
    # Handle channel matching
    if channels == 1:
        img = img.convert("L")
    elif channels == 3:
        img = img.convert("RGB")
        
    tensor = transform(img).unsqueeze(0) # Add batch dimension -> [1, C, H, W]
    return tensor

def call_fusion_model(tool_input: ToolInput) -> ToolOutput:
    """
    Main inference interface for the P5 Agent.
    Takes a ToolInput, runs Optical+SAR fusion, and returns ToolOutput.
    """
    try:
        # 1. Validate Input and Extract Modalities
        opt_path = None
        sar_path = None
        
        for img_obj in tool_input.images:
            if img_obj.modality == Modality.optical:
                opt_path = img_obj.file_path
            elif img_obj.modality == Modality.sar:
                sar_path = img_obj.file_path

        if not opt_path or not sar_path:
            return ToolOutput(
                status="error",
                model_used="optical_sar_fusion_v1",
                error_message="Fusion requires exactly one optical and one SAR image."
            )

        # 2. Setup Device and Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OpticalSARFusionModel(num_classes=NUM_CLASSES, sar_in_channels=SAR_CHANNELS)
        
        # 3. Load Checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"WARNING: No checkpoint found at {CHECKPOINT_PATH}. Using untrained weights.")

        model.to(device)
        model.eval()

        # 4. Prepare Tensors
        opt_tensor = load_image_as_tensor(opt_path, channels=3).to(device)
        sar_tensor = load_image_as_tensor(sar_path, channels=SAR_CHANNELS).to(device)

        # 5. Forward Pass
        with torch.inference_mode():
            logits = model(opt_tensor, sar_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
            predicted_class = LABELS[pred_idx.item()]
            conf_score = confidence.item()

        # 6. Construct ToolOutput
        text_answer = f"Based on the fusion of Optical and SAR data, the predominant land cover is {predicted_class}."
        
        return ToolOutput(
            status="success",
            text_answer=text_answer,
            spatial_evidence=[SpatialEvidence(type="none")], # Fusion classification provides no bbox
            confidence=conf_score,
            model_used="optical_sar_fusion_v1"
        )
        
    except Exception as e:
        return ToolOutput(
            status="error",
            model_used="optical_sar_fusion_v1",
            error_message=f"Inference failed: {str(e)}"
        )
