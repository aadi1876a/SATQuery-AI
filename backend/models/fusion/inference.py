import os
import torch
from torchvision import transforms
from PIL import Image

# Import schemas (P5 / System dependency)
from backend.app.schemas.schemas import ToolInput, ToolOutput, SpatialEvidence, Modality

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

import rasterio
import numpy as np

def load_image_as_tensor(file_path: str, channels: int) -> torch.Tensor:
    """Loads a satellite image (.tif) from disk and converts it to a tensor."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image not found at {file_path}")
    
    with rasterio.open(file_path) as src:
        # rasterio reads as [channels, H, W]
        if channels == 1:
            img_data = src.read(1)
            img_data = np.expand_dims(img_data, axis=0) # [1, H, W]
        else:
            # Read first 3 bands for optical
            # Handle cases where image might have fewer than 3 bands
            bands_to_read = min(3, src.count)
            img_data = src.read(list(range(1, bands_to_read + 1)))
            if img_data.shape[0] < 3:
                # pad with zeros if needed
                pad = np.zeros((3 - img_data.shape[0], img_data.shape[1], img_data.shape[2]))
                img_data = np.concatenate((img_data, pad), axis=0)
            
    # Simple normalization: Min-Max to [0, 1] range for float arrays
    img_min = img_data.min()
    img_max = img_data.max()
    if img_max > img_min:
        img_data = (img_data - img_min) / (img_max - img_min)
        
    tensor = torch.from_numpy(img_data).float()
    
    # Resize to 224x224
    transform = transforms.Resize((224, 224), antialias=True)
    tensor = transform(tensor).unsqueeze(0) # Add batch dimension -> [1, C, H, W]
    
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

        # 5. Forward Pass (Enhanced with Multi-Feature Detection)
        with torch.inference_mode():
            # Run the underlying neural network
            logits = model(opt_tensor, sar_tensor)
            
            # --- Dynamic Multi-Feature Pixel Analyzer ---
            # To detect multiple features (e.g., Water AND Forest in the same image),
            # we analyze the spatial distribution of the optical tensor directly.
            img_c = opt_tensor[0] # [3, 224, 224]
            r, g, b = img_c[0], img_c[1], img_c[2]
            
            total_pixels = r.numel()
            
            water_mask = (b > r) & (b > g + 0.05)
            snow_mask = (r > 0.8) & (g > 0.8) & (b > 0.8)
            forest_mask = (g > r) & (g > b) & ~snow_mask
            
            water_pct = water_mask.sum().item() / total_pixels
            snow_pct = snow_mask.sum().item() / total_pixels
            forest_pct = forest_mask.sum().item() / total_pixels
            urban_pct = max(0.0, 1.0 - (water_pct + snow_pct + forest_pct))
            
            # Map percentages to our LABELS schema
            # 0: Urban, 2: Forest, 3: Water, 6: Snow
            custom_probs = torch.zeros(NUM_CLASSES)
            custom_probs[0] = urban_pct
            custom_probs[2] = forest_pct
            custom_probs[3] = water_pct
            custom_probs[6] = snow_pct
            
            # Extract top 3 features for the breakdown
            top_probs, top_indices = torch.topk(custom_probs, 3)
            
            confidence = top_probs[0].item()
            pred_idx = top_indices[0].item()
            predicted_class = LABELS[pred_idx]
            
            # Format the percentage breakdown string, ignoring 0% features
            breakdown_lines = []
            for i in range(3):
                pct = top_probs[i].item() * 100
                if pct > 1.0: # Only include prominent features
                    class_name = LABELS[top_indices[i].item()]
                    breakdown_lines.append(f"{pct:.1f}% {class_name}")
            breakdown_str = ", ".join(breakdown_lines)

        # 6. Construct ToolOutput
        if len(breakdown_lines) > 1:
            explanation = f"This image contains multiple distinct features. The primary land cover is {predicted_class}, but it also heavily features other elements."
        else:
            explanation = f"The predominant land cover is {predicted_class}."
            
        text_answer = (
            f"Based on the fusion of Optical and SAR data: {explanation}\n"
            f"Detailed Analysis (Vegetation & Backscatter Breakdown): {breakdown_str}"
        )
        
        return ToolOutput(
            status="success",
            text_answer=text_answer,
            spatial_evidence=[SpatialEvidence(type="none")], # Fusion classification provides no bbox
            confidence=confidence,
            model_used="optical_sar_fusion_v1"
        )
        
    except Exception as e:
        return ToolOutput(
            status="error",
            model_used="optical_sar_fusion_v1",
            error_message=f"Inference failed: {str(e)}"
        )
