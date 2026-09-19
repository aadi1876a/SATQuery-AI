import torch
import torch.nn as nn
import torchvision.models as models

class OpticalEncoder(nn.Module):
    def __init__(self, pretrained=True, feature_dim=256):
        super().__init__()
        # Load a pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Remove the final classification layer (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection layer: 512 (ResNet-18 output) -> feature_dim
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )

    def forward(self, x):
        # x shape: [B, 3, H, W]
        features = self.backbone(x) # shape: [B, 512, 1, 1]
        projected = self.projection(features) # shape: [B, feature_dim]
        return projected


class SAREncoder(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        # Use a non-pretrained ResNet-18 for SAR, as SAR statistics differ wildly from ImageNet
        resnet = models.resnet18(weights=None)
        
        # Modify the first convolutional layer to accept `in_channels` instead of 3
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection layer: 512 -> feature_dim
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )

    def forward(self, x):
        # x shape: [B, in_channels, H, W]
        features = self.backbone(x) # shape: [B, 512, 1, 1]
        projected = self.projection(features) # shape: [B, feature_dim]
        return projected


class OpticalSARFusionModel(nn.Module):
    def __init__(self, num_classes=10, sar_in_channels=1, feature_dim=256):
        super().__init__()
        self.optical_encoder = OpticalEncoder(pretrained=True, feature_dim=feature_dim)
        self.sar_encoder = SAREncoder(in_channels=sar_in_channels, feature_dim=feature_dim)
        
        # Fused dimension is 2 * feature_dim because of concatenation
        fused_dim = feature_dim * 2
        
        # Analysis head (Task: Classification)
        self.analysis_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, optical_img=None, sar_img=None):
        """
        Forward pass for the fusion model. Supports missing modalities for ablation.
        
        Args:
            optical_img: Tensor of shape [B, 3, H, W] or None
            sar_img: Tensor of shape [B, sar_in_channels, H, W] or None
            
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        if optical_img is None and sar_img is None:
            raise ValueError("At least one modality must be provided.")

        B = optical_img.size(0) if optical_img is not None else sar_img.size(0)
        device = optical_img.device if optical_img is not None else sar_img.device

        # 1. Encode modalities
        if optical_img is not None:
            opt_features = self.optical_encoder(optical_img) # [B, feature_dim]
        else:
            opt_features = torch.zeros(B, self.optical_encoder.projection[-1].num_features, device=device)
            
        if sar_img is not None:
            sar_features = self.sar_encoder(sar_img)         # [B, feature_dim]
        else:
            sar_features = torch.zeros(B, self.sar_encoder.projection[-1].num_features, device=device)
        
        # 2. Feature Fusion (Concatenation)
        fused_features = torch.cat([opt_features, sar_features], dim=1) # [B, 2 * feature_dim]
        
        # 3. Task Prediction
        logits = self.analysis_head(fused_features) # [B, num_classes]
        
        return logits

def test_model_forward():
    """Simple test to verify tensor shapes and forward pass."""
    batch_size = 2
    num_classes = 10
    
    # Create dummy tensors
    dummy_optical = torch.randn(batch_size, 3, 224, 224)
    dummy_sar = torch.randn(batch_size, 1, 224, 224)
    
    model = OpticalSARFusionModel(num_classes=num_classes, sar_in_channels=1)
    
    # Set to eval to avoid batchnorm issues with small batch
    model.eval()
    
    with torch.no_grad():
        out = model(dummy_optical, dummy_sar)
        
    assert out.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {out.shape}"
    print("Forward pass successful! Tensor shapes are correct.")

if __name__ == "__main__":
    test_model_forward()
