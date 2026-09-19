import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from backend.models.fusion.fusion_model import OpticalSARFusionModel

# ---------------------------------------------------------
# Simulated Structured Dataset
# ---------------------------------------------------------
class SimulatedSatelliteDataset(Dataset):
    """Simulates a proper satellite dataset structure for P4."""
    def __init__(self, num_samples=100, num_classes=10, split="train"):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.split = split
        # We will use random data for now, but this represents the structure 
        # where we'd use rasterio to load .tif files.

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a deterministic pattern so the model can actually learn it
        # Real data will replace this in the future
        label = idx % self.num_classes
        opt_img = torch.ones(3, 224, 224) * (label * 0.1)
        sar_img = torch.ones(1, 224, 224) * (label * 0.1)
        # Data Augmentation: Flips prevent overfitting on satellite imagery
        if self.split == 'train':
            # Concatenate for identical transform
            combined = torch.cat([opt_img, sar_img], dim=0)
            if torch.rand(1) > 0.5:
                combined = transforms.functional.hflip(combined)
            if torch.rand(1) > 0.5:
                combined = transforms.functional.vflip(combined)
            opt_img, sar_img = combined[:3], combined[3:]
            
        return opt_img, sar_img, label

class TrainConfig:
    NUM_EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 10
    SAR_CHANNELS = 1
    CHECKPOINT_DIR = "backend/models/fusion/checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_fusion_model.pth")

def train():
    config = TrainConfig()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Proper train/val/test splits
    train_dataset = SimulatedSatelliteDataset(num_samples=200, split="train")
    val_dataset = SimulatedSatelliteDataset(num_samples=50, split="val")
    test_dataset = SimulatedSatelliteDataset(num_samples=50, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = OpticalSARFusionModel(num_classes=config.NUM_CLASSES, sar_in_channels=config.SAR_CHANNELS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Cosine Annealing Learning Rate Scheduler for higher final accuracy
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (opt_imgs, sar_imgs, labels) in enumerate(train_loader):
            opt_imgs = opt_imgs.to(device)
            sar_imgs = sar_imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(opt_imgs, sar_imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for opt_imgs, sar_imgs, labels in val_loader:
                opt_imgs, sar_imgs, labels = opt_imgs.to(device), sar_imgs.to(device), labels.to(device)
                logits = model(opt_imgs, sar_imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Step the scheduler
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, config.BEST_MODEL_PATH)
            
    print("Training Complete. Run evaluate.py for full test metrics.")

if __name__ == "__main__":
    train()
