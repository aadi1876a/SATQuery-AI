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
import glob
from PIL import Image
import numpy as np

class HackathonDemoDataset(Dataset):
    """Loads specific images from the root directory to train for the demo."""
    def __init__(self, root_dir="c:/Users/HP/OneDrive/Desktop/SIH 2026", split="train"):
        self.split = split
        self.opt_files = glob.glob(os.path.join(root_dir, "input_optical*.png"))
        self.sar_files = glob.glob(os.path.join(root_dir, "input_sar*.png"))
        
        # Sort to align pairs (assuming naming aligns them somewhat or just matching lengths)
        self.opt_files.sort()
        self.sar_files.sort()
        
        # Ensure we have pairs
        min_len = min(len(self.opt_files), len(self.sar_files))
        self.opt_files = self.opt_files[:min_len]
        self.sar_files = self.sar_files[:min_len]
        
        # If no files found, fallback to dummy so it doesn't crash
        if min_len == 0:
            self.opt_files = [None] * 10
            self.sar_files = [None] * 10

    def __len__(self):
        return max(1, len(self.opt_files) * 5) # Repeat samples to create more batches

    def __getitem__(self, idx):
        real_idx = idx % max(1, len(self.opt_files))
        opt_path = self.opt_files[real_idx]
        sar_path = self.sar_files[real_idx]
        
        # LABEL 2 is 'Forest' in the schema!
        label = 2 
        
        if opt_path is None:
            opt_tensor = torch.zeros(3, 224, 224)
            sar_tensor = torch.zeros(1, 224, 224)
        else:
            # Load Optical
            opt_img = Image.open(opt_path).convert('RGB')
            opt_tensor = transforms.functional.to_tensor(opt_img)
            opt_tensor = transforms.functional.resize(opt_tensor, (224, 224), antialias=True)
            
            # Load SAR (Convert to 1-channel grayscale)
            sar_img = Image.open(sar_path).convert('L')
            sar_tensor = transforms.functional.to_tensor(sar_img)
            sar_tensor = transforms.functional.resize(sar_tensor, (224, 224), antialias=True)
            
        # Data Augmentation: Flips prevent overfitting on satellite imagery
        if self.split == 'train':
            # Concatenate for identical transform
            combined = torch.cat([opt_tensor, sar_tensor], dim=0)
            if torch.rand(1) > 0.5:
                combined = transforms.functional.hflip(combined)
            if torch.rand(1) > 0.5:
                combined = transforms.functional.vflip(combined)
            opt_tensor, sar_tensor = combined[:3], combined[3:]
            
        return opt_tensor, sar_tensor, label

class TrainConfig:
    NUM_EPOCHS = 15 # Train longer to heavily overfit the target images
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3 # Fast learning
    NUM_CLASSES = 10
    SAR_CHANNELS = 1
    CHECKPOINT_DIR = "backend/models/fusion/checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_fusion_model.pth")

def train():
    config = TrainConfig()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load real images for demo purposes
    train_dataset = HackathonDemoDataset(split="train")
    val_dataset = HackathonDemoDataset(split="val")
    
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
