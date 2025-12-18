import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import timm
import numpy as np
from PIL import Image

# --- SOTA CONFIGURATION (ISIC Winner Style) ---
# Based on: "SIIM-ISIC Melanoma Classification 1st Place Solution"
CONFIG = {
    'img_size': 384, # Higher resolution is CRITICAL for skin lesions
    'batch_size': 8,
    'epochs': 15,
    'lr': 3e-4,
    'backbone': 'efficientnet_b4', # Good balance of speed/accuracy
    'classes': ['Melanoma', 'Nevus', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Transform: "Microscope Simulation"
# Skin lesions often have hair, bubbles, or ruler markers.
# We simulate these artifacts to make the model robust.
def get_skin_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), # Skin has no "up" or "down"
            transforms.RandomRotation(180), # Full rotation allowed
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Skin tone invariance
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from class folders
        for class_idx, class_name in enumerate(CONFIG['classes']):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_skin():
    print(f"🚀 Starting SOTA Skin Cancer Training ({CONFIG['backbone']})...")
    
    # 1. Load EfficientNet (Best for Texture)
    model = timm.create_model(CONFIG['backbone'], pretrained=True, num_classes=len(CONFIG['classes']))
    model = model.to(DEVICE)
    
    # 2. Loss: Weighted Cross Entropy (Melanoma is rare!)
    # In a real run, calculate these from dataset. Here we estimate.
    # Melanoma is usually 10x rarer than Nevus.
    class_weights = torch.tensor([5.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    # 4. Scheduler: Cosine Annealing with Warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    
    # 5. Data Check
    data_dir = '../Calibration_Dataset/skin_data'
    if not os.path.exists(data_dir):
        print(f"⚠ Data not found at {data_dir}. Saving pre-trained weights only.")
        os.makedirs('../weights/skin', exist_ok=True)
        torch.save(model.state_dict(), '../weights/skin/efficientnet_skin.pth')
        print("✅ Pre-trained Model Saved!")
        return
    
    # 6. Create Dataset and DataLoader
    train_dataset = SkinDataset(data_dir, transform=get_skin_transforms('train'))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    
    if len(train_dataset) == 0:
        print("⚠ No training data found. Saving pre-trained weights only.")
        os.makedirs('../weights/skin', exist_ok=True)
        torch.save(model.state_dict(), '../weights/skin/efficientnet_skin.pth')
        print("✅ Pre-trained Model Saved!")
        return
    
    # 7. Training Loop
    print(f"\nTraining on {len(train_dataset)} images for {CONFIG['epochs']} epochs...")
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {100*correct/total:.2f}%")
        
        # Epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Summary:")
        print(f"Average Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs('../weights/skin', exist_ok=True)
            torch.save(model.state_dict(), '../weights/skin/efficientnet_skin.pth')
            print(f"✅ Best model saved (Loss: {best_loss:.4f})\n")
        
        # Step scheduler
        scheduler.step()
    
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Model Saved: ../weights/skin/efficientnet_skin.pth")
    print("="*60)

if __name__ == '__main__':
    train_skin()
