import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import time
import copy
from PIL import Image
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score

# --- SOTA CONFIGURATION (CheXpert Style) ---
# Based on: "CheXpert: A Large Chest Radiograph Dataset..." (Irvin et al., 2019)
CONFIG = {
    'img_size': 224,
    'batch_size': 16, # Small batch size for better generalization
    'epochs': 20,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'model_name': 'densenet121-res224-all', # Starting point
    'target_classes': ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Normal']
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        # Recursive search for images
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(root, file))
        
        # In a real scenario, you would load a CSV with multi-labels here.
        # For this script, we assume folder names = class names (Single Label)
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # XRV expects 1 channel (Grayscale)
        image = Image.open(img_path).convert('L')
        
        # Get label from folder name
        folder_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx.get(folder_name, 0) # Default to 0 if issue
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_sota_transforms(mode='train'):
    # XRV specific normalization is critical
    # We use simple 0.5 mean/std for grayscale as a robust baseline
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), # 50% flip
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Robustness
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

def train_xray():
    print(f"🚀 Starting SOTA X-Ray Training (DenseNet121)...")
    
    # 1. Load SOTA Model
    # We use the library to fetch the architecture but we might need to adjust the head
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # Replace Classifier for our specific classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(CONFIG['target_classes']))
    model = model.to(DEVICE)
    
    # 2. Loss Function (BCE is standard for Multi-label, but CE for Single-label folders)
    # We use CrossEntropy with Label Smoothing (0.1) to prevent overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 3. Optimizer (AdamW is SOTA)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # 4. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # 5. Data Loaders
    data_dir = '../Calibration_Dataset'
    if not os.path.exists(data_dir):
        print(f"⚠ Data not found at {data_dir}. Saving pre-trained weights only.")
        os.makedirs('../weights/xray', exist_ok=True)
        torch.save(model.state_dict(), '../weights/xray/densenet_sota.pth')
        print("✅ Pre-trained Model Saved!")
        return

    dataset = XRayDataset(data_dir, transform=get_sota_transforms('train'))
    
    if len(dataset) == 0:
        print("⚠ No training data found. Saving pre-trained weights only.")
        os.makedirs('../weights/xray', exist_ok=True)
        torch.save(model.state_dict(), '../weights/xray/densenet_sota.pth')
        print("✅ Pre-trained Model Saved!")
        return
    
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    
    # Training Loop
    print(f"\\nTraining on {len(dataset)} images for {CONFIG['epochs']} epochs...")
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Batch [{batch_idx}/{len(loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {100*correct/total:.2f}%")
        
        # Epoch statistics
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total
        print(f"\\n{'='*60}")
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Summary:")
        print(f"Average Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        print(f"{'='*60}\\n")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs('../weights/xray', exist_ok=True)
            torch.save(model.state_dict(), '../weights/xray/densenet_sota.pth')
            print(f"✅ Best model saved (Loss: {best_loss:.4f})\\n")
        
        scheduler.step()
        
    print("\\n" + "="*60)
    print("🎉 Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Model Saved: ../weights/xray/densenet_sota.pth")
    print("="*60)

if __name__ == '__main__':
    train_xray()
