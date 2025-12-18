import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import os
import numpy as np
from PIL import Image

# --- SOTA CONFIGURATION (MRI Slice Classification) ---
# Note: Full 3D segmentation requires MONAI. This script implements
# the "2.5D" approach (classifying individual slices) which is efficient for MVP.
CONFIG = {
    'img_size': 256,
    'batch_size': 32,
    'epochs': 30,
    'lr': 1e-4,
    'model': 'resnet50', # ResNet is standard for MRI slice classification
    'classes': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Transform: MRI Specific
def get_mri_transforms(mode='train'):
    # MRI images are grayscale but models expect RGB.
    # We replicate the channel 3 times.
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomRotation(10), # MRI heads are usually upright
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Normalize using MRI-specific mean/std if available, else ImageNet
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class MRIDataset(torch.utils.data.Dataset):
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
        image = Image.open(img_path).convert('RGB')  # Convert grayscale to RGB
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_mri():
    print(f"🚀 Starting SOTA MRI Training ({CONFIG['model']})...")
    
    # 1. Load ResNet50
    model = models.resnet50(pretrained=True)
    
    # 2. Modify Head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CONFIG['classes']))
    model = model.to(DEVICE)
    
    # 3. Loss
    criterion = nn.CrossEntropyLoss()
    
    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 5. Data Check
    data_dir = '../Calibration_Dataset/mri_data'
    if not os.path.exists(data_dir):
        print(f"⚠ Data not found at {data_dir}. Saving pre-trained weights only.")
        os.makedirs('../weights/mri', exist_ok=True)
        torch.save(model.state_dict(), '../weights/mri/resnet50_mri.pth')
        print("✅ Pre-trained Model Saved!")
        return
    
    # 6. Create Dataset and DataLoader
    from torch.utils.data import DataLoader
    train_dataset = MRIDataset(data_dir, transform=get_mri_transforms('train'))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    
    if len(train_dataset) == 0:
        print("⚠ No training data found. Saving pre-trained weights only.")
        os.makedirs('../weights/mri', exist_ok=True)
        torch.save(model.state_dict(), '../weights/mri/resnet50_mri.pth')
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
            
            if batch_idx % 5 == 0:
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
            os.makedirs('../weights/mri', exist_ok=True)
            torch.save(model.state_dict(), '../weights/mri/resnet50_mri.pth')
            print(f"✅ Best model saved (Loss: {best_loss:.4f})\n")
    
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Model Saved: ../weights/mri/resnet50_mri.pth")
    print("="*60)

if __name__ == '__main__':
    train_mri()
