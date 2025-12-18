# 🎓 Training Guide - Neuron AI Medical Imaging Models

**Complete guide for training custom DenseNet121 models on medical imaging datasets**

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Custom Models](#training-custom-models)
4. [Model Architecture](#model-architecture)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Evaluation & Testing](#evaluation--testing)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## 📦 Prerequisites

### System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 10GB free disk space

**Recommended:**
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM
- 16GB+ RAM
- 50GB+ free disk space

### Install Dependencies

```bash
# Install PyTorch with CUDA support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.0.0+cu118
CUDA Available: True
```

---

## 📊 Dataset Preparation

### Folder Structure

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── Pneumonia/
│   │   ├── patient001.jpg
│   │   ├── patient002.jpg
│   │   └── ...
│   ├── Normal/
│   │   ├── normal001.jpg
│   │   └── ...
│   └── [Other_Classes]/
├── val/
│   ├── Pneumonia/
│   ├── Normal/
│   └── ...
└── test/
    ├── Pneumonia/
    ├── Normal/
    └── ...
```

### Dataset Split Recommendations

| Split | Percentage | Purpose |
|-------|-----------|---------|
| **Train** | 70-80% | Model learning |
| **Validation** | 10-15% | Hyperparameter tuning |
| **Test** | 10-15% | Final evaluation |

### Image Requirements

- **Format**: JPG, PNG, or DICOM
- **Resolution**: Minimum 224×224 (will be resized)
- **Color**: Grayscale (1 channel) for X-rays, RGB for CT/MRI
- **Quality**: High resolution medical grade images

### Automated Dataset Preparation

```python
# Create dataset splits automatically
from sklearn.model_selection import train_test_split
import shutil
import os

def create_dataset_splits(source_dir, output_dir, test_size=0.2, val_size=0.1):
    """
    Automatically split dataset into train/val/test
    
    Args:
        source_dir: Directory with class folders
        output_dir: Output directory for splits
        test_size: Percentage for test set (0.2 = 20%)
        val_size: Percentage for validation set (0.1 = 10%)
    """
    classes = os.listdir(source_dir)
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        images = os.listdir(class_path)
        
        # Split train+val and test
        train_val, test = train_test_split(images, test_size=test_size, random_state=42)
        
        # Split train and val
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
        
        # Create directories and copy files
        for split_name, split_images in [('train', train), ('val', val), ('test', test)]:
            split_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)
            
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)
        
        print(f"✅ {class_name}: {len(train)} train, {len(val)} val, {len(test)} test")

# Usage
create_dataset_splits('raw_dataset/', 'processed_dataset/')
```

---

## 🏋️ Training Custom Models

### Method 1: Using Training Script

**Create `train.py`:**

```python
"""
Training script for custom DenseNet121 on medical imaging datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import json

# Import custom model
from models.densenet import densenet121
from models.preprocessing import DataAugmentation

class MedicalImageDataset(Dataset):
    """Custom dataset for medical images"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with class folders
            transform: Optional transform to apply
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"📊 Loaded {len(self.samples)} images from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Grayscale for X-ray
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_model(
    train_dir,
    val_dir,
    num_classes,
    epochs=30,
    batch_size=16,
    learning_rate=0.0001,
    save_dir='checkpoints'
):
    """
    Train DenseNet121 model
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        num_classes: Number of output classes
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Data transforms
    train_transform = DataAugmentation.get_train_transforms('xray', img_size=224)
    val_transform = DataAugmentation.get_val_transforms('xray', img_size=224)
    
    # Datasets
    train_dataset = MedicalImageDataset(train_dir, transform=train_transform)
    val_dataset = MedicalImageDataset(val_dir, transform=val_transform)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print(f"🤖 Creating DenseNet121 model with {num_classes} classes...")
    model = densenet121(num_classes=num_classes, in_channels=1)
    model = model.to(device)
    
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 Starting Training")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"   Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'history': history
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"   ✅ Best model saved! (Val Acc: {best_val_acc*100:.2f}%)")
        
        # Learning rate scheduling
        scheduler.step(epoch_val_acc)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"   Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"   Models saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    return model, history


if __name__ == "__main__":
    # Configuration
    TRAIN_DIR = 'dataset/train'
    VAL_DIR = 'dataset/val'
    NUM_CLASSES = 18  # Number of pathologies
    EPOCHS = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    
    # Train model
    model, history = train_model(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        num_classes=NUM_CLASSES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_dir='checkpoints'
    )
```

### Running Training

```bash
# Basic training
python train.py

# With custom parameters
python train.py --train_dir dataset/train \
                --val_dir dataset/val \
                --num_classes 18 \
                --epochs 50 \
                --batch_size 32 \
                --learning_rate 0.0001
```

---

## 🧬 Model Architecture

### DenseNet121 Structure

```
Input (224×224×1) - Grayscale X-ray
    ↓
Initial Conv (7×7, stride 2) + MaxPool
    ↓
DenseBlock1 (6 layers) → Transition1
    ↓
DenseBlock2 (12 layers) → Transition2
    ↓
DenseBlock3 (24 layers) → Transition3
    ↓
DenseBlock4 (16 layers)
    ↓
Global Average Pooling
    ↓
Fully Connected (1024 → 18)
    ↓
Output (18 pathology predictions)
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Growth Rate** | 32 | Features added per layer |
| **Block Config** | (6,12,24,16) | Layers per block |
| **Total Layers** | 121 | Deep network |
| **Parameters** | 6,966,034 | ~7M trainable params |
| **Input Size** | 224×224×1 | Grayscale images |
| **Output Size** | 18 | Multi-class prediction |

---

## ⚙️ Hyperparameter Tuning

### Recommended Hyperparameters

**Small Dataset (<1000 images):**
```python
BATCH_SIZE = 8
LEARNING_RATE = 0.00001
EPOCHS = 50
WEIGHT_DECAY = 0.0001
DROPOUT = 0.3
```

**Medium Dataset (1000-10000 images):**
```python
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 30
WEIGHT_DECAY = 0.00001
DROPOUT = 0.2
```

**Large Dataset (>10000 images):**
```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
WEIGHT_DECAY = 0.00001
DROPOUT = 0.1
```

### Learning Rate Scheduling

```python
# ReduceLROnPlateau - recommended
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
    verbose=True
)

# CosineAnnealingLR - alternative
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)
```

---

## 📈 Evaluation & Testing

### Evaluate Trained Model

```python
"""
Evaluate model on test set
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, device, class_names):
    """Comprehensive model evaluation"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    print("\n" + "="*60)
    print("📊 Classification Report")
    print("="*60 + "\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ Confusion matrix saved to confusion_matrix.png")
    
    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\n📊 Per-Class Accuracy:")
    for name, acc in zip(class_names, class_acc):
        print(f"   {name}: {acc*100:.2f}%")

# Usage
evaluate_model(model, test_loader, device, class_names)
```

---

## 🚀 Deployment

### Export Model for Production

```python
# Save model weights only
torch.save(model.state_dict(), 'weights/xray_densenet121.pth')

# Load for inference
from models.densenet import densenet121

model = densenet121(num_classes=18)
model.load_state_dict(torch.load('weights/xray_densenet121.pth'))
model.eval()
```

### Use in API Server

```python
# Update api_server_custom.py
MODEL_PATHS = {
    'xray': 'weights/xray_densenet121.pth',  # Your trained model
}
```

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size
BATCH_SIZE = 8  # Instead of 16 or 32

# Or use gradient accumulation
accumulation_steps = 4
```

**2. Training Loss Not Decreasing**
```python
# Solution: Lower learning rate
LEARNING_RATE = 0.00001  # Instead of 0.0001

# Or use warmup
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(epoch):
    return (epoch + 1) / 5  # 5 epoch warmup

scheduler = LambdaLR(optimizer, warmup_lambda)
```

**3. Overfitting (Train >> Val Accuracy)**
```python
# Solution: Add regularization
- Increase dropout
- Increase data augmentation
- Add weight decay
- Early stopping
```

**4. Model Not Learning (Accuracy ~Random)**
```python
# Check:
- Label distribution (balanced?)
- Learning rate (too high/low?)
- Data preprocessing (correct normalization?)
- Model architecture (correct num_classes?)
```

---

## 📞 Support

For issues or questions:
- Check existing documentation
- Review training logs
- Test on smaller dataset first
- Verify data preprocessing

**✅ Happy Training! Your medical AI model is ready to save lives! 🏥**
