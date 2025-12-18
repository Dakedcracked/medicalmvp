#!/usr/bin/env python3
"""
COMPREHENSIVE TRAINING SCRIPT FOR INDIAN MEDICAL DATASETS
==========================================================
Flexible template for training deep learning models on custom datasets
Supports: X-Ray, CT Scan, MRI, Ultrasound, Skin Lesions

Author: Medical AI Team
Date: December 2025
Usage: python train_indian_dataset.py --config config.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import timm
import torchxrayvision as xrv
from PIL import Image
import os
import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== CONFIGURATION ====================

DEFAULT_CONFIG = {
    'modality': 'xray',  # xray, ct, mri, ultrasound, skin
    'dataset_path': './Indian_Medical_Dataset',
    'output_dir': './trained_models_indian',
    'model_name': 'efficientnet_b4',  # efficientnet_b4, densenet121, resnet50
    'img_size': 224,
    'batch_size': 16,
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 4,
    'use_pretrained': True,
    'use_augmentation': True,
    'early_stopping_patience': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==================== DATASET CLASS ====================

class IndianMedicalDataset(Dataset):
    """
    Generic dataset class for Indian medical images
    
    Expected folder structure:
    dataset_path/
        ├── class1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        ├── class2/
        │   ├── image3.jpg
        │   └── image4.jpg
        └── class3/
            └── image5.jpg
    """
    def __init__(self, root_dir, transform=None, modality='xray'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.modality = modality
        self.images = []
        self.labels = []
        self.classes = []
        
        # Load all images and labels
        for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            self.classes.append(class_name)
            
            # Supported image formats
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.dcm', '*.tif', '*.tiff']
            
            for ext in image_extensions:
                for img_path in class_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"✅ Loaded {len(self.images)} images from {len(self.classes)} classes")
        print(f"   Classes: {', '.join(self.classes)}")
        
        # Show distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\n📊 Dataset Distribution:")
        for cls, count in zip(self.classes, counts):
            print(f"   • {cls}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path)
            
            # Handle different modalities
            if self.modality == 'xray':
                # X-rays are typically grayscale
                if image.mode != 'L':
                    image = image.convert('L')
                # Convert to RGB for pretrained models
                image = image.convert('RGB')
            else:
                # MRI, CT, Ultrasound, Skin - convert to RGB
                image = image.convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== DATA AUGMENTATION ====================

def get_transforms(config, mode='train'):
    """
    Get appropriate transforms based on modality and mode
    """
    img_size = config['img_size']
    
    if mode == 'train' and config['use_augmentation']:
        # Training with augmentation
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test - no augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# ==================== MODEL CREATION ====================

def create_model(config, num_classes):
    """
    Create model based on configuration
    """
    model_name = config['model_name']
    pretrained = config['use_pretrained']
    
    print(f"\n🔧 Creating model: {model_name}")
    print(f"   Pretrained: {pretrained}")
    print(f"   Output classes: {num_classes}")
    
    if 'densenet' in model_name.lower() and config['modality'] == 'xray':
        # Use torchxrayvision for X-ray
        print("   Using torchxrayvision DenseNet121 (medical imaging optimized)")
        model = xrv.models.DenseNet(weights='densenet121-res224-all' if pretrained else None)
        
        # Replace classifier for custom number of classes
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        # Use timm for other models
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

# ==================== TRAINING LOOP ====================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels

# ==================== MAIN TRAINING FUNCTION ====================

def train_indian_model(config):
    """
    Main training function
    """
    print("=" * 70)
    print("🇮🇳 TRAINING MODEL ON INDIAN MEDICAL DATASET")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\n📁 Loading dataset...")
    full_dataset = IndianMedicalDataset(
        root_dir=config['dataset_path'],
        transform=get_transforms(config, 'train'),
        modality=config['modality']
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_dataset)} images")
    print(f"   Validation: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Create model
    num_classes = len(full_dataset.classes)
    model = create_model(config, num_classes)
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")
    print(f"   Device: {device}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = \
            validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"   Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            model_path = output_dir / f"best_model_{config['modality']}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"   ✅ Best model saved! (Val Acc: {best_val_acc:.4f})")
            
            # Save confusion matrix
            cm = confusion_matrix(val_labels, val_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=full_dataset.classes,
                       yticklabels=full_dataset.classes)
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
            plt.close()
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n⏹️ Early stopping triggered (no improvement for {config['early_stopping_patience']} epochs)")
            break
    
    # Save final model
    final_model_path = output_dir / f"final_model_{config['modality']}.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"   Best Validation F1-Score: {max(history['val_f1']):.4f}")
    print(f"\n💾 Saved Files:")
    print(f"   • Best Model: {model_path}")
    print(f"   • Final Model: {final_model_path}")
    print(f"   • Training History: {history_path}")
    print(f"   • Confusion Matrix: {output_dir / 'confusion_matrix.png'}")
    print(f"   • Training Curves: {output_dir / 'training_curves.png'}")
    print("=" * 70)
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return model, history, full_dataset.classes

# ==================== COMMAND LINE INTERFACE ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on Indian medical dataset')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--modality', type=str, choices=['xray', 'ct', 'mri', 'ultrasound', 'skin'],
                       help='Medical imaging modality')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        config = {**DEFAULT_CONFIG, **config}
    else:
        config = DEFAULT_CONFIG.copy()
    
    # Override with command line arguments
    if args.modality:
        config['modality'] = args.modality
    if args.dataset_path:
        config['dataset_path'] = args.dataset_path
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # Train model
    train_indian_model(config)
