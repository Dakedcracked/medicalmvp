"""
Complete Training Script for Custom DenseNet121 Medical Imaging Model
Usage: python train.py --train_dir dataset/train --val_dir dataset/val --num_classes 18
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import argparse
from tqdm import tqdm
import json
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"📊 Loaded {len(self.samples)} images from {len(self.classes)} classes")
        for cls in self.classes:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[cls])
            print(f"   {cls}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('L')  # Grayscale for X-ray
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            # Return black image as fallback
            image = Image.new('L', (224, 224), 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_model(args):
    """
    Train DenseNet121 model
    
    Args:
        args: Command line arguments
    """
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*60}\n")
    
    # Data transforms
    print("🔄 Setting up data transforms...")
    train_transform = DataAugmentation.get_train_transforms('xray', img_size=224)
    val_transform = DataAugmentation.get_val_transforms('xray', img_size=224)
    
    # Datasets
    print(f"\n📁 Loading datasets from:")
    print(f"   Train: {args.train_dir}")
    train_dataset = MedicalImageDataset(args.train_dir, transform=train_transform)
    
    print(f"   Val: {args.val_dir}")
    val_dataset = MedicalImageDataset(args.val_dir, transform=val_transform)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    print(f"\n🤖 Creating DenseNet121 model...")
    print(f"   Classes: {args.num_classes}")
    print(f"   Input channels: 1 (grayscale)")
    
    model = densenet121(num_classes=args.num_classes, in_channels=1)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=args.patience, 
        verbose=True,
        min_lr=1e-7
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 Starting Training")
    print(f"{'='*60}")
    print(f"📊 Training for {args.epochs} epochs")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Early stopping patience: {args.early_stopping}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
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
            pbar = tqdm(val_loader, desc='Validation', leave=False)
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
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train - Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc*100:.2f}%")
        print(f"   Val   - Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc*100:.2f}%")
        print(f"   Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'history': history,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            
            # Save model weights only (for deployment)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_weights.pth'))
            
            print(f"   ✅ Best model saved! (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{args.early_stopping})")
        
        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
        
        # Learning rate scheduling
        scheduler.step(epoch_val_acc)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_weights.pth'))
    
    # Save training history
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save model info
    model_info = {
        'num_classes': args.num_classes,
        'architecture': 'DenseNet121',
        'parameters': total_params,
        'best_val_acc': float(best_val_acc),
        'final_train_acc': float(history['train_acc'][-1]),
        'epochs_trained': len(history['train_loss']),
        'class_names': train_dataset.classes
    }
    with open(os.path.join(args.save_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"   Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"   Final Training Accuracy: {history['train_acc'][-1]*100:.2f}%")
    print(f"   Epochs Completed: {len(history['train_loss'])}")
    print(f"   Models saved to: {args.save_dir}/")
    print(f"   - best_weights.pth (for deployment)")
    print(f"   - best_model.pth (full checkpoint)")
    print(f"   - training_history.json")
    print(f"   - model_info.json")
    print(f"{'='*60}\n")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train DenseNet121 on medical imaging dataset')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True,
                      help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True,
                      help='Path to validation data directory')
    parser.add_argument('--num_classes', type=int, default=18,
                      help='Number of output classes (default: 18)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay for regularization (default: 1e-5)')
    
    # Scheduler arguments
    parser.add_argument('--patience', type=int, default=3,
                      help='Patience for learning rate scheduler (default: 3)')
    parser.add_argument('--early_stopping', type=int, default=10,
                      help='Early stopping patience (default: 10)')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers (default: 4)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints (default: checkpoints)')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.train_dir):
        print(f"❌ Error: Training directory not found: {args.train_dir}")
        return
    if not os.path.exists(args.val_dir):
        print(f"❌ Error: Validation directory not found: {args.val_dir}")
        return
    
    # Train model
    try:
        model, history = train_model(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
