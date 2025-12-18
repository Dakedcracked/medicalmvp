import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
import torchxrayvision as xrv
import os
import time
import copy
from collections import Counter

# --- CONFIGURATION ---
DATA_ROOT = '../Calibration_Dataset' 
WEIGHTS_ROOT = '../weights'
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
IMG_SIZE = 224 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure weights sub-directories exist
os.makedirs(os.path.join(WEIGHTS_ROOT, 'xray'), exist_ok=True)
os.makedirs(os.path.join(WEIGHTS_ROOT, 'mri'), exist_ok=True)
os.makedirs(os.path.join(WEIGHTS_ROOT, 'ultrasound'), exist_ok=True)

def get_transforms(mode='train', img_size=224):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def train_model(model_name, modality, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print(f"\n🚀 Starting training for {modality}/{model_name} on {DEVICE}")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Check if dataloader exists for this phase
            if phase not in dataloaders:
                continue

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save to specific modality folder
                save_path = os.path.join(WEIGHTS_ROOT, modality, f'{model_name}.pth')
                torch.save(model.state_dict(), save_path)
                print(f"   💾 New best model saved to {save_path}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def setup_and_train(modality, data_dir, num_classes):
    print(f"\n=== Setting up {modality.upper()} Pipeline ===")
    
    # 1. Data Setup
    if os.path.exists(data_dir):
        print(f"📂 Loading Data from {data_dir}...")
        full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms('train'))
        
        # Split 80/20
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset.transform = get_transforms('val') 

        # Handle Imbalance
        targets = [label for _, label in full_dataset.samples]
        class_counts = Counter(targets)
        weights = [1.0 / class_counts[i] for i in range(len(class_counts))]
        samples_weights = [weights[t] for t in targets]
        train_indices = train_dataset.indices
        train_weights = [samples_weights[i] for i in train_indices]
        sampler = WeightedRandomSampler(train_weights, len(train_weights))

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        }
    else:
        print(f"⚠ Data directory {data_dir} not found. Skipping training, downloading pre-trained weights only.")
        dataloaders = {}

    # 2. Model Setup & Training
    
    # --- X-RAY STRATEGY: TorchXRayVision (DenseNet) ---
    if modality == 'xray':
        print("🏗️  Initializing SOTA X-Ray Model (DenseNet121-all)...")
        # Load model pre-trained on ALL major datasets (RSNA, CheXpert, NIH, etc.)
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        # Modify classifier for our specific classes
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model = model.to(DEVICE)
        
        if dataloaders:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
            train_model('densenet_sota', modality, model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS)
        else:
            # Save the pre-trained base model if no data to fine-tune
            save_path = os.path.join(WEIGHTS_ROOT, modality, 'densenet_sota.pth')
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Pre-trained SOTA model saved to {save_path}")

    # --- MRI & ULTRASOUND STRATEGY: EfficientNet Transfer Learning ---
    else:
        print(f"🏗️  Initializing SOTA {modality.upper()} Model (EfficientNet-B4)...")
        # EfficientNet-B4 is excellent for texture (Ultrasound) and structure (MRI)
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        model = model.to(DEVICE)
        
        if dataloaders:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
            train_model('efficientnet_b4', modality, model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS)
        else:
             save_path = os.path.join(WEIGHTS_ROOT, modality, 'efficientnet_b4.pth')
             torch.save(model.state_dict(), save_path)
             print(f"   💾 Pre-trained ImageNet model saved to {save_path}")

if __name__ == '__main__':
    # Define your modalities and their specific configurations
    
    # 1. Chest X-Ray (9 Classes)
    # Target: Pneumonia, Normal, Cardiomegaly, etc.
    setup_and_train(
        modality='xray', 
        data_dir=os.path.join(DATA_ROOT, 'xray_data'), # Expects subfolder
        num_classes=9
    )
    
    # 2. MRI (Brain Tumor - 4 Classes)
    # Target: Glioma, Meningioma, Pituitary, No Tumor
    setup_and_train(
        modality='mri', 
        data_dir=os.path.join(DATA_ROOT, 'mri_data'),
        num_classes=4
    )
    
    # 3. Ultrasound (Breast Cancer - 3 Classes)
    # Target: Benign, Malignant, Normal
    setup_and_train(
        modality='ultrasound', 
        data_dir=os.path.join(DATA_ROOT, 'ultrasound_data'),
        num_classes=3
    )

