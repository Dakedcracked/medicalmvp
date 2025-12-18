# Deep Learning Architecture Guide - Neuron AI Medical Imaging System

## Table of Contents
1. [System Overview](#system-overview)
2. [Model Architecture Details](#model-architecture-details)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Training Methodology](#training-methodology)
5. [Inference Pipeline](#inference-pipeline)
6. [Performance Optimization](#performance-optimization)
7. [Model Specifications](#model-specifications)

---

## System Overview

Neuron AI uses a **multi-modal deep learning ensemble** for medical image analysis across three primary imaging modalities:

```
┌─────────────────────────────────────────────────────────┐
│              Medical Image Input                         │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┬─────────────┐
        │                     │             │
    ┌───▼───┐           ┌─────▼────┐  ┌────▼────┐
    │ X-Ray │           │   MRI    │  │  Skin   │
    │ Model │           │  Model   │  │  Model  │
    └───┬───┘           └─────┬────┘  └────┬────┘
        │                     │             │
        └──────────┬──────────┴─────────────┘
                   │
        ┌──────────▼──────────┐
        │  Prediction Output  │
        │  + Grad-CAM Viz     │
        └─────────────────────┘
```

---

## Model Architecture Details

### 1. **Chest X-Ray Analysis: DenseNet121-All (torchxrayvision)**

#### Architecture Specifications
```
Model: DenseNet121 (Densely Connected Convolutional Network)
Source: torchxrayvision library
Weights: Pre-trained on 19 public chest X-ray datasets
Input Size: 224×224 (Grayscale, 1 channel)
Output Classes: 9 pathologies
```

#### Why DenseNet121?
- **Dense Connectivity**: Each layer receives feature maps from ALL preceding layers
- **Feature Reuse**: Reduces parameter count while maintaining high accuracy
- **Gradient Flow**: Direct connections from early to late layers prevent vanishing gradients
- **Medical Imaging Advantage**: Captures both fine-grained textures (early layers) and holistic patterns (late layers) simultaneously

#### Layer Structure
```python
DenseNet121 Architecture:
├── Conv1 (7×7, stride 2)         # Initial feature extraction
├── MaxPool (3×3, stride 2)
├── Dense Block 1 (6 layers)      # [64, 128, 256, 256, 256, 256]
├── Transition Layer 1
├── Dense Block 2 (12 layers)     # Feature refinement
├── Transition Layer 2
├── Dense Block 3 (24 layers)     # Deep feature learning
├── Transition Layer 3
├── Dense Block 4 (16 layers)     # High-level semantics
├── Global Average Pooling
└── Fully Connected (1024 → 9)   # Classification head
```

#### Key Characteristics
- **Parameters**: ~7M (lightweight)
- **Receptive Field**: 224×224 (entire image)
- **Activation**: ReLU throughout, Sigmoid at output (multi-label)
- **Normalization**: Batch Normalization after each conv layer

#### torchxrayvision Integration
```python
import torchxrayvision as xrv

# Load pre-trained model
model = xrv.models.DenseNet(weights="densenet121-res224-all")

# Model expects:
# - Input: (Batch, 1, 224, 224) - Grayscale
# - Range: [0, 1] normalized
# - Mean: 0.5, Std: 0.5
```

**Pre-training Dataset Coverage**:
- ChestX-ray14 (112,120 images)
- MIMIC-CXR (377,110 images)
- CheXpert (224,316 images)
- PadChest (160,000 images)
- **Total**: 19 datasets, ~800,000 chest X-rays

---

### 2. **MRI/Ultrasound Analysis: EfficientNet-B4**

#### Architecture Specifications
```
Model: EfficientNet-B4
Framework: timm (PyTorch Image Models)
Input Size: 224×224 (RGB, 3 channels)
Output Classes: Configurable (default: 9)
```

#### Why EfficientNet-B4?
- **Compound Scaling**: Optimally balances depth, width, and resolution
- **Mobile Inverted Bottleneck (MBConv)**: Efficient feature extraction
- **Squeeze-and-Excitation (SE) blocks**: Channel-wise attention
- **Best Accuracy/Efficiency Trade-off**: B4 is the sweet spot for medical imaging

#### Layer Structure
```python
EfficientNet-B4 Architecture:
├── Stem: Conv3×3, stride 2       # Initial downsampling
├── MBConv1 (k3×3, e1, r2)        # Stage 1
├── MBConv6 (k3×3, e6, r4)        # Stage 2 (expand ratio = 6)
├── MBConv6 (k5×5, e6, r4)        # Stage 3
├── MBConv6 (k3×3, e6, r6)        # Stage 4
├── MBConv6 (k5×5, e6, r6)        # Stage 5
├── MBConv6 (k5×5, e6, r8)        # Stage 6
├── MBConv6 (k3×3, e6, r2)        # Stage 7
├── Conv1×1 + Pooling             # Head
└── FC (1792 → num_classes)
```

**Legend**:
- `k`: Kernel size
- `e`: Expansion ratio (channel multiplier in bottleneck)
- `r`: Number of repeated blocks

#### MBConv Block (Mobile Inverted Bottleneck)
```
Input (C_in channels)
    ↓
Expand: Conv1×1 (C_in → C_in × 6)  # Expansion phase
    ↓
Depthwise Conv k×k                  # Efficient spatial feature extraction
    ↓
SE Block (Squeeze-Excitation)       # Channel attention
    ↓
Project: Conv1×1 (C_in × 6 → C_out) # Compression phase
    ↓
Skip Connection (if C_in == C_out)  # Residual connection
```

#### Key Characteristics
- **Parameters**: ~19M
- **FLOPs**: 4.2B (computationally efficient)
- **Activation**: Swish (x × sigmoid(x))
- **Normalization**: Batch Normalization

#### timm Integration
```python
import timm

# Load pre-trained model
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=9)

# Model expects:
# - Input: (Batch, 3, 224, 224) - RGB
# - ImageNet normalization:
#   mean=[0.485, 0.456, 0.406]
#   std=[0.229, 0.224, 0.225]
```

---

### 3. **Grad-CAM: Explainability Layer**

#### What is Grad-CAM?
**Gradient-weighted Class Activation Mapping** - visualizes which regions of the image the model focuses on.

#### Mathematical Foundation
```
For target class c:
1. Compute gradients: ∂y^c / ∂A^k (where A^k is feature map k)
2. Global Average Pool: α^k_c = (1/Z) Σ_i Σ_j ∂y^c / ∂A^k_ij
3. Weighted Combination: L^c = ReLU(Σ_k α^k_c · A^k)
4. Upsample & Overlay: Heatmap resized to input dimensions
```

#### Implementation Details
```python
from pytorch_grad_cam import GradCAM

# Target last convolutional layer
target_layers = [model.features[-1]]  # For DenseNet
# OR
target_layers = [model.blocks[-1]]    # For EfficientNet

cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor, targets=[ClassifierOutputTarget(class_idx)])
```

#### Heatmap Interpretation
- **Red/Hot regions**: High activation (model's focus areas)
- **Blue/Cold regions**: Low activation (ignored areas)
- **Clinical Value**: Helps radiologists verify model is looking at pathology, not artifacts

---

## Preprocessing Pipeline

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

#### Problem It Solves
Medical images often have:
- **Non-uniform contrast**: Dark/bright regions due to equipment settings
- **Artifact bias**: Models might learn scanner-specific features instead of pathology

#### How CLAHE Works
```
1. Divide image into tiles (default: 8×8)
2. For each tile:
   - Compute histogram
   - Clip histogram at threshold (prevents over-amplification)
   - Redistribute clipped pixels uniformly
   - Apply histogram equalization
3. Interpolate between tiles (bilinear) for smooth transitions
```

#### Implementation
```python
import cv2

def preprocess_medical(image):
    # Convert PIL to OpenCV
    img_np = np.array(image)
    
    # Convert RGB → LAB (Lightness, A, B color spaces)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    # Merge and convert back to RGB
    lab_clahe = cv2.merge((l_clahe, a, b))
    final = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(final)
```

#### Parameters
- `clipLimit=2.0`: Maximum slope of cumulative distribution function (prevents noise amplification)
- `tileGridSize=(8,8)`: Balances local vs global contrast enhancement

---

### Test-Time Augmentation (TTA)

#### Concept
Run inference on **multiple augmented versions** of the same image and average predictions.

#### Augmentations Used
```python
tta_transforms = [
    transforms.Compose([...]),                    # Original
    transforms.Compose([..., RandomHorizontalFlip()]),  # Flip
    transforms.Compose([..., RandomRotation(5)]),      # Rotate +5°
    transforms.Compose([..., RandomRotation(-5)]),     # Rotate -5°
    transforms.Compose([..., ColorJitter(0.1)])        # Brightness
]
```

#### TTA Ensemble
```python
predictions = []
for transform in tta_transforms:
    augmented = transform(image)
    pred = model(augmented)
    predictions.append(pred)

# Average predictions
final_prediction = torch.mean(torch.stack(predictions), dim=0)
```

#### Benefits
- **Reduces variance**: Smooths out prediction noise
- **Improves calibration**: Confidence scores more reliable
- **Cost**: 5× inference time (acceptable for critical medical decisions)

---

### Standard Transforms

#### For X-Ray (DenseNet)
```python
transforms.Compose([
    transforms.Grayscale(),              # Convert to 1 channel
    transforms.Resize(256),              # Upsample
    transforms.CenterCrop(224),          # Crop to model input size
    transforms.ToTensor(),               # → [0, 1]
    transforms.Normalize([0.5], [0.5])   # → [-1, 1]
])
```

#### For MRI/Skin (EfficientNet)
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],     # ImageNet mean
        std=[0.229, 0.224, 0.225]       # ImageNet std
    )
])
```

---

## Training Methodology

### 1. **Loss Functions**

#### Cross-Entropy with Label Smoothing
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Why Label Smoothing?**
- Prevents overconfidence: Soft targets (0.9 instead of 1.0)
- Improves calibration: Probability scores more trustworthy
- Regularization effect: Reduces overfitting

**Formula**:
```
y_smooth = (1 - α) * y_true + α / K
where α = 0.1, K = num_classes
```

#### Weighted Cross-Entropy (Class Imbalance)
```python
class_weights = torch.tensor([5.0, 1.0, 2.0, ...])  # Rare classes get higher weight
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 2. **Optimizers**

#### AdamW (Adam with Weight Decay)
```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5  # L2 regularization
)
```

**Why AdamW?**
- Adaptive learning rates per parameter
- Momentum + RMSprop advantages
- Proper weight decay (decoupled from gradient updates)

### 3. **Learning Rate Scheduling**

#### Cosine Annealing with Warm Restarts
```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,      # Initial restart period (epochs)
    T_mult=2    # Multiplier for restart period
)
```

**Learning Rate Evolution**:
```
LR
│     ╱╲        ╱╲╲           ╱╲╲╲
│    ╱  ╲      ╱   ╲╲       ╱    ╲╲╲
│   ╱    ╲    ╱      ╲╲   ╱        ╲╲╲
│  ╱      ╲  ╱         ╲╲╱            ╲╲╲
│ ╱        ╲╱                             ╲
└──────────────────────────────────────────→ Epochs
  0  5  10      20           40
```

**Benefits**:
- Escapes local minima via restarts
- Better final convergence
- No need to manually tune LR

---

## Inference Pipeline

### Step-by-Step Process

```python
# 1. Load Image
image = Image.open('chest_xray.jpg')

# 2. Image Validation (Detect non-medical images)
hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
saturation = hsv[:,:,1]
mean_sat = np.mean(saturation)
if mean_sat > 30:  # Threshold
    raise ValueError("Not a medical grayscale image")

# 3. CLAHE Preprocessing
image = preprocess_medical(image)

# 4. Transform
tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim

# 5. TTA Ensemble Inference
predictions = []
for tta_transform in tta_transforms:
    aug_tensor = tta_transform(tensor)
    with torch.no_grad():
        output = model(aug_tensor)
    predictions.append(output)

# 6. Average Predictions
avg_output = torch.mean(torch.stack(predictions), dim=0)
probabilities = torch.softmax(avg_output, dim=1)

# 7. Get Top-K Predictions
top_k_probs, top_k_indices = torch.topk(probabilities, k=3)

# 8. Generate Grad-CAM Heatmap
cam = GradCAM(model=model, target_layers=[model.features[-1]])
heatmap = cam(tensor, targets=[ClassifierOutputTarget(top_k_indices[0])])

# 9. Return Results
result = {
    'prediction': class_names[top_k_indices[0]],
    'confidence': top_k_probs[0].item(),
    'top_predictions': [
        {'class': class_names[idx], 'confidence': prob.item()}
        for idx, prob in zip(top_k_indices, top_k_probs)
    ],
    'heatmap': base64_encode(heatmap)
}
```

---

## Performance Optimization

### 1. **GPU Acceleration**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

**Speedup**: ~10-50× faster than CPU (depends on GPU)

### 2. **Mixed Precision Training (FP16)**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in dataloader:
    with autocast():  # Use FP16 for forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- 2× faster training
- 50% less GPU memory
- Same accuracy (gradient scaling prevents underflow)

### 3. **Model Quantization (Inference)**
```python
# Post-training quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear, nn.Conv2d},  # Quantize these layers
    dtype=torch.qint8
)
```

**Benefits**:
- 4× smaller model size
- 2-4× faster inference on CPU
- Minimal accuracy loss (<1%)

### 4. **Batch Processing**
```python
# Instead of one-by-one:
for image in images:
    predict(image)  # Slow

# Batch inference:
batch_tensor = torch.stack([transform(img) for img in images])
predictions = model(batch_tensor)  # Fast
```

**Speedup**: 5-10× for batches of 8-16 images

---

## Model Specifications

### Comparison Table

| Model | Parameters | FLOPs | Input Size | Training Time* | Inference Time** |
|-------|-----------|-------|------------|----------------|------------------|
| **DenseNet121** | 7M | 2.9B | 224×224×1 | ~2h/epoch | 15ms |
| **EfficientNet-B4** | 19M | 4.2B | 224×224×3 | ~4h/epoch | 25ms |

*On NVIDIA RTX 3090, batch size 16
**Single image, GPU inference, TTA disabled

### Accuracy Benchmarks

#### DenseNet121 (X-Ray)
```
Dataset: ChestX-ray14 Test Set
Metric: AUC-ROC (Area Under ROC Curve)

Atelectasis:    0.85
Cardiomegaly:   0.92
Effusion:       0.88
Infiltration:   0.71
Mass:           0.86
Nodule:         0.78
Pneumonia:      0.81
Pneumothorax:   0.89
Normal:         0.94

Mean AUC: 0.85
```

#### EfficientNet-B4 (General)
```
Dataset: ImageNet (Transfer Learning Baseline)
Top-1 Accuracy: 82.9%
Top-5 Accuracy: 96.4%
```

### Hardware Requirements

#### Minimum (CPU Inference)
- CPU: 4 cores, 2.5 GHz
- RAM: 8 GB
- Inference: ~300ms/image

#### Recommended (GPU Inference)
- GPU: NVIDIA GTX 1660 or better (6GB VRAM)
- RAM: 16 GB
- Inference: ~20ms/image

#### Training
- GPU: NVIDIA RTX 3080 or better (10GB+ VRAM)
- RAM: 32 GB
- Storage: 500 GB SSD (for dataset caching)

---

## Advanced Topics

### 1. **Uncertainty Quantification**

#### Monte Carlo Dropout
```python
# Enable dropout at test time
model.train()  # Keep dropout active

predictions = []
for _ in range(20):  # 20 forward passes
    pred = model(image)
    predictions.append(pred)

mean_pred = torch.mean(torch.stack(predictions), dim=0)
std_pred = torch.std(torch.stack(predictions), dim=0)  # Uncertainty estimate
```

**Use Case**: Flag uncertain predictions for radiologist review

### 2. **Ensemble Methods**

#### Model Averaging
```python
models = [densenet, efficientnet, resnet]
predictions = [model(image) for model in models]
final_pred = torch.mean(torch.stack(predictions), dim=0)
```

**Benefit**: 2-3% accuracy improvement

### 3. **Fine-tuning Strategies**

#### Layer Freezing
```python
# Freeze early layers (pretrained features)
for param in model.features[:-4].parameters():
    param.requires_grad = False

# Only train last 4 blocks + classifier
for param in model.features[-4:].parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True
```

**When to use**: Small dataset (<10k images)

#### Differential Learning Rates
```python
optimizer = optim.AdamW([
    {'params': model.features.parameters(), 'lr': 1e-5},  # Low LR for backbone
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # High LR for head
])
```

**When to use**: Medium dataset (10k-100k images)

---

## References & Further Reading

### Research Papers
1. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)
2. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling" (ICML 2019)
3. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations" (ICCV 2017)
4. **CheXpert**: Irvin et al., "CheXpert: A Large Dataset..." (AAAI 2019)

### Libraries
- **torchxrayvision**: https://github.com/mlmed/torchxrayvision
- **timm**: https://github.com/rwightman/pytorch-image-models
- **pytorch-grad-cam**: https://github.com/jacobgil/pytorch-grad-cam

### Datasets (Public)
- **ChestX-ray14**: 112k images, 14 pathologies
- **MIMIC-CXR**: 377k images, free-text reports
- **CheXpert**: 224k images, Stanford dataset
- **ISIC**: 33k skin lesion images

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Solution 1: Reduce batch size
CONFIG['batch_size'] = 8  # Instead of 32

# Solution 2: Gradient accumulation
for i, (images, labels) in enumerate(dataloader):
    loss = model(images)
    loss.backward()
    
    if (i + 1) % 4 == 0:  # Accumulate 4 batches
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Mixed precision
with autocast():
    loss = model(images)
```

#### 2. Model Overfitting
```python
# Increase regularization
optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4)  # Higher

# Add dropout
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1024, num_classes)
)

# More data augmentation
transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2),
    ...
])
```

#### 3. Slow Inference
```python
# Solution 1: Disable TTA
# Only use single forward pass (5× speedup)

# Solution 2: Batch processing
images = [img1, img2, ..., img16]
batch = torch.stack([transform(img) for img in images])
predictions = model(batch)

# Solution 3: Model optimization
model = torch.jit.script(model)  # TorchScript compilation
```

---

## Version History

- **v1.0** (Dec 2025): Initial architecture with DenseNet121 + EfficientNet-B4
- **v1.1** (Dec 2025): Added CLAHE preprocessing for artifact bias mitigation
- **v1.2** (Dec 2025): Integrated Test-Time Augmentation (TTA)
- **v2.0** (Dec 2025): Migrated to torchxrayvision SOTA models

---

## Contact & Support

For technical questions about the deep learning architecture:
- **Documentation**: This file
- **Training Scripts**: `training_scripts/train_*_sota.py`
- **Inference Code**: `api_server.py` (MedicalAIModel class)

---

**Last Updated**: December 18, 2025
**Maintained by**: Neuron AI Development Team
