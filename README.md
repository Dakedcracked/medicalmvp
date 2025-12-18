# Neuron AI - Medical Imaging Platform

## 🏥 Custom Model Implementation

This version uses **custom DenseNet121 architecture** without external dependencies (no torchxrayvision).

### 📁 Project Structure

```
medical-ai-mvp/
├── api_server_custom.py         # Flask API with custom models
├── api_server.py                # Original API (with torchxrayvision)
├── models/                      # Custom model architectures
│   ├── __init__.py
│   ├── densenet.py             # DenseNet121 implementation
│   └── preprocessing.py        # Image preprocessing
├── landeros-clone/              # Next.js frontend
├── weights/                     # Model weights
└── training_scripts/            # Training scripts
```

### 🚀 Quick Start

```bash
# 1. Start backend server (custom model version)
python api_server_custom.py

# 2. Start frontend (separate terminal)
cd landeros-clone
npm run dev

# 3. Access application
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000
```

### �� Model Architecture

**Custom DenseNet121** (No torchxrayvision dependency)
- **Architecture**: DenseNet-121 (6, 12, 24, 16 blocks)
- **Input**: 224×224 grayscale (1 channel)
- **Output**: 18 pathology predictions
- **Parameters**: 6,966,034 (~7 million)
- **Growth rate**: 32

**Detected Pathologies** (18 classes):
1. Atelectasis
2. Consolidation
3. Infiltration
4. Pneumothorax
5. Edema
6. Emphysema
7. Fibrosis
8. Effusion
9. Pneumonia
10. Pleural_Thickening
11. Cardiomegaly
12. Nodule
13. Mass
14. Hernia
15. Lung Lesion
16. Fracture
17. Lung Opacity
18. Enlarged Cardiomediastinum

### 📊 Test Model

```bash
# Test custom DenseNet121
python models/densenet.py

# Expected output:
# ✅ Model created successfully!
# 📊 Total parameters: 6,966,034
# 🏥 Pathologies detected: 18
```

### 🔬 Preprocessing

```python
from models.preprocessing import XRayPreprocessor

preprocessor = XRayPreprocessor(img_size=224)
tensor = preprocessor(image)  # PIL Image → Tensor
```

### 📡 API Endpoints

**Authentication:**
- `POST /api/register` - Register new user
- `POST /api/login` - Login user
- `GET /api/user` - Get user info

**Prediction:**
- `POST /api/predict` - Analyze medical image

**Health:**
- `GET /api/health` - Server status

### 📦 Dependencies

```
torch>=2.0.0
torchvision
timm
flask
flask-cors
pillow
opencv-python
pytorch-grad-cam
bcrypt
pyjwt
numpy
```

### 🏗️ Architecture Comparison

| Feature | TorchXRayVision | Custom Implementation |
|---------|----------------|----------------------|
| Dependencies | External library | Self-contained |
| Customization | Limited | Full control |
| Training | Pretrained only | Train from scratch |
| Flexibility | Fixed | Fully modifiable |

---

**✅ Custom model implementation complete - No external medical imaging dependencies required!**
