# 🚀 Quick Start Guide

**Get Neuron AI running in 5 minutes!**

---

## ⚡ Super Quick Start

```bash
# 1. Clone and navigate
git clone https://github.com/yourusername/medical-ai-mvp.git
cd medical-ai-mvp

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Verify setup
python verify_setup.py

# 4. Test the model
python models/densenet.py

# 5. Start the application
bash start_servers.sh
```

**Access the app:** http://localhost:3000

---

## 📋 Step-by-Step Installation

### 1. Prerequisites

**Required:**
- Python 3.8 or higher
- pip package manager

**Recommended:**
- NVIDIA GPU with CUDA 11.8+
- 8GB+ RAM
- 10GB free disk space

**Check your setup:**
```bash
python --version          # Should be 3.8+
pip --version            # Should be installed
nvidia-smi               # Check GPU (optional)
```

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/medical-ai-mvp.git
cd medical-ai-mvp
```

### 3. Install PyTorch

**With CUDA (GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch torchvision
```

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- Flask (API server)
- timm (model utilities)
- Pillow (image processing)
- OpenCV (computer vision)
- NumPy (numerical computing)
- And more...

### 5. Verify Installation

```bash
python verify_setup.py
```

**Expected output:**
```
✅ All Python packages installed
✅ Custom models working
✅ Model creation successful
✅ Preprocessing functional
✅ All files present
✅ CUDA available (or CPU mode)
✅ System ready for deployment!
```

### 6. Test the Model

```bash
python models/densenet.py
```

**Expected output:**
```
✅ Model created successfully!
📊 Total parameters: 6,966,034
🏥 Pathologies detected: 18
✅ Forward pass successful!
```

---

## 🎯 Running the Application

### Option 1: Automated Start (Recommended)

```bash
bash start_servers.sh
```

This will start both backend and frontend automatically.

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
python api_server_custom.py
```

**Terminal 2 - Frontend:**
```bash
cd landeros-clone
npm install
npm run dev
```

### Access Points

- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:5000
- **API Health:** http://localhost:5000/api/health

---

## 🎓 Training Your Own Model

### Prepare Your Dataset

Create this structure:
```
my_dataset/
├── train/
│   ├── Pneumonia/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── Normal/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── Pneumonia/
    └── Normal/
```

### Start Training

```bash
python train.py \
    --train_dir my_dataset/train \
    --val_dir my_dataset/val \
    --num_classes 2 \
    --epochs 30 \
    --batch_size 16
```

### Monitor Training

Watch the output:
```
Epoch 1/30: Train Loss: 0.3421, Val Loss: 0.2876, Val Acc: 0.8234
Epoch 2/30: Train Loss: 0.2987, Val Loss: 0.2534, Val Acc: 0.8567
...
✅ Training complete! Best accuracy: 94.32%
```

### Use Trained Model

Your trained model will be saved in:
```
checkpoints/
├── best_weights.pth       ← Use this for deployment
├── training_history.json  ← View training curves
└── model_info.json        ← Model metadata
```

**For detailed training instructions:** [Read TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

## 🔧 Common Issues & Solutions

### Issue: "CUDA not available"

**Solution:**
```bash
# Check if CUDA is installed
nvidia-smi

# If no CUDA, the model will run on CPU (slower but works)
# Or install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads
```

### Issue: "Module not found"

**Solution:**
```bash
# Make sure you're in the project directory
cd medical-ai-mvp

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Port 5000 already in use"

**Solution:**
```bash
# Find and kill the process using port 5000
lsof -ti:5000 | xargs kill -9

# Or use a different port
export FLASK_RUN_PORT=5001
python api_server_custom.py
```

### Issue: "Out of memory (GPU)"

**Solution:**
```bash
# Reduce batch size
python train.py --batch_size 8 ...

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
python train.py ...
```

---

## 📚 Next Steps

### 1. Explore the Documentation
- [README.md](README.md) - Complete project overview
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training tutorial
- [FINAL_TECHNICAL_REPORT.md](FINAL_TECHNICAL_REPORT.md) - Technical details

### 2. Try the API

```python
import requests

# Login
response = requests.post('http://localhost:5000/api/login', json={
    'email': 'test@example.com',
    'password': 'password'
})
token = response.json()['token']

# Analyze image
with open('chest_xray.jpg', 'rb') as f:
    files = {'image': f}
    data = {'modality': 'xray'}
    headers = {'Authorization': f'Bearer {token}'}
    
    response = requests.post(
        'http://localhost:5000/api/predict',
        files=files,
        data=data,
        headers=headers
    )
    
print(response.json())
```

### 3. Train on Your Data

Follow the [TRAINING_GUIDE.md](TRAINING_GUIDE.md) to:
- Prepare your medical imaging dataset
- Configure hyperparameters
- Train the model
- Evaluate performance
- Deploy to production

### 4. Customize the Model

Edit [`models/densenet.py`](models/densenet.py) to:
- Change number of classes
- Adjust model architecture
- Add custom layers
- Modify preprocessing

---

## 🎯 Quick Commands Reference

| Command | Purpose |
|---------|---------|
| `python verify_setup.py` | Verify installation |
| `python models/densenet.py` | Test model creation |
| `python train.py --help` | View training options |
| `python api_server_custom.py` | Start backend API |
| `bash start_servers.sh` | Start full application |

---

## 💡 Pro Tips

1. **Use GPU for training** - 10-100x faster than CPU
2. **Start with small batch size** - Increase gradually if no memory errors
3. **Monitor training curves** - Use training_history.json
4. **Save checkpoints frequently** - Use early stopping
5. **Test on validation set** - Before deploying to production

---

## 🆘 Need Help?

- **Documentation**: Check [README.md](README.md) and [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Verification**: Run `python verify_setup.py`

---

## ✅ Quick Checklist

- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Verification test passed
- [ ] Model test passed
- [ ] Application running

**All checked?** You're ready to go! 🎉

---

<div align="center">

**🚀 Happy Coding!**

[View Full Documentation](README.md) | [Training Guide](TRAINING_GUIDE.md) | [Report Issue](https://github.com/yourusername/medical-ai-mvp/issues)

</div>
