#!/usr/bin/env python3
"""
Verification script to check if all components are working correctly.
Run this before deploying or pushing to GitHub.
"""

import sys
import os

def check_imports():
    """Check if all required packages can be imported"""
    print("\n" + "="*60)
    print("1. Checking Python Packages...")
    print("="*60)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('numpy', 'NumPy'),
        ('timm', 'timm'),
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name:15s} - OK")
        except ImportError:
            print(f"❌ {name:15s} - MISSING")
            all_good = False
    
    return all_good

def check_custom_models():
    """Check if custom models can be imported"""
    print("\n" + "="*60)
    print("2. Checking Custom Models...")
    print("="*60)
    
    try:
        from models.densenet import DenseNet121
        print("✅ DenseNet121 - OK")
        
        from models.preprocessing import XRayPreprocessor, RGBPreprocessor, DataAugmentation
        print("✅ Preprocessing modules - OK")
        
        return True
    except Exception as e:
        print(f"❌ Error importing models: {e}")
        return False

def check_model_creation():
    """Test model creation"""
    print("\n" + "="*60)
    print("3. Testing Model Creation...")
    print("="*60)
    
    try:
        from models.densenet import DenseNet121
        import torch
        
        model = DenseNet121(num_classes=18)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {total_params:,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 224, 224)
        output = model(dummy_input)
        print(f"✅ Forward pass: {dummy_input.shape} → {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def check_preprocessing():
    """Test preprocessing pipelines"""
    print("\n" + "="*60)
    print("4. Testing Preprocessing...")
    print("="*60)
    
    try:
        from models.preprocessing import XRayPreprocessor
        import numpy as np
        from PIL import Image
        
        # Create dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512), dtype=np.uint8))
        preprocessor = XRayPreprocessor(img_size=224)
        tensor = preprocessor(dummy_img)
        
        print(f"✅ Preprocessing: 512×512 → {tensor.shape}")
        return True
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False

def check_file_structure():
    """Check if all important files exist"""
    print("\n" + "="*60)
    print("5. Checking File Structure...")
    print("="*60)
    
    required_files = [
        'README.md',
        'TRAINING_GUIDE.md',
        'train.py',
        'api_server_custom.py',
        'requirements.txt',
        'models/__init__.py',
        'models/densenet.py',
        'models/preprocessing.py',
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path:30s} ({size:,} bytes)")
        else:
            print(f"❌ {file_path:30s} - MISSING")
            all_good = False
    
    return all_good

def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*60)
    print("6. Checking CUDA/GPU...")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA not available (CPU-only mode)")
        return True
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("\n" + "="*60)
    print("🏥 Neuron AI - Setup Verification")
    print("="*60)
    
    checks = [
        check_imports(),
        check_custom_models(),
        check_model_creation(),
        check_preprocessing(),
        check_file_structure(),
        check_cuda(),
    ]
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all(checks):
        print("✅ All checks passed! System ready for deployment.")
        print("\n📋 Next steps:")
        print("   1. Review README.md and TRAINING_GUIDE.md")
        print("   2. Update GitHub repository URL in README.md")
        print("   3. Test training: python train.py --help")
        print("   4. Push to GitHub: git push origin main")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
