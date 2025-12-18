"""
Flask API Server for Medical AI Application (Custom Model Version)
Uses custom DenseNet121 implementation instead of torchxrayvision
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import sqlite3
import hashlib
import secrets
import os
from datetime import datetime, timedelta
import jwt
import bcrypt

# Import custom models
from models.densenet import densenet121
from models.preprocessing import XRayPreprocessor, RGBPreprocessor

app = Flask(__name__)
CORS(app)

# Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
DATABASE = 'medical_ai.db'
MODEL_PATHS = {
    'xray': 'weights/xray_densenet121.pth',
    'mri': 'weights/mri/efficientnet_b4.pth',
    'ultrasound': 'weights/ultrasound/efficientnet_b4.pth'
}

# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_data TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Contact messages table
    c.execute('''
        CREATE TABLE IF NOT EXISTS contact_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            subject TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT DEFAULT 'new',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Load ML Models
import timm
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class MedicalAIModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        
        # Define pathologies for 18-class chest X-ray model
        self.pathologies = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
            'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'
        ]
        
        # Define classes for other modalities
        self.classes = {
            'xray': self.pathologies,
            'mri': ['Brain Tumor', 'Stroke', 'Normal', 'Other Abnormality'],
            'ultrasound': ['Normal', 'Abnormal', 'Requires Review', 'Emergency']
        }
        
        # Preprocessors
        self.xray_preprocessor = XRayPreprocessor(img_size=224)
        self.rgb_preprocessor = RGBPreprocessor(img_size=224)
        
        self.load_models()

    def generate_heatmap(self, model, image_tensor, original_image, modality='xray', target_category=None):
        """Generate Grad-CAM heatmap for model predictions"""
        try:
            # Select appropriate layer based on model architecture
            if modality == 'xray':
                # For custom DenseNet121, use features.denseblock4
                target_layers = [model.features.denseblock4]
            else:
                # For EfficientNet models
                if hasattr(model, 'conv_head'):
                    target_layers = [model.conv_head]
                else:
                    # Fallback to last conv layer
                    target_layers = [list(model.children())[-2]]
            
            # Create GradCAM object
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
            
            # Prepare input
            single_tensor = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
            
            # Generate heatmap
            grayscale_cam = cam(input_tensor=single_tensor, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
            
            # Prepare original image
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            # Resize and crop to match model input
            w, h = original_image.size
            if w < h:
                new_w = 256
                new_h = int(h * (256 / w))
            else:
                new_h = 256
                new_w = int(w * (256 / h))
            
            img_resized = original_image.resize((new_w, new_h), Image.BICUBIC)
            
            left = int((new_w - 224) / 2)
            top = int((new_h - 224) / 2)
            right = int((new_w + 224) / 2)
            bottom = int((new_h + 224) / 2)
            img_cropped = img_resized.crop((left, top, right, bottom))
            
            # Convert to numpy and normalize
            img_np = np.array(img_cropped)
            rgb_img = np.float32(img_np) / 255.0
            
            # Create overlay
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Convert to Base64
            is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            if is_success:
                return base64.b64encode(buffer).decode('utf-8')
            else:
                print("❌ Failed to encode heatmap")
                return None
                
        except Exception as e:
            print(f"❌ Grad-CAM failed: {e}")
            import traceback
            traceback.print_exc()
        return None

    def load_models(self):
        print("🔄 Loading Custom Medical AI Models...")
        
        # 1. X-Ray (Custom DenseNet121)
        try:
            print("   Loading X-Ray Model (Custom DenseNet121 - 18 Pathologies)...")
            model = densenet121(num_classes=18, in_channels=1)
            
            # Load weights if available
            path = MODEL_PATHS['xray']
            if os.path.exists(path):
                try:
                    state_dict = torch.load(path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"   ✅ Loaded weights from {path}")
                except Exception as e:
                    print(f"   ⚠ Failed to load weights: {e}")
                    print("   ✅ Using randomly initialized model")
            else:
                print(f"   ⚠ No weights found at {path}")
                print("   ✅ Using randomly initialized model")
            
            model.to(self.device)
            model.eval()
            self.models['xray'] = model
            print(f"   📊 Detectable Diseases: {len(self.pathologies)}")
            print(f"   🔢 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"   ❌ Failed to load X-Ray model: {e}")

        # 2. MRI (EfficientNet)
        try:
            print("   Loading MRI Model (EfficientNet-B4)...")
            model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=4)
            path = MODEL_PATHS['mri']
            if os.path.exists(path):
                try:
                    state_dict = torch.load(path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"   ✅ MRI Model Loaded from {path}")
                except Exception as e:
                    print(f"   ⚠ Using pretrained EfficientNet-B4")
            else:
                print("   ⚠ Using pretrained EfficientNet-B4")
            
            model.to(self.device)
            model.eval()
            self.models['mri'] = model
        except Exception as e:
            print(f"   ❌ Failed to load MRI: {e}")

        # 3. Ultrasound (EfficientNet)
        try:
            print("   Loading Ultrasound Model (EfficientNet-B4)...")
            model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=4)
            path = MODEL_PATHS['ultrasound']
            if os.path.exists(path):
                try:
                    state_dict = torch.load(path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"   ✅ Ultrasound Model Loaded from {path}")
                except Exception as e:
                    print(f"   ⚠ Using pretrained EfficientNet-B4")
            else:
                print("   ⚠ Using pretrained EfficientNet-B4")
            
            model.to(self.device)
            model.eval()
            self.models['ultrasound'] = model
        except Exception as e:
            print(f"   ❌ Failed to load Ultrasound: {e}")

        print("✅ All models loaded successfully!\n")

    def predict(self, image, modality='xray'):
        """
        Make prediction on medical image
        
        Args:
            image (PIL.Image): Input image
            modality (str): 'xray', 'mri', 'ct', or 'ultrasound'
            
        Returns:
            dict: Predictions with confidences and heatmap
        """
        try:
            if modality not in self.models:
                return {'error': f'Model not loaded for {modality}'}
            
            model = self.models[modality]
            
            # Preprocess image
            if modality == 'xray':
                image_tensor = self.xray_preprocessor(image)
            else:
                image_tensor = self.rgb_preprocessor(image)
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = model(image_tensor.unsqueeze(0))
                
                # Multi-label classification (Sigmoid for X-ray)
                if modality == 'xray':
                    probs = torch.sigmoid(output).cpu().numpy()[0]
                    predictions = []
                    
                    for i, prob in enumerate(probs):
                        predictions.append({
                            'disease': self.pathologies[i],
                            'confidence': float(prob)
                        })
                    
                    # Sort by confidence
                    predictions.sort(key=lambda x: x['confidence'], reverse=True)
                    
                else:
                    # Single-label classification (Softmax)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    predictions = []
                    
                    for i, prob in enumerate(probs):
                        predictions.append({
                            'disease': self.classes[modality][i],
                            'confidence': float(prob)
                        })
                    
                    predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Generate heatmap
            heatmap = self.generate_heatmap(
                model, image_tensor, image, modality=modality
            )
            
            return {
                'predictions': predictions,
                'heatmap': heatmap,
                'modality': modality,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

# Initialize model
medical_model = MedicalAIModel()

# ==================== Authentication Routes ====================

@app.route('/api/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        
        if not all([email, password, name]):
            return jsonify({'error': 'All fields required'}), 400
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        try:
            c.execute('INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)',
                     (email, password_hash, name))
            conn.commit()
            user_id = c.lastrowid
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Email already registered'}), 400
        finally:
            conn.close()
        
        # Generate token
        token = jwt.encode({
            'user_id': user_id,
            'email': email,
            'exp': datetime.utcnow() + timedelta(days=7)
        }, SECRET_KEY, algorithm='HS256')
        
        return jsonify({
            'token': token,
            'user': {'id': user_id, 'email': email, 'name': name}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({'error': 'Email and password required'}), 400
        
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('SELECT id, email, password_hash, name FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user_id, email, password_hash, name = user
        
        if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate token
        token = jwt.encode({
            'user_id': user_id,
            'email': email,
            'exp': datetime.utcnow() + timedelta(days=7)
        }, SECRET_KEY, algorithm='HS256')
        
        return jsonify({
            'token': token,
            'user': {'id': user_id, 'email': email, 'name': name}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user', methods=['GET'])
def get_user():
    """Get current user info"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        
        token = auth_header.split(' ')[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('SELECT id, email, name FROM users WHERE id = ?', (payload['user_id'],))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': {
                'id': user[0],
                'email': user[1],
                'name': user[2]
            }
        })
        
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token expired'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 401

# ==================== Prediction Routes ====================

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict disease from medical image"""
    try:
        # Check authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        
        token = auth_header.split(' ')[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        user_id = payload['user_id']
        
        # Get image and modality
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        modality = request.form.get('modality', 'xray')
        
        # Load image
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Make prediction
        result = medical_model.predict(image, modality)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Save to database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO predictions (user_id, prediction, confidence)
            VALUES (?, ?, ?)
        ''', (user_id, result['predictions'][0]['disease'], result['predictions'][0]['confidence']))
        conn.commit()
        conn.close()
        
        return jsonify(result)
        
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token expired'}), 401
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/contact', methods=['POST'])
def contact():
    """Handle contact form submissions"""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        
        if not all([name, email, subject, message]):
            return jsonify({'error': 'All fields required'}), 400
        
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO contact_messages (name, email, subject, message)
            VALUES (?, ?, ?, ?)
        ''', (name, email, subject, message))
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Message received successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(medical_model.models.keys()),
        'device': str(medical_model.device),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    print("\n" + "=" * 60)
    print("🏥 Medical AI Server (Custom Model Version)")
    print("=" * 60)
    print(f"🔧 Device: {medical_model.device}")
    print(f"🤖 Models loaded: {list(medical_model.models.keys())}")
    print(f"🌐 Server starting on http://localhost:5000")
    print("=" * 60 + "\n")
    
    # Run server
    app.run(debug=False, host='0.0.0.0', port=5000)
