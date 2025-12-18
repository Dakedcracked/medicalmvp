"""
Flask API Server for Medical AI Application
Handles user authentication and medical image predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
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

app = Flask(__name__)
CORS(app)

# Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
DATABASE = 'medical_ai.db'
MODEL_PATHS = {
    'xray': 'weights/xray/densenet_sota.pth',
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
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import torchxrayvision as xrv

class MedicalAIModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        
        # Define classes for each modality
        self.classes = {
            'xray': [
                'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Normal'
            ]
        }
        
        # NEW TRANSFORM: Preserves Aspect Ratio to prevent "Squashed Heart" error
        self.transform = transforms.Compose([
            transforms.Resize(256),        # Resize shortest side to 256
            transforms.CenterCrop(224),    # Crop center 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.load_models()
    
    def preprocess_medical(self, image):
        """
        Advanced Preprocessing (CLAHE) to fix 'Artifact Bias'
        Standardizes contrast so models aren't fooled by dark/light scans.
        """
        # Convert PIL to OpenCV
        img_np = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel (Lightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge and convert back
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(final)

    def validate_image(self, image):
        """Check if image is likely a medical scan (grayscale-ish)"""
        # Convert PIL to CV2 format (RGB)
        img_np = np.array(image)
        
        # Calculate color saturation
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        saturation = hsv[:,:,1]
        mean_sat = np.mean(saturation)
        
        # Threshold: Medical images usually have very low saturation (< 20)
        # Random photos usually have > 50
        if mean_sat > 30:
            return False, f"Image rejected: High color saturation detected ({mean_sat:.1f}). Please upload a medical grayscale scan."
            
        return True, ""

    def generate_heatmap(self, model, image_tensor, original_image):
        """Generate Grad-CAM Heatmap (Uses first model in ensemble if list)"""
        try:
            # Handle Ensemble: Use the first model for Grad-CAM
            target_model = model[0] if isinstance(model, list) else model
            
            # Target the last block of the model (ConvNeXt specific, need adjust for others)
            # For EfficientNet: conv_head or blocks[-1]
            # For DenseNet: features.denseblock4
            
            target_layers = None
            if 'efficientnet' in str(type(target_model)).lower():
                 target_layers = [target_model.conv_head]
            elif 'densenet' in str(type(target_model)).lower():
                 # For TorchXRayVision DenseNet
                 target_layers = [target_model.features.denseblock4.denselayer16] 
            else:
                 # Fallback for ConvNeXt
                 target_layers = [target_model.stages[-1].blocks[-1]]

            if not target_layers:
                 return None

            cam = GradCAM(model=target_model, target_layers=target_layers)
            
            # Generate CAM
            grayscale_cam = cam(input_tensor=image_tensor)
            grayscale_cam = grayscale_cam[0, :]
            
            # Prepare original image for overlay (Must match the Crop!)
            # 1. Resize shortest side to 256
            w, h = original_image.size
            if w < h:
                new_w = 256
                new_h = int(h * (256 / w))
            else:
                new_h = 256
                new_w = int(w * (256 / h))
                
            img_resized = original_image.resize((new_w, new_h), Image.BICUBIC)
            
            # 2. Center Crop 224
            left = (new_w - 224) / 2
            top = (new_h - 224) / 2
            right = (new_w + 224) / 2
            bottom = (new_h + 224) / 2
            img_cropped = img_resized.crop((left, top, right, bottom))
            
            img_np = np.array(img_cropped)
            rgb_img = np.float32(img_np) / 255.0
            
            # Create overlay
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Convert to Base64
            is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            if is_success:
                return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
        return None

    def load_models(self):
        print("🔄 Loading Medical AI Models...")
        
        # 1. X-Ray (DenseNet SOTA)
        try:
            print("   Loading X-Ray Model (DenseNet121-All)...")
            # Re-create the exact architecture used in training
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 9) # 9 Classes
            
            path = MODEL_PATHS['xray']
            if os.path.exists(path):
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                print("   ✅ X-Ray Model Loaded!")
            else:
                print(f"   ⚠ Weights not found at {path}")
            
            model.to(self.device)
            model.eval()
            self.models['xray'] = model
        except Exception as e:
            print(f"   ❌ Failed to load X-Ray: {e}")

        # 2. MRI (EfficientNet)
        try:
            print("   Loading MRI Model (EfficientNet-B4)...")
            model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=4)
            path = MODEL_PATHS['mri']
            if os.path.exists(path):
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                print("   ✅ MRI Model Loaded!")
            
            model.to(self.device)
            model.eval()
            self.models['mri'] = model
        except Exception as e:
            print(f"   ❌ Failed to load MRI: {e}")

    def predict(self, image_bytes):
        if not self.models:
            return {"error": "No models loaded"}, 500
        
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # 1. SAFETY CHECK: Is this a medical image?
            is_valid, reason = self.validate_image(image)
            if not is_valid:
                return {"error": reason, "is_rejected": True}, 400

            # 1.5. ARTIFACT REMOVAL (CLAHE)
            image = self.preprocess_medical(image)

            # Preprocess with Test-Time Augmentation (TTA)
            # 1. Original
            t_orig = self.transform(image)
            
            # 2. Horizontal Flip
            t_flip = torch.flip(t_orig, [2])
            
            # 3. Zoom (Center Crop 90%)
            w, h = image.size
            crop_size = int(min(w, h) * 0.9)
            left = (w - crop_size)/2
            top = (h - crop_size)/2
            img_zoom = image.crop((left, top, left+crop_size, top+crop_size))
            t_zoom = self.transform(img_zoom)
            
            # Stack batch: [3, C, H, W]
            image_tensor = torch.stack([t_orig, t_flip, t_zoom]).to(self.device)
            
            # 2. SELECT MODEL (Default to X-Ray for now, can be dynamic later)
            target_modality = 'xray' 
            model_obj = self.models.get(target_modality)
            
            if not model_obj:
                 return {"error": f"Model for {target_modality} not available"}, 500

            # PREDICTION
            probabilities = None
            
            with torch.no_grad():
                # TTA: Average across batch (dim=0)
                out = model_obj(image_tensor) # [3, 9]
                probs = torch.softmax(out, dim=1)
                probabilities = probs.mean(dim=0)

                confidence, predicted_idx = torch.max(probabilities, 0)
            
            class_list = self.classes[target_modality]
            
            # --- SAFETY NET PROTOCOL ---
            normal_idx = 8 # Assuming Normal is last
            normal_prob = probabilities[normal_idx].item()
            
            # Critical Indices: 1=Cardio, 4=Mass, 5=Nodule, 6=Pneumonia, 7=Pneumothorax
            critical_indices = [1, 4, 5, 6, 7] 
            
            override_disease = None
            override_conf = 0.0
            
            for idx in critical_indices:
                prob = probabilities[idx].item()
                if prob > 0.15 and normal_prob < 0.90:
                    if prob > override_conf:
                        override_conf = prob
                        override_disease = idx

            if override_disease is not None:
                predicted_idx = torch.tensor(override_disease)
                confidence = torch.tensor(override_conf)
                print(f"   ⚠️ Safety Net Triggered! Found {class_list[override_disease]} ({override_conf:.2f})")
            
            prediction = class_list[predicted_idx.item()]
            
            # SAFETY GUARD: UNCERTANTY THRESHOLD
            THRESHOLD = 0.35
            
            if confidence.item() < THRESHOLD:
                prediction = f"Uncertain ({confidence.item()*100:.1f}%) - Clinical Review Required"
            
            # 3. EXPLAINABILITY: Generate Grad-CAM
            # Use the first image in the batch (Original) for heatmap
            heatmap_b64 = self.generate_heatmap(model_obj, image_tensor[0].unsqueeze(0), image)
            
            # Get top 3
            top3_prob, top3_idx = torch.topk(probabilities, min(3, len(class_list)))
            top_predictions = [
                {
                    "class": class_list[i.item()],
                    "confidence": p.item()
                }
                for p, i in zip(top3_prob, top3_idx)
            ]
            
            return {
                "prediction": f"{prediction}",
                "confidence": confidence.item(),
                "top_predictions": top_predictions,
                "modality": "Chest X-Ray",
                "heatmap": heatmap_b64
            }, 200
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            return {"error": str(e)}, 500

# Initialize model
ml_model = MedicalAIModel()

# Helper functions
def hash_password(password):
    """Hash password using bcrypt for secure storage"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password, hashed):
    """Verify password against bcrypt hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_token(user_id, email):
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": ml_model.model is not None,
        "device": str(ml_model.device)
    })

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    
    if not email or not password or not name:
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        # Check if user exists
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if c.fetchone():
            return jsonify({"error": "Email already registered"}), 400
        
        # Create user
        password_hash = hash_password(password)
        c.execute(
            'INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)',
            (email, password_hash, name)
        )
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Generate token
        token = generate_token(user_id, email)
        
        return jsonify({
            "message": "User created successfully",
            "token": token,
            "user": {"id": user_id, "email": email, "name": name}
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400
    
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        c.execute(
            'SELECT id, email, name, password_hash FROM users WHERE email = ?',
            (email,)
        )
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401
        
        user_id, user_email, name, password_hash = user
        
        # Verify password with bcrypt
        if not verify_password(password, password_hash):
            return jsonify({"error": "Invalid credentials"}), 401
        
        token = generate_token(user_id, user_email)
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {"id": user_id, "email": user_email, "name": name}
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/verify', methods=['GET'])
def verify():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "No token provided"}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    
    if not payload:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    return jsonify({"valid": True, "user_id": payload['user_id']}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    # Verify authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Authentication required"}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    
    if not payload:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    user_id = payload['user_id']
    
    # Check if file is present
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    # Get selected modality (default to xray) - IGNORED
    # modality = request.form.get('modality', 'xray')

    try:
        # Read image
        image_bytes = file.read()
        
        # Make prediction (Always X-Ray)
        result, status_code = ml_model.predict(image_bytes)
        
        if status_code != 200:
            return jsonify(result), status_code
        
        # Save prediction to database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        # Convert image to base64 for storage (optional)
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        c.execute(
            'INSERT INTO predictions (user_id, image_data, prediction, confidence) VALUES (?, ?, ?, ?)',
            (user_id, image_b64[:1000], result['prediction'], result['confidence'])  # Store first 1000 chars
        )
        prediction_id = c.lastrowid
        conn.commit()
        conn.close()
        
        result['prediction_id'] = prediction_id
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/history', methods=['GET'])
def get_history():
    # Verify authentication
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Authentication required"}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    
    if not payload:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    user_id = payload['user_id']
    
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        c.execute(
            'SELECT id, prediction, confidence, created_at FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 50',
            (user_id,)
        )
        predictions = c.fetchall()
        conn.close()
        
        history = [
            {
                "id": p[0],
                "prediction": p[1],
                "confidence": p[2],
                "created_at": p[3]
            }
            for p in predictions
        ]
        
        return jsonify({"history": history}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/contact', methods=['POST'])
def contact():
    """Handle contact form submissions"""
    data = request.json
    name = data.get('name')
    email = data.get('email')
    subject = data.get('subject')
    message = data.get('message')
    
    if not all([name, email, subject, message]):
        return jsonify({"error": "All fields are required"}), 400
    
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        c.execute(
            'INSERT INTO contact_messages (name, email, subject, message) VALUES (?, ?, ?, ?)',
            (name, email, subject, message)
        )
        message_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": "Message received successfully",
            "id": message_id
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    print("=" * 60)
    print("🏥 Medical AI API Server")
    print("=" * 60)
    print(f"Database: {DATABASE}")
    print(f"Models: {list(MODEL_PATHS.keys())}")
    print(f"Device: {ml_model.device}")
    print(f"Models Loaded: {len(ml_model.models)} / 3")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
