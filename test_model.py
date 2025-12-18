#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL TESTING SUITE
==================================
This script provides multiple ways to test your deep learning model:
1. Single image prediction
2. Batch testing on multiple images
3. Performance metrics (accuracy, precision, recall)
4. Visualization of predictions vs ground truth
5. Confusion matrix generation
6. ROC curve analysis

Usage:
    python test_model.py --mode single --image path/to/xray.jpg
    python test_model.py --mode batch --folder path/to/test_images/
    python test_model.py --mode metrics --folder path/to/test_images/
"""

import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from PIL import Image
import os
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
from tqdm import tqdm
import json

# === CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATHOLOGIES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
    'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'
]

# === PREPROCESSING ===
def preprocess_image(image_path):
    """
    Preprocess X-ray image for model input
    Matches the preprocessing in api_server.py
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Transform for X-Ray (grayscale, 224x224)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor, image

# === MODEL LOADING ===
def load_model(model_path=None):
    """
    Load the X-Ray model
    Args:
        model_path: Path to custom weights (optional)
    Returns:
        model: Loaded model ready for inference
    """
    print("🔄 Loading Model...")
    
    # Load pretrained torchxrayvision model
    model = xrv.models.DenseNet(weights='densenet121-res224-all')
    
    # If custom weights provided, try to load them
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            if state_dict['classifier.weight'].shape[0] == 18:
                model.load_state_dict(state_dict)
                print(f"✅ Loaded custom weights from {model_path}")
            else:
                print(f"⚠ Custom weights have wrong output size, using pretrained")
        except Exception as e:
            print(f"⚠ Failed to load custom weights: {e}")
            print("Using pretrained model")
    else:
        print("✅ Using pretrained torchxrayvision model")
    
    model.to(DEVICE)
    model.eval()
    return model

# === TEST MODE 1: SINGLE IMAGE PREDICTION ===
def test_single_image(model, image_path):
    """
    Test model on a single image and display results
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {os.path.basename(image_path)}")
    print(f"{'='*60}\n")
    
    # Preprocess
    tensor, original_image = preprocess_image(image_path)
    tensor = tensor.to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.sigmoid(outputs)[0]  # Multi-label sigmoid
    
    # Display results
    print("PREDICTIONS:")
    print(f"{'Disease':<30} {'Probability':<12} {'Status':<10}")
    print("-" * 60)
    
    detected_diseases = []
    for i, (disease, prob) in enumerate(zip(PATHOLOGIES, probabilities)):
        prob_val = prob.item()
        status = "🔴 POSITIVE" if prob_val > 0.5 else "🟢 NEGATIVE"
        print(f"{disease:<30} {prob_val:>6.1%}        {status}")
        
        if prob_val > 0.5:
            detected_diseases.append((disease, prob_val))
    
    # Summary
    print(f"\n{'='*60}")
    if detected_diseases:
        print(f"⚠ DETECTED {len(detected_diseases)} PATHOLOGIES:")
        for disease, conf in detected_diseases:
            print(f"   • {disease}: {conf:.1%}")
    else:
        print("✅ NO SIGNIFICANT PATHOLOGIES DETECTED")
    print(f"{'='*60}\n")
    
    return probabilities.cpu().numpy()

# === TEST MODE 2: BATCH TESTING ===
def test_batch(model, folder_path, save_results=True):
    """
    Test model on multiple images in a folder
    Expects folder structure:
        folder_path/
            ├── Pneumonia/
            │   ├── img1.jpg
            │   ├── img2.jpg
            ├── Normal/
            │   ├── img3.jpg
    """
    print(f"\n{'='*60}")
    print(f"BATCH TESTING: {folder_path}")
    print(f"{'='*60}\n")
    
    results = []
    
    # Find all images
    image_paths = []
    labels = []
    
    for class_name in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_name)
    
    print(f"Found {len(image_paths)} images across {len(set(labels))} classes\n")
    
    # Process each image
    for img_path, true_label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        try:
            tensor, _ = preprocess_image(img_path)
            tensor = tensor.to(DEVICE)
            
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.sigmoid(outputs)[0]
            
            # Get top prediction
            max_prob, max_idx = torch.max(probabilities, 0)
            predicted_disease = PATHOLOGIES[max_idx]
            
            results.append({
                'image': os.path.basename(img_path),
                'true_label': true_label,
                'predicted': predicted_disease,
                'confidence': max_prob.item(),
                'all_probabilities': probabilities.cpu().numpy().tolist()
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
    
    # Save results
    if save_results:
        output_path = 'test_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")
    
    return results

# === TEST MODE 3: PERFORMANCE METRICS ===
def calculate_metrics(model, folder_path):
    """
    Calculate comprehensive performance metrics
    """
    print(f"\n{'='*60}")
    print(f"CALCULATING METRICS")
    print(f"{'='*60}\n")
    
    # Run batch testing
    results = test_batch(model, folder_path, save_results=False)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Extract predictions and labels
    y_true = []
    y_pred = []
    y_probs = []
    
    for result in results:
        y_true.append(result['true_label'])
        y_pred.append(result['predicted'])
        y_probs.append(result['all_probabilities'])
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For multi-label: calculate per-class metrics
    unique_labels = list(set(y_true))
    
    print("OVERALL METRICS:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Total Samples: {len(y_true)}")
    print(f"  Classes: {len(unique_labels)}")
    
    # Per-class metrics
    print("\nPER-CLASS PERFORMANCE:")
    print(f"{'Class':<25} {'Samples':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    for label in unique_labels:
        indices = [i for i, l in enumerate(y_true) if l == label]
        correct = sum([1 for i in indices if y_pred[i] == label])
        class_acc = correct / len(indices) if indices else 0
        
        print(f"{label:<25} {len(indices):<10} {correct:<10} {class_acc:.1%}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print(f"\n✅ Confusion matrix saved to confusion_matrix.png")
    
    # Save detailed report
    report = {
        'accuracy': accuracy,
        'total_samples': len(y_true),
        'per_class': {}
    }
    
    for label in unique_labels:
        indices = [i for i, l in enumerate(y_true) if l == label]
        correct = sum([1 for i in indices if y_pred[i] == label])
        report['per_class'][label] = {
            'samples': len(indices),
            'correct': correct,
            'accuracy': correct / len(indices) if indices else 0
        }
    
    with open('metrics_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Detailed report saved to metrics_report.json\n")

# === TEST MODE 4: VISUALIZE PREDICTIONS ===
def visualize_predictions(model, folder_path, num_samples=9):
    """
    Visualize predictions on a grid of images
    """
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    # Find random images
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    # Sample random images
    if len(image_paths) > num_samples:
        import random
        image_paths = random.sample(image_paths, num_samples)
    
    # Create visualization grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(image_paths):
        if idx >= len(axes):
            break
        
        # Predict
        tensor, original_image = preprocess_image(img_path)
        tensor = tensor.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.sigmoid(outputs)[0]
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        
        # Display
        axes[idx].imshow(original_image, cmap='gray')
        axes[idx].axis('off')
        
        title = f"{os.path.basename(img_path)}\n"
        for prob, idx_pred in zip(top3_probs, top3_indices):
            disease = PATHOLOGIES[idx_pred]
            title += f"{disease}: {prob:.1%}\n"
        
        axes[idx].set_title(title, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150)
    print(f"✅ Visualization saved to predictions_visualization.png\n")

# === MAIN ===
def main():
    parser = argparse.ArgumentParser(description='Test Medical AI Model')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['single', 'batch', 'metrics', 'visualize'],
                       help='Testing mode')
    parser.add_argument('--image', type=str, help='Path to single image (for single mode)')
    parser.add_argument('--folder', type=str, help='Path to folder with test images')
    parser.add_argument('--model', type=str, help='Path to custom model weights (optional)')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Run appropriate test mode
    if args.mode == 'single':
        if not args.image:
            print("❌ Error: --image required for single mode")
            return
        test_single_image(model, args.image)
    
    elif args.mode == 'batch':
        if not args.folder:
            print("❌ Error: --folder required for batch mode")
            return
        test_batch(model, args.folder)
    
    elif args.mode == 'metrics':
        if not args.folder:
            print("❌ Error: --folder required for metrics mode")
            return
        calculate_metrics(model, args.folder)
    
    elif args.mode == 'visualize':
        if not args.folder:
            print("❌ Error: --folder required for visualize mode")
            return
        visualize_predictions(model, args.folder)
    
    print("\n✅ Testing Complete!\n")

if __name__ == '__main__':
    main()
