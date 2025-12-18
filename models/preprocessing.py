"""
Image Preprocessing for Medical Imaging
Handles X-ray, CT, MRI, and Ultrasound image preprocessing
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


class XRayPreprocessor:
    """Preprocessing pipeline for chest X-ray images"""
    
    def __init__(self, img_size=224):
        """
        Args:
            img_size (int): Target image size (default: 224)
        """
        self.img_size = img_size
        
        # Standard X-ray transforms (grayscale, 1 channel)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Single channel normalization
        ])
    
    def __call__(self, image):
        """
        Preprocess X-ray image
        
        Args:
            image (PIL.Image or np.ndarray): Input image
            
        Returns:
            torch.Tensor: Preprocessed tensor [1, 224, 224]
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.transform(image)
    
    def preprocess_with_clahe(self, image):
        """
        Preprocess with CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Improves contrast in medical images
        
        Args:
            image (PIL.Image or np.ndarray): Input image
            
        Returns:
            torch.Tensor: Preprocessed tensor with CLAHE enhancement
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('L'))
        else:
            img_array = image
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_array)
        
        # Convert back to PIL and apply standard transforms
        enhanced_pil = Image.fromarray(enhanced)
        return self.transform(enhanced_pil)


class RGBPreprocessor:
    """Preprocessing pipeline for CT, MRI, and Ultrasound (RGB/color images)"""
    
    def __init__(self, img_size=224):
        """
        Args:
            img_size (int): Target image size (default: 224)
        """
        self.img_size = img_size
        
        # Standard ImageNet normalization for RGB images
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, image):
        """
        Preprocess RGB medical image
        
        Args:
            image (PIL.Image or np.ndarray): Input image
            
        Returns:
            torch.Tensor: Preprocessed tensor [3, 224, 224]
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return self.transform(image)


class DataAugmentation:
    """Data augmentation for training medical models"""
    
    @staticmethod
    def get_train_transforms(modality='xray', img_size=224):
        """
        Get training transforms with augmentation
        
        Args:
            modality (str): 'xray', 'ct', 'mri', or 'ultrasound'
            img_size (int): Target image size
            
        Returns:
            torchvision.transforms.Compose: Augmentation pipeline
        """
        if modality == 'xray':
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(256),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:  # RGB modalities
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    @staticmethod
    def get_val_transforms(modality='xray', img_size=224):
        """
        Get validation transforms (no augmentation)
        
        Args:
            modality (str): 'xray', 'ct', 'mri', or 'ultrasound'
            img_size (int): Target image size
            
        Returns:
            torchvision.transforms.Compose: Validation pipeline
        """
        if modality == 'xray':
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:  # RGB modalities
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


def denormalize_image(tensor, modality='xray'):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
        modality (str): 'xray' or 'rgb'
        
    Returns:
        np.ndarray: Denormalized image (0-255)
    """
    if modality == 'xray':
        mean, std = [0.5], [0.5]
        tensor = tensor * std[0] + mean[0]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    
    # Clamp to [0, 1] and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    img_array = tensor.cpu().numpy()
    
    # Transpose if needed (C, H, W) -> (H, W, C)
    if img_array.shape[0] in [1, 3]:
        img_array = np.transpose(img_array, (1, 2, 0))
    
    # Convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    
    return img_array


if __name__ == "__main__":
    print("=" * 60)
    print("Medical Image Preprocessing Test")
    print("=" * 60)
    
    # Test X-ray preprocessor
    print("\n1. Testing X-Ray Preprocessor:")
    xray_prep = XRayPreprocessor(img_size=224)
    dummy_xray = Image.new('L', (512, 512), color=128)
    xray_tensor = xray_prep(dummy_xray)
    print(f"   Input: 512x512 grayscale")
    print(f"   Output: {xray_tensor.shape} (1 channel)")
    print(f"   ✅ X-Ray preprocessing successful!")
    
    # Test RGB preprocessor
    print("\n2. Testing RGB Preprocessor (CT/MRI/Ultrasound):")
    rgb_prep = RGBPreprocessor(img_size=224)
    dummy_rgb = Image.new('RGB', (512, 512), color=(128, 128, 128))
    rgb_tensor = rgb_prep(dummy_rgb)
    print(f"   Input: 512x512 RGB")
    print(f"   Output: {rgb_tensor.shape} (3 channels)")
    print(f"   ✅ RGB preprocessing successful!")
    
    # Test augmentation
    print("\n3. Testing Data Augmentation:")
    train_aug = DataAugmentation.get_train_transforms('xray', 224)
    aug_tensor = train_aug(dummy_xray)
    print(f"   Augmented shape: {aug_tensor.shape}")
    print(f"   ✅ Augmentation pipeline successful!")
    
    print("\n" + "=" * 60)
    print("✅ All preprocessing tests passed!")
    print("=" * 60)
