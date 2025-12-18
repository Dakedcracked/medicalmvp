"""
Custom DenseNet121 Architecture for Medical Chest X-Ray Analysis
Replicates TorchXRayVision model architecture without external dependencies

Based on: Densely Connected Convolutional Networks (Huang et al., 2017)
Adapted for medical imaging with 18-class multi-label classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    """Single dense layer within a dense block"""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        
        # Bottleneck layer (1x1 conv)
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        
        # Standard convolution (3x3 conv)
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """Dense block consisting of multiple dense layers"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    """Transition layer between dense blocks"""
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet121(nn.Module):
    """
    DenseNet-121 Architecture for Medical Chest X-Ray Classification
    
    Configuration:
    - Growth rate (k): 32
    - Block configuration: [6, 12, 24, 16] layers
    - Compression factor: 0.5
    - Input: 224x224 grayscale images (1 channel)
    - Output: 18 pathology predictions (multi-label)
    
    Architecture:
    1. Initial Conv: 7x7, stride 2
    2. DenseBlock1: 6 layers
    3. Transition1
    4. DenseBlock2: 12 layers
    5. Transition2
    6. DenseBlock3: 24 layers
    7. Transition3
    8. DenseBlock4: 16 layers
    9. Global Average Pooling
    10. Classifier: Linear(1024 → 18)
    """
    
    def __init__(self, num_classes=18, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, in_channels=1):
        """
        Args:
            num_classes (int): Number of output classes (18 for chest X-ray)
            growth_rate (int): How many filters to add each layer (k in paper)
            block_config (tuple): How many layers in each pooling block
            num_init_features (int): Number of filters in first convolution
            bn_size (int): Multiplicative factor for bottleneck layers
            drop_rate (float): Dropout rate
            in_channels (int): Number of input channels (1 for grayscale X-ray)
        """
        super(DenseNet121, self).__init__()
        
        # First convolution (7x7 conv, stride 2)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                               bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            # Add transition layer (except after last block)
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, 
                                   num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Linear classifier
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Store pathology names
        self.pathologies = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
            'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'
        ]
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (Tensor): Input tensor [batch_size, 1, 224, 224]
            
        Returns:
            Tensor: Output logits [batch_size, 18]
        """
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def get_features(self, x):
        """Extract feature maps for visualization (Grad-CAM)"""
        return self.features(x)


def densenet121(num_classes=18, pretrained=False, in_channels=1):
    """
    Constructs a DenseNet-121 model for medical chest X-ray analysis
    
    Args:
        num_classes (int): Number of output classes (default: 18)
        pretrained (bool): If True, returns a model pre-trained on medical data
        in_channels (int): Number of input channels (default: 1 for grayscale)
        
    Returns:
        DenseNet121: Initialized model
    """
    model = DenseNet121(
        num_classes=num_classes,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        in_channels=in_channels
    )
    
    if pretrained:
        # Load pretrained weights if available
        # For now, returns model with random initialization
        # To load weights: model.load_state_dict(torch.load('weights.pth'))
        print("⚠ Pretrained weights not implemented. Using random initialization.")
        
    return model


# Model statistics
def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("=" * 60)
    print("DenseNet121 Medical Chest X-Ray Model")
    print("=" * 60)
    
    model = densenet121(num_classes=18, in_channels=1)
    print(f"✅ Model created successfully!")
    print(f"📊 Total parameters: {count_parameters(model):,}")
    print(f"🏥 Pathologies detected: {len(model.pathologies)}")
    print(f"📋 Classes: {model.pathologies}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)
    print(f"\n🧪 Test Forward Pass:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ Forward pass successful!")
    
    # Model summary
    print(f"\n📐 Model Summary:")
    print(f"   - Architecture: DenseNet-121")
    print(f"   - Growth rate: 32")
    print(f"   - Block config: (6, 12, 24, 16)")
    print(f"   - Input: 224x224 grayscale (1 channel)")
    print(f"   - Output: {len(model.pathologies)} classes (multi-label)")
    print(f"   - Parameters: {count_parameters(model):,}")
    print("=" * 60)
