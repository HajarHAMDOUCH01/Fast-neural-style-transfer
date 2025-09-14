import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import os
import numpy as np

import sys
sys.path.append('/content/real-time-neural-style-transfer')
from models.model import StyleTransferNet
from models.vgg19_net import VGG19
from config import training_config, loss_weights_config, vgg_loss_layers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def normalize_batch(batch):
    """Ensure batch is in [0,1] range for VGG"""
    return torch.clamp(batch, 0.0, 1.0)

def denormalize_batch(batch):
    """Convert from [-1,1] to [0,1] range"""
    return (batch + 1.0) / 2.0 


def test_inference(model_path, content_path, output_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
    ])
    style_net = StyleTransferNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    style_net.load_state_dict(model_state_dict)
    style_net.eval()
    with torch.no_grad():
        # Load and preprocess test image
        test_image = Image.open(content_path).convert("RGB")
        test_tensor = transform(test_image).unsqueeze(0).to(device)
        
        # Generate stylized image
        stylized_tensor = style_net(test_tensor)
        
        # Denormalize and convert to PIL
        # Reverse the normalization
        denorm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        stylized_tensor = denorm(stylized_tensor[0])
        stylized_tensor = torch.clamp(stylized_tensor, 0, 1)
        
        # Convert to PIL and save
        stylized_img = transforms.ToPILImage()(stylized_tensor.cpu())
        stylized_img.save(f"{output_path}/noraml_output.jpg")

