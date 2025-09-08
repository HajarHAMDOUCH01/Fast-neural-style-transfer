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
from models.model import StyleTransferNet, VGG16, gram_matrix, style_loss, total_variation_loss, content_loss

TOTAL_STEPS = 40000
BATCH_SIZE      = 4  
LEARNING_RATE   = 1e-2
NUM_EPOCHS      = 2

CONTENT_WEIGHT = 1.0
STYLE_WEIGHT   = 3
TV_WEIGHT      = 1e-4

TRAIN_IMAGE_SHAPE = (256, 256)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def normalize_batch(batch):
    """Ensure batch is in [0,1] range for VGG"""
    return torch.clamp(batch, 0.0, 1.0)

def denormalize_batch(batch):
    """Convert from [-1,1] to [0,1] range"""
    return (batch + 1.0) / 2.0 


def test_inference(model_path, content_path, output_path):
    style_net = StyleTransferNet().to(device)
    style_net.load_state_dict(torch.load(model_path, map_location=device))
    style_net.eval()
    
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SHAPE),
        transforms.ToTensor(),  
        transforms.Lambda(lambda x: x * 2.0 - 1.0)  
    ])
    
    content_img = Image.open(content_path)
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stylized_tensor = style_net(content_tensor)  
        stylized_tensor = denormalize_batch(stylized_tensor)
        stylized_tensor = torch.clamp(stylized_tensor * 255, 0, 255)
        
    stylized_img = transforms.ToPILImage()(stylized_tensor[0].cpu())
    stylized_img.save(output_path)
    print(f"Stylized image saved to {output_path}")