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
    style_net = StyleTransferNet().to(device)
    style_net.load_state_dict(torch.load(model_path, map_location=device))
    style_net.eval()
    
    transform = transforms.Compose([
        transforms.Resize(training_config['TRAIN_IMAGE_SHAPE']),
        transforms.ToTensor(), # -> [0,1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0) # -> [-1,1]
    ])
    
    content_img = Image.open(content_path)
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stylized_tensor = style_net(content_tensor)  
        stylized_tensor = denormalize_batch(stylized_tensor)
        stylized_tensor = torch.clamp(stylized_tensor * 255, 0, 255)
        
    stylized_img = transforms.ToPILImage()(stylized_tensor[0].device())
    stylized_img.save(output_path)
    print(f"Stylized image saved to {output_path}")

if __name__ == "__main__":
    test_inference("model", "content_img", "output")