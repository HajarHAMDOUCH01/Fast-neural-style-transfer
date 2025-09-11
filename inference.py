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
    vgg = VGG19().to(device)
    vgg.eval()

    vgg_weights = vgg.vgg_model_weights.IMAGENET1K_V1
    transform = vgg_weights.transforms()

    style_net = StyleTransferNet().to(device)
    style_net.load_state_dict(model_path)
    style_net.eval()
    
    image = Image.open("/content/dancing (1).jpg").convert("RGB")
    content_image = transform(image).unsqueeze(0).to(device)
    # print("image tensor shape : ", content_image[0].shape)
    # print("image tensor to check values : ", content_image)
    with torch.no_grad():
        stylized_tensor = style_net(content_image)
        # print("output image tensor to check values : ", sample_image)
    stylized_tensor = denormalize_batch(stylized_tensor)
    stylized_tensor = torch.clamp(stylized_tensor * 255, 0, 255)

    stylized_img = transforms.ToPILImage()(stylized_tensor[0].device())
    stylized_img.save(f"{output_path}")
    print(f"Stylized image saved to {output_path}")
