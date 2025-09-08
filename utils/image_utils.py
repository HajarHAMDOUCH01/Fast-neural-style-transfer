import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import os
import numpy as np
from config import training_config, loss_weights_config

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def normalize_batch(batch):
    """Ensure batch is in [0,1] range for VGG"""
    return torch.clamp(batch, 0.0, 1.0)

def denormalize_batch(batch):
    """Convert from [-1,1] to [0,1] range"""
    return (batch + 1.0) / 2.0  