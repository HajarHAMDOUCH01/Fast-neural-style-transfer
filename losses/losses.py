import torch
import torch.nn as nn
import torch.nn.functional as F
from config import vgg_loss_layers

def gram_matrix(input_feat):
    """Compute Gram matrix for style representation"""
    b, c, h, w = input_feat.size()
    features = input_feat.view(b, c, h * w)
    
    gram = torch.bmm(features, features.transpose(1, 2))
    
    return gram.div(c*h*w)

def style_loss(input_features, target_grams):
    """Calculate style loss using Gram matrices"""
    # Indices of style layers from VGG19
    style_indices = [0, 1, 2, 3, 5]  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
    
    layers_weights = [0.2,0.2,0.2,0.4,0.2]  
    
    total_loss = 0.0
    
    for idx, weight in zip(style_indices, layers_weights):
        input_feat = input_features[idx]
        target_gram = target_grams[idx]
        
        # Calculate Gram matrix for current layer
        gram = gram_matrix(input_feat)
        print("target gram : ", target_gram)
        c = target_gram.shape[0]
        
        # Ensure target_gram has batch dimension
        if target_gram.dim() == 2:
            target_gram = target_gram.unsqueeze(0)
        
        # Expand target to match batch size
        if gram.size(0) != target_gram.size(0):
            target_gram = target_gram.expand_as(gram)
        
        # Calculate MSE loss between Gram matrices
        layer_loss = F.mse_loss(gram, target_gram, reduction='sum')
        total_loss += (weight * layer_loss) / c*c
    
    return total_loss

def content_loss(input_features, target_features):
    """Calculate content loss using relu4_2 layer"""
    # Use relu4_2 (index 4) as content layer
    content_layer_idx = 4
    
    input_content = input_features[content_layer_idx]
    target_content = target_features[content_layer_idx]
    
    loss = F.mse_loss(input_content, target_content, reduction='sum')
    
    # Normalize by feature map size
    b, c, h, w = input_content.size()
    loss = loss / (c * h * w)
    
    return loss

def total_variation_loss(img):
    """Calculate total variation loss to reduce noise"""
    batch_size, channels, height, width = img.size()
    
    # Calculate differences between adjacent pixels
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    
    # Normalize by image size
    tv_loss = (tv_h + tv_w) / (batch_size * channels * height * width)
    
    return tv_loss