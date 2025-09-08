import torch
import torch.nn as nn
import torch.nn.functional as F
from config import vgg_loss_layers

def gram_matrix(input_feat):
    b, c, h, w = input_feat.size()
    features = input_feat.view(b, c, h * w)
    
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = gram.div(c*h*w) 

    return gram 

#loss functions
def style_loss(input_features, target_grams):
    style_indices = [0, 1, 2, 3, 5]  
    
    total_loss = 0.0
    
    for idx in style_indices:
        input_feat = input_features[idx]
        target_gram = target_grams[idx]
        
        gram = gram_matrix(input_feat)
        
        # adding batch dimention to style targets gram matrix 
        if target_gram.dim() == 2:
            target_gram = target_gram.unsqueeze(0)

        if gram.size(0) != target_gram.size(0):
            target_gram = target_gram.expand_as(gram)
        
        loss = F.mse_loss(gram, target_gram) 
        # gets the width and height of loss
        loss_w_h = loss.size[:2]
        # Frobenius Normalization
        normalized_loss = torch.linalg.norm(loss, ord='for', dim=loss_w_h) 
        total_loss +=  normalized_loss # square Frobenius Normalization
    
    return total_loss

def content_loss(input_features, target_features):
    # relu4_2 (index 4) as content layer
    loss = F.mse_loss(input_features[4], target_features[4]) 
    # loss should be normalized by the c*h*w of the size of the content layer of the vgg19 
    content_layer_size = input_features.size.to(torch.float32)
    normalized_loss = loss / content_layer_size
    return normalized_loss

def total_variation_loss(img):
    batch_size, channels, height, width = img.size()
    
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    
    return (tv_h + tv_w) / (batch_size * channels * height * width)