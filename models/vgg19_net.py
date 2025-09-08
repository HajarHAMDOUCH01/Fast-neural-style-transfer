import torch
import torch.nn as nn
import torch.nn.functional as F


VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

class VGG19(nn.Module):
    """VGG16 features for perceptual losses"""
    def __init__(self):
        super(VGG19, self).__init__()

        from torchvision.models import vgg16, VGG16_Weights
        vgg_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Extraction of specific layers for content and style losses
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.slice6 = nn.Sequential() 
        
        # relu1_1 (conv1_1 + relu1_1)
        for x in range(2):
            self.slice1.add_module(str(x), vgg_features[x])
        # relu2_1 (pool1 + conv2_1 + relu2_1)  
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_features[x])
        # relu3_1 (pool2 + conv3_1 + relu3_1)
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_features[x])
        # relu4_1 (pool3 + conv4_1 + relu4_1)
        for x in range(12, 19):
            self.slice4.add_module(str(x), vgg_features[x])
        # relu4_2 (conv4_2 + relu4_2) - for content loss
        for x in range(19, 21):
            self.slice5.add_module(str(x), vgg_features[x])
        # relu5_1 (pool4 + conv5_1 + relu5_1)
        for x in range(21, 26):
            self.slice6.add_module(str(x), vgg_features[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Normalize input for VGG (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        h_relu1_1 = self.slice1(x)
        h_relu2_1 = self.slice2(h_relu1_1)
        h_relu3_1 = self.slice3(h_relu2_1)
        h_relu4_1 = self.slice4(h_relu3_1)
        h_relu4_2 = self.slice5(h_relu4_1)  # Content layer
        h_relu5_1 = self.slice6(h_relu4_2)  # Style layer
        
        return [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu4_2, h_relu5_1]