import torch
import torch.nn as nn
import torch.nn.functional as F

STYLE_WEIGHT = 1e7       
CONTENT_WEIGHT = 1  

class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return F.relu(out + residual)

class StyleTransferNet(nn.Module):
    """Feed-forward style transfer network"""
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        
        # Encoder - Downsampling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.in1 = nn.InstanceNorm2d(32, affine=True, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(64, affine=True, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(128, affine=True, track_running_stats=False)
        
        # Residual blocks - Main processing
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Decoder - Upsampling layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True, track_running_stats=False)
        
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True, track_running_stats=False)
        
        # Final output layer
        self.deconv3 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
      
    def forward(self, x):
        # Encoder
        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        
        # Decoder
        x = F.relu(self.in4(self.deconv1(x)))
        x = F.relu(self.in5(self.deconv2(x)))
        
        # Final output with scaled tanh
        x = self.deconv3(x)
        # Scale tanh to [0, 1]
        x = (torch.tanh(x) + 1) / 2
        
        return x

class VGG16(nn.Module):
    """VGG16 features for perceptual losses"""
    def __init__(self):
        super(VGG16, self).__init__()
        # Load pretrained VGG16 with updated syntax
        from torchvision.models import vgg16, VGG16_Weights
        vgg_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Extract specific layers for content and style losses
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        # relu1_2
        for x in range(4):
            self.slice1.add_module(str(x), vgg_features[x])
        # relu2_2  
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_features[x])
        # relu3_3
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_features[x])
        # relu4_3
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_features[x])
            
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Normalize input for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(x.device)
        x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

def gram_matrix(features):
    """Compute Gram matrix for style loss"""
    b, c, h, w = features.size()
    # Reshape features to (batch * channels, height * width)
    features = features.view(b, c, h * w)
    # Compute gram matrix: (batch, channels, channels)
    G = torch.bmm(features, features.transpose(1, 2))
    # Normalize by number of elements
    return G.div(c * h * w)

# Loss functions
def style_loss(input_features, target_gram):
    """Enhanced style loss with feature normalization"""
    input_gram = gram_matrix(input_features)
    # Normalize by feature dimension
    norm_factor = input_features.shape[1] * input_features.shape[2] * input_features.shape[3]
    return F.mse_loss(input_gram/norm_factor, target_gram/norm_factor) * STYLE_WEIGHT

def content_loss(input_features, target_features):
    """Stabilized content loss"""
    return F.l1_loss(input_features[2], target_features[2]) * CONTENT_WEIGHT

def total_variation_loss(img):
    """Total variation regularization"""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)