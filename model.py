import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1):
        super().__init__()
        pad = kernel // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.conv(self.reflection_pad(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel=3)
        self.in1 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
        self.conv2 = ConvLayer(channels, channels, kernel=3)
        self.in2 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)

    def forward(self, x):
        y = F.relu(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        return F.relu(x + y)
    

class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, scale=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        self.conv = ConvLayer(in_ch, out_ch, kernel)

    def forward(self, x):
        return self.conv(self.upsample(x))

class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = ConvLayer(3, 32, kernel=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True, track_running_stats=False)

        self.conv2 = ConvLayer(32, 64, kernel=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True, track_running_stats=False)

        self.conv3 = ConvLayer(64, 128, kernel=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True, track_running_stats=False)

        # Residuals
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Decoder (upsample + conv)
        self.up1 = UpsampleConv(128, 64, kernel=3, scale=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True, track_running_stats=False)

        self.up2 = UpsampleConv(64, 32, kernel=3, scale=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True, track_running_stats=False)

        self.final_conv = ConvLayer(32, 3, kernel=9, stride=1)

    def forward(self, x):
        inp = x
        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))

        x = self.res1(x); x = self.res2(x); x = self.res3(x); x = self.res4(x); x = self.res5(x)

        x = F.relu(self.in4(self.up1(x)))
        x = F.relu(self.in5(self.up2(x)))

        # residual output (better content preservation)
        x = torch.clamp(self.final_conv(x) + inp, 0.0, 1.0)
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
    if features.dim() == 3:
        features = features.unsqueeze(0)
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))  
    return G.div(c* h * w) 

# Fixed loss functions
def style_loss(input_features, target_grams):
    """Fixed style loss - handles batch dimension properly"""
    total_loss = 0
    for input_feat, target_gram in zip(input_features, target_grams):
        input_gram = gram_matrix(input_feat)
        # Expand target gram to match batch size
        batch_size = input_gram.size(0)
        if target_gram.dim() == 2:
            target_gram = target_gram.unsqueeze(0)
        target_gram = target_gram.expand(batch_size, -1, -1)
        total_loss += F.mse_loss(input_gram, target_gram)
    return total_loss 

def content_loss(input_features, target_features):
    """Content loss using relu3_3 (index 2)"""
    return F.mse_loss(input_features[2], target_features[2])  

def total_variation_loss(img):
    """Total variation regularization"""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)