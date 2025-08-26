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
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel=3)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        y = F.relu(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        return x + y
    

class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, scale=2):
        super().__init__()
        pad = kernel // 2
        self.conv_transpose = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=kernel, stride=scale, 
            padding=pad, output_padding=scale-1
        )

    def forward(self, x):
        return self.conv_transpose(x)

class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, kernel=9, stride=1)
        self.norm1 = nn.InstanceNorm2d(32, affine = True)

        self.conv2 = ConvLayer(32, 64, kernel=3, stride=2)
        self.norm2 = nn.InstanceNorm2d(64, affine = True)

        self.conv3 = ConvLayer(64, 128, kernel=3, stride=2)
        self.norm3 = nn.InstanceNorm2d(128, affine = True)

        self.conv3 = ConvLayer(128, 256, kernel=3, stride=2)
        self.norm3 = nn.InstanceNorm2d(128, affine = True)

        self.conv3 = ConvLayer(256, 512, kernel=3, stride=2)
        self.norm3 = nn.InstanceNorm2d(128, affine = True)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(512) for _ in range(5)  
        ])

        self.up1 = UpsampleConv(512, 256, kernel=3, scale=2)
        self.norm4 = nn.InstanceNorm2d(64, affine = True)

        self.up2 = UpsampleConv(256, 128, kernel=3, scale=2)
        self.norm5 = nn.InstanceNorm2d(128, affine = True)

        self.up2 = UpsampleConv(128, 64, kernel=3, scale=2)
        self.norm5 = nn.InstanceNorm2d(256, affine = True)

        self.up2 = UpsampleConv(64, 32, kernel=3, scale=2)
        self.norm5 = nn.InstanceNorm2d(256, affine = True)

        self.up2 = UpsampleConv(32, 3, kernel=3, scale=2)
        self.norm5 = nn.InstanceNorm2d(256, affine = True)

        self.final_conv = ConvLayer(3, 3, kernel=9, stride=1)

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.norm1(self.conv1(x)))
        enc2 = F.relu(self.norm2(self.conv2(enc1)))
        enc3 = F.relu(self.norm3(self.conv3(enc2)))

        # Residual blocks
        res = enc3
        for res_block in self.res_blocks:
            res = res_block(res)

        # Decoder 
        dec1 = F.relu(self.norm4(self.up1(res))) 
        dec2 = F.relu(self.norm5(self.up2(dec1))) 

        # Final output
        output = self.final_conv(dec2)
        return torch.tanh(output) 
    

VGG16_layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3','relu4_3', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'
)

class VGG16(nn.Module):
    """VGG16 features for perceptual losses"""
    def __init__(self):
        super(VGG16, self).__init__()

        from torchvision.models import vgg16, VGG16_Weights
        vgg_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Extraction of specific layers for content and style losses
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        # relu1_1
        for x in range(1):
            self.slice1.add_module(str(x), vgg_features[x])
        # relu2_1
        for x in range(1,6):
            self.slice2.add_module(str(x), vgg_features[x])
        # relu3_1
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg_features[x])
        # relu4_1
        for x in range(11, 18):
            self.slice4.add_module(str(x), vgg_features[x])
        # relu5_1
        for x in range(18, 25):
            self.slice4.add_module(str(x), vgg_features[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):

        mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        h_relu1_1 = self.slice1(x)
        h_relu2_1 = self.slice2(h_relu1_1)
        h_relu3_1 = self.slice3(h_relu2_1)
        h_relu4_1 = self.slice4(h_relu3_1)
        h_relu5_1 = self.slice4(h_relu4_1)
        
        return [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]

class PerceptualLoss(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.vgg = vgg
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, input_features, target_features):
        loss = 0
        for i, (inp_feat, tgt_feat, weight) in enumerate(zip(input_features, target_features, self.weights)):
            loss += weight * F.l1_loss(inp_feat, tgt_feat)
        return loss

def gram_matrix(input_feat):
    b, c, h, w = input_feat.size()
    n = input_feat.numel() // c
    features = input_feat.view(b, c, h * w)
    
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = gram.div(c * n) 

    return gram # supposed to return a gram matrix with batch dim ! -> check

#loss functions
def style_loss(input_features, target_grams):
    style_weights = [1,1,1,1]

    total_loss = 0.0
    
    for input_feat, target_gram, weight in zip(input_features, target_grams, style_weights):
        gram = gram_matrix(input_feat)
        
        if target_gram.dim() == 2:
            target_gram = target_gram.unsqueeze(0)
        target_gram = target_gram.expand_as(gram)
        
        total_loss += weight * F.mse_loss(gram, target_gram)
    
    return total_loss


# def content_loss(input_features, target_features):
#     return F.mse_loss(input_features[3], target_features[3]) 

def total_variation_loss(img):
    batch_size, channels, height, width = img.size()
    
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    
    return (tv_h + tv_w) / (batch_size * channels * height * width)