import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, scale=2, method='transpose'):
        super().__init__()
        self.scale = scale
        self.method = method
        
        if method == 'transpose':
            # Method 1: Transpose Convolution (ConvTranspose2d)
            # More learnable parameters, can produce sharper results
            pad = kernel // 2
            self.upsample_conv = nn.ConvTranspose2d(
                in_ch, out_ch, 
                kernel_size=kernel, 
                stride=scale, 
                padding=pad,
                output_padding=scale-1 
            )
            
        elif method == 'bilinear':
            # Method 2: Bilinear Upsampling + Convolution
            # Smoother upsampling, less checkerboard artifacts
            pad = kernel // 2
            self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=pad)
            
        elif method == 'pixelshuffle':
            # Method 3: Sub-pixel Convolution (Pixel Shuffle)
            # Often produces the sharpest results
            self.conv = nn.Conv2d(in_ch, out_ch * (scale ** 2), kernel_size=kernel, stride=1, padding=kernel//2)
            self.pixel_shuffle = nn.PixelShuffle(scale)
            
        elif method == 'nearest_reflect':
            # Method 4: Nearest + Reflection Padding 
            self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
            self.reflection_pad = nn.ReflectionPad2d(kernel // 2)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=1)
        
    def forward(self, x):
        if self.method == 'transpose':
            return self.upsample_conv(x)
            
        elif self.method == 'bilinear':
            x = self.upsample(x)
            return self.conv(x)
            
        elif self.method == 'pixelshuffle':
            x = self.conv(x)
            return self.pixel_shuffle(x)
            
        elif self.method == 'nearest_reflect':
            x = self.upsample(x)
            x = self.reflection_pad(x)
            return self.conv(x)
            
        else:  # nearest
            x = self.upsample(x)
            return self.conv(x)

class StyleTransferNet(nn.Module):
    def __init__(self, upsample_method='transpose'):
        super().__init__()
        # Encoder layers
        self.conv1 = ConvLayer(3, 64, kernel=9, stride=2)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = ConvLayer(64, 256, kernel=3, stride=2)  
        self.norm2 = nn.InstanceNorm2d(256, affine=True)     

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(5)  
        ])

        # Decoder with improved upsampling
        self.up1 = UpsampleConv(256, 64, kernel=3, scale=2, method=upsample_method)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)       

        self.up2 = UpsampleConv(64, 32, kernel=3, scale=2, method=upsample_method)
        self.norm4 = nn.InstanceNorm2d(32, affine=True)     

        # Final output layer
        self.final_conv = ConvLayer(32, 3, kernel=9, stride=1)

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.norm1(self.conv1(x)))
        enc2 = F.relu(self.norm2(self.conv2(enc1)))

        # Residual blocks
        res = enc2
        for res_block in self.res_blocks:
            res = res_block(res)

        # Decoder 
        dec1 = F.relu(self.norm3(self.up1(res))) 
        dec2 = F.relu(self.norm4(self.up2(dec1))) 

        # Final output 
        output = self.final_conv(dec2)
        return output

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
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        y = F.relu(self.in1(self.conv1(x)))
        y = self.dropout(y)
        y = self.in2(self.conv2(y))
        return x + y