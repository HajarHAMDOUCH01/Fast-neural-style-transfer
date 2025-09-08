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
    def __init__(self, in_ch, out_ch, kernel, scale=1/2):
        super().__init__()
        pad = kernel // 2 # kernel - 1 ? 
        self.conv_transpose = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=kernel, stride=scale, 
            padding=pad, output_padding=scale-1
        )

    def forward(self, x):
        return self.conv_transpose(x)

class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 64, kernel=9, stride=2)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = ConvLayer(64, 256, kernel=3, stride=2)  
        self.norm2 = nn.InstanceNorm2d(256, affine=True)     

        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(5)  
        ])

        self.up1 = UpsampleConv(256, 64, kernel=3, scale=1/2)
        self.norm5 = nn.InstanceNorm2d(64, affine=True)       

        self.up3 = UpsampleConv(64, 32, kernel=3, scale=1/2)
        self.norm7 = nn.InstanceNorm2d(32, affine=True)     

        # Final output layer
        self.final_conv = ConvLayer(32, 3, kernel=9, stride=1)

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.norm1(self.conv1(x)))
        enc2 = F.relu(self.norm2(self.conv2(enc1)))
        enc3 = F.relu(self.norm3(self.conv3(enc2)))
        enc4 = F.relu(self.norm4(self.conv4(enc3)))

        # Residual blocks
        res = enc4
        for res_block in self.res_blocks:
            res = res_block(res)

        # Decoder 
        dec1 = F.relu(self.norm5(self.up1(res))) 
        dec2 = F.relu(self.norm6(self.up2(dec1))) 
        dec3 = F.relu(self.norm7(self.up3(dec2)))
        dec4 = F.relu(self.norm8(self.up4(dec3)))

        # Final output
        output = self.final_conv(dec4)
        # print("........", output.shape)
        return torch.tanh(output) 