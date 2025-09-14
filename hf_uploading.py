import torch 
import os 

import sys 

sys.path.append("/content/real-time-neural-style-transfer")

from models.model import StyleTransferNet

checkpoint = torch.load("/content/checkpoint_20000.pth", map_location="cuda")

style_transfer_net = StyleTransferNet().to("cuda")

model_file = checkpoint['model_state_dict']

torch.save(model_file, "/content/pytorch_model.pth")
torch.rename(model_file, "pytorch_model.bin")


