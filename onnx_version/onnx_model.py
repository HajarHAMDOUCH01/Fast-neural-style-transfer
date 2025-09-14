import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import StyleTransferNet

import sys

sys.path.append('/content/real-time-neural-style-transfer')

model = StyleTransferNet()

model_state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(model_state_dict)
model.eval()

dummy_input = (torch.randn(1,3,256,256))
torch.onnx.export(model, dummy_input, dynamo=True)

onnx_program.save("/content/nst_model.onnx")


