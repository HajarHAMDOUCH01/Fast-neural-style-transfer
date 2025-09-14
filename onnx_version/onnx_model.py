import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import StyleTransferNet

import sys

sys.path.append('/content/real-time-neural-style-transfer')

model = StyleTransferNet()

model_state_dict = torch.load("", map_location="cpu")
model.load_state_dict(model_state_dict)
model.eval()

dummy_input = (torch.randn(1,3,256,256))
torch.onnx.export(model, 
                dummy_input,
                "/content/nst_onnx_model.onnx",
                export_params=True,
                opset_version=17,
                input_names=["input_image"],
                output_names=["output_image"],
                dynamo=True)

print("ONNX model saved at /content/nst_model.onnx")