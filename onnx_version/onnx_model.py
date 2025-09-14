import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import StyleTransferNet

model = StyleTransferNet().eval()
dummy_input = (torch.randn(1,3,256,256))
onnx_program = torch.onnx.export(model, dummy_input, dynamo=True)

onnx_program.save("nst_model.onnx")