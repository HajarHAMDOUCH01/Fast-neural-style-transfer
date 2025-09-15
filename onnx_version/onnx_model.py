import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import StyleTransferNet

import sys

sys.path.append('/content/real-time-neural-style-transfer')

model = StyleTransferNet()

checkpoint = torch.load("", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = (torch.randn(1,3,256,256))
torch.onnx.export(
    model,
    dummy_input,
    "/content/nst_onnx_model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input_image"],
    output_names=["output_image"],
    dynamic_axes={
        "input_image": {0: "batch", 2: "height", 3: "width"},
        "output_image": {0: "batch", 2: "height", 3: "width"}
    }
)
print("ONNX model saved successfully!")