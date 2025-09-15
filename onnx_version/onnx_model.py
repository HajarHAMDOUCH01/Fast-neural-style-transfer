import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import StyleTransferNet

import sys
sys.path.append('/content/real-time-neural-style-transfer')

model = StyleTransferNet()

# Load checkpoint
checkpoint = torch.load("path_to_your_weights.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    dummy_input = torch.randn(1, 3, 256, 256)
    
    test_output = model(dummy_input)
    print(f"Model output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    torch.onnx.export(
        model,
        dummy_input,
        "/content/nst_onnx_model.onnx",
        export_params=True,
        opset_version=11,  
        input_names=["input_image"],
        output_names=["output_image"],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'output_image': {0: 'batch_size'}
        },
        do_constant_folding=True,  # Enable constant folding optimization
        verbose=False
    )

print("ONNX model saved at /content/nst_onnx_model.onnx")