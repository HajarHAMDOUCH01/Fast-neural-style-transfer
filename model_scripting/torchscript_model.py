import torch
import torch.nn.functional as F

import sys

sys.path.append('../')
from models.model import StyleTransferNet

checkpoint = torch.load("./checkpoint_20000 (1).pth", map_location="cpu")

style_transfer_net = StyleTransferNet()
style_transfer_net.load_state_dict(checkpoint['model_state_dict'])
style_transfer_net.eval()

# scripted_model = torch.jit.script(style_transfer_net)
# scripted_model.save("style_transfer_model.pt")

# try:
#     scripted_model = torch.jit.script(style_transfer_net)
#     print("Scripting successful")
# except Exception as e:
#     print(f"Scripting failed: {e}")
dummy_input = torch.rand(1,3,256,256)

traced_model = torch.jit.trace(style_transfer_net, dummy_input, strict=False)
traced_model.save("model_traced.pt")



# output_model = style_transfer_net(dummy_input)
# output_scripted_model = scripted_model(dummy_input)

# diff = torch.abs(output_model - output_scripted_model)
# print(diff)