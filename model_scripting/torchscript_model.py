import torch
import torch.nn.functional as F

import sys

sys.path.append('../')
from models.model import StyleTransferNet

style_transfer_net = StyleTransferNet()
style_transfer_net.eval()

scripted_model = torch.jit.script(style_transfer_net)

dummy_input = torch.rand(1,3,256,256)

output_model = style_transfer_net(dummy_input)
output_scripted_model = scripted_model(dummy_input)

diff = torch.abs(output_model - output_scripted_model)
print(diff)