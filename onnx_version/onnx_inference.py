import onnx 
import torchvision.transforms as transforms
from PIL import Image
import torch
import onnxruntime
import numpy as np

onnx_model = onnx.load("/content/nst_onnx_model.onnx")
onnx.checker.check_model(onnx_model)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
])

def from_img_to_tensor(image_path):
    image = Image.open(image_path).to("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def fom_tensor_to_image(image_tensor):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    image_tensor = torch.clamp(image_tensor, 0, 1)
    image = transforms.ToPILImage()(image_tensor.cpu())
    return image

def onnx_inference(image_path, output_dir, onnx_model_path):
    input_tensor = from_img_to_tensor(image_path)
    onnx_input = input_tensor.detach().cpu().numpy()
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )
    onnxruntime_input = {ort_session.get_inputs()[0].name: onnx_input[0]}
    onnxruntime_output = ort_session.run(None, onnxruntime_input)[0]
    output_image = fom_tensor_to_image(onnxruntime_output)
    output_image.save(f"{output_dir}/ouput_image.jpg")

if __name__=="__main__":
  onnx_inference("/content/dancing.jpg", "/content", "/content/nst_onnx_model.onnx")