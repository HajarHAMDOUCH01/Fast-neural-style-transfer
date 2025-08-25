import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from PIL import Image
import torch.nn.functional as F
import os
import numpy as np

import sys
sys.path.append('/content/real-time-neural-style-transfer')
from model import StyleTransferNet, VGG16, gram_matrix, PerceptualLoss, style_loss, total_variation_loss


BATCH_SIZE      = 4  
LEARNING_RATE   = 1e-4
NUM_EPOCHS      = 2

CONTENT_WEIGHT = 1.0
STYLE_WEIGHT   = 1000
TV_WEIGHT      = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_style_image(style_path, size=512):
    style_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()  
    ])
    
    style_img = Image.open(style_path).convert('RGB')
    style_img = style_transform(style_img)
    return style_img.unsqueeze(0).to(device)

def get_style_targets(vgg, style_img):
    with torch.no_grad():
        style_features = vgg(style_img)
        style_targets = []
        
        for feat in style_features:
            gram = gram_matrix(feat)
            style_targets.append(gram.squeeze(0))
    return style_targets

class COCODataset(torch.utils.data.Dataset): 
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        for subdir, dirs, files in os.walk(root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(subdir, file))
        
        print(f"Found {len(self.images)} images in dataset")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.images)))

def train_style_transfer(resume_from_checkpoint=False, checkpoint_path=None):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()  
    ])
    
    dataset = COCODataset(root='/kaggle/input/human-faces/Humans', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=2, pin_memory=True)
    
    style_net = StyleTransferNet().to(device)
    vgg = VGG16().to(device)
    vgg.eval()
    
    style_img = load_style_image('/content/style.jpg')
    
    with torch.no_grad():
        style_targets = get_style_targets(vgg, style_img)
        style_targets = [t.detach() for t in style_targets]

    
    print("Style target shapes:")
    for i, target in enumerate(style_targets):
        print(f"Layer {i}: {target.shape}")

    start_iteration = 0
    total_steps = 40000
    
    optimizer = optim.Adam(style_net.parameters(),
                        lr=LEARNING_RATE,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=1e-5)  

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )
            
    print("Starting training...")
    style_net.train()
    
    total_iterations = start_iteration
    running_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        for batch_idx, content_batch in enumerate(dataloader):
            if total_iterations >= total_steps:
                break
                
            content_batch = content_batch.to(device)
                        
            stylized_batch = style_net(content_batch)
            stylized_batch_scaled = (stylized_batch + 1.0) / 2.0
            
            content_features = vgg(content_batch)

            stylized_features = vgg(stylized_batch_scaled)
            
            c_loss = PerceptualLoss(vgg)(stylized_features, content_features)
            s_loss = style_loss(stylized_features, style_targets)
            tv_loss = total_variation_loss(stylized_batch)
            
            total_loss = (CONTENT_WEIGHT * c_loss + 
                         STYLE_WEIGHT * s_loss + 
                         TV_WEIGHT * tv_loss)
            
            if torch.isnan(total_loss):
                print(f"NaN loss detected at iteration {total_iterations}")
                print(f"Content loss: {c_loss}, Style loss: {s_loss}, TV loss: {tv_loss}")
                continue
            
            optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(style_net.parameters(), max_norm=5.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += total_loss.item()
            total_iterations += 1
            
            if total_iterations <= 1000 and total_iterations % 50 == 0:
                avg_loss = running_loss / 50
                print(f"Iteration [{total_iterations}/{total_steps}] "
                      f"Avg Loss: {avg_loss:.4f} "
                      f"Content: {c_loss.item():.4f} "
                      f"Style: {s_loss.item():.6f} "
                      f"TV: {tv_loss.item():.6f}")
                running_loss = 0.0
            elif total_iterations % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Iteration [{total_iterations}/{total_steps}] "
                      f"Avg Loss: {avg_loss:.4f} "
                      f"Content: {c_loss.item():.4f} "
                      f"Style: {s_loss.item():.6f} "
                      f"TV: {tv_loss.item():.6f}")
                running_loss = 0.0
            
            # checkpoints saving
            if total_iterations % 5000 == 0 and total_iterations > start_iteration:
                torch.save(style_net.state_dict(), 
                          f"/content/drive/MyDrive/style_transfer__{total_iterations}.pth")
                
            if total_iterations >= total_steps:
                break
        
        if total_iterations >= total_steps:
            break
    
    # final model saving
    torch.save(style_net.state_dict(), '/content/drive/MyDrive/style_transfer_final.pth')
    print("Training completed!")

def test_inference(model_path, content_path, output_path):
    """Test the trained model on a single image"""
    # Loading the model
    style_net = StyleTransferNet().to(device)
    style_net.load_state_dict(torch.load(model_path, map_location=device))
    style_net.eval()
    
    # Loading the content image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    content_img = Image.open(content_path)
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    
    # Generating stylized image
    with torch.no_grad():
        stylized_tensor = style_net(content_tensor)
        
    stylized_img = transforms.ToPILImage()(stylized_tensor[0].cpu())
    stylized_img.save(output_path)
    print(f"Stylized image saved to {output_path}")

if __name__ == '__main__':

    # TO DO : a function that takes command parameters and handles this

    # if training from the start : 
    train_style_transfer(resume_from_checkpoint=False)

    # if training from a checkpoint : 
    # train_style_transfer(resume_from_checkpoint=True, checkpoint_path='/content/drive/MyDrive/style_transfer_checkpoint_10000.pth')

    # test_inference('style_transfer_final.pth', 'test_content.jpg', 'stylized_output.jpg')