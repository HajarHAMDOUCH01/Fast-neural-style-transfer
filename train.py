import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from PIL import Image
import torch.nn.functional as F
import os
import time
import numpy as np

import sys
sys.path.append('/content/real-time-neural-style-transfer')
from model import StyleTransferNet, VGG16, gram_matrix, content_loss, style_loss, total_variation_loss


BATCH_SIZE      = 4
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 2

CONTENT_WEIGHT  = 1.0
STYLE_WEIGHT    = 1e3     
TV_WEIGHT       = 1e-5      


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_style_image(style_path, size=256):
    """Load and preprocess style image"""
    style_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    
    style_img = Image.open(style_path).convert('RGB')
    style_img = style_transform(style_img).unsqueeze(0)
    return style_img.to(device)

def get_style_targets(vgg, style_img):
    """Fixed style target computation"""
    with torch.no_grad():
        style_features = vgg(style_img)
        style_targets = []
        
        for feat in style_features:
            gram = gram_matrix(feat)
            style_targets.append(gram.squeeze(0))  # Remove batch dimension for storage
    return style_targets

class COCODataset(torch.utils.data.Dataset):
    """Custom COCO dataset for content images"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Get all image files
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
            # Return a random index if current image fails
            return self.__getitem__(np.random.randint(0, len(self.images)))

def train_style_transfer():
    """Final fixed training function"""
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor()
    ])
    
    # Load dataset
    dataset = COCODataset(root='/kaggle/input/imagenet/imagenet/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=2, pin_memory=True)
    
    # Initialize networks
    style_net = StyleTransferNet().to(device)
    vgg = VGG16().to(device)
    vgg.eval()
    
    # Load style image and compute targets
    style_img = load_style_image('/content/picasso.jpg')
    style_targets = get_style_targets(vgg, style_img)
    
    # Print style target shapes for debugging
    print("Style target shapes:")
    for i, target in enumerate(style_targets):
        print(f"Layer {i}: {target.shape}")
    
    steps_per_epoch = len(dataset) // BATCH_SIZE
    total_steps     = min(40000, steps_per_epoch * NUM_EPOCHS)
    
    # Optimizer with better settings
    optimizer = optim.Adam(style_net.parameters(), lr=LEARNING_RATE, 
                          betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.3
    )
    
    # Training loop
    print("Starting training...")
    style_net.train()
    
    total_iterations = 0
    target_iterations = 40000
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_content_loss = 0.0
        epoch_style_loss = 0.0
        epoch_tv_loss = 0.0
        
        for batch_idx, content_batch in enumerate(dataloader):
            if total_iterations >= target_iterations:
                break
                
            content_batch = content_batch.to(device)
            
            # Forward pass
            stylized_batch = style_net(content_batch)
            
            # Extract features
            content_features = vgg(content_batch)
            stylized_features = vgg(stylized_batch)
            
            # Compute losses with proper weighting
            c_loss = content_loss(stylized_features, content_features)
            s_loss = style_loss(stylized_features, style_targets)
            tv_loss = total_variation_loss(stylized_batch)
            
            # Total loss with proper weighting
            total_loss = (CONTENT_WEIGHT * c_loss + 
                         STYLE_WEIGHT * s_loss + 
                         TV_WEIGHT * tv_loss)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(style_net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_content_loss += c_loss.item()
            epoch_style_loss += s_loss.item()
            epoch_tv_loss += tv_loss.item()
            
            total_iterations += 1
            
            # Print progress
            if total_iterations % 50 == 0:
                print(f"Iteration [{total_iterations}/{target_iterations}] "
                      f"Total Loss: {total_loss.item():.4f} "
                      f"Content: {c_loss.item():.4f} "
                      f"Style: {s_loss.item():.6f} "
                      f"TV: {tv_loss.item():.6f}")
            
            # Debug gram matrix ranges occasionally
            if total_iterations % 1000 == 0:
                print(f"\n=== Debug at iteration {total_iterations} ===")
                with torch.no_grad():
                    for i, (s_feat, target) in enumerate(zip(stylized_features, style_targets)):
                        s_gram = gram_matrix(s_feat)
                        print(f"Layer {i}: stylized_gram range: {s_gram.min():.8f} - {s_gram.max():.8f}")
                        print(f"Layer {i}: target_gram range: {target.min():.8f} - {target.max():.8f}")
                        mse = F.mse_loss(s_gram, target.unsqueeze(0).expand_as(s_gram))
                        print(f"Layer {i}: MSE = {mse:.8f}")
            
            if total_iterations >= target_iterations:
                break
        
        if total_iterations >= target_iterations:
            break
    
    # Save final model
    torch.save(style_net.state_dict(), 'style_transfer_final.pth')
    print("Training completed!")

def test_inference(model_path, content_path, output_path):
    """Test the trained model on a single image"""
    # Load model
    style_net = StyleTransferNet().to(device)
    style_net.load_state_dict(torch.load(model_path, map_location=device))
    style_net.eval()
    
    # Load content image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    content_img = Image.open(content_path).convert('RGB')
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    
    # Generate stylized image
    with torch.no_grad():
        stylized_tensor = style_net(content_tensor)
        
    # Save result
    stylized_img = transforms.ToPILImage()(stylized_tensor[0].cpu())
    stylized_img.save(output_path)
    print(f"Stylized image saved to {output_path}")

if __name__ == '__main__':
    train_style_transfer()

    # test_inference('style_transfer_final.pth', 'test_content.jpg', 'stylized_output.jpg')