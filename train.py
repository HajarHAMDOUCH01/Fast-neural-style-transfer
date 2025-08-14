import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from PIL import Image
import os
import time
import numpy as np

import sys
sys.path.append('/content/real-time-neural-style-transfer')
from model import StyleTransferNet, VGG16, gram_matrix, content_loss, style_loss, total_variation_loss

# Hyperparameters from Johnson et al. paper
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2  # 2 passes over COCO dataset
STYLE_WEIGHT = 1e6
CONTENT_WEIGHT = 1.0
TV_WEIGHT = 1e-4

# Device configuration
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
    """Precompute style target Gram matrices"""
    with torch.no_grad():
        style_features = vgg(style_img)
        style_targets = []
        for feat in style_features:
            gram = gram_matrix(feat)
            # Remove batch dimension since style image is single image
            style_targets.append(gram.squeeze(0))
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
    """Main training function"""
    
    # Data transforms following Johnson et al.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor()
    ])
    
    # Load dataset - Replace with your COCO path
    dataset = COCODataset(root='/root/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/train2017', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=2, pin_memory=True)  # Reduced workers for CPU
    
    # Initialize networks
    style_net = StyleTransferNet().to(device)
    vgg = VGG16().to(device)
    vgg.eval()
    
    # Load style image and compute targets
    style_img = load_style_image('/content/picasso.jpg')
    style_targets = get_style_targets(vgg, style_img)
    
    # Optimizer
    optimizer = optim.Adam(style_net.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    style_net.train()
    
    total_iterations = 0
    target_iterations = 40000  # Approximately 2 epochs on 80k images with batch_size=4
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_content_loss = 0.0
        epoch_style_loss = 0.0
        epoch_tv_loss = 0.0
        
        for batch_idx, content_batch in enumerate(dataloader):
            if total_iterations >= target_iterations:
                break
                
            content_batch = content_batch.to(device)
            
            # Forward pass through style network
            stylized_batch = style_net(content_batch)
            
            # Extract features for losses
            content_features = vgg(content_batch)
            stylized_features = vgg(stylized_batch)
            
            # Compute content loss (relu2_2 only)
            c_loss = content_loss(stylized_features[1], content_features[1])
            
            # Compute style losses (all layers)
            s_loss = 0.0
            for stylized_feat, target_gram in zip(stylized_features, style_targets):
                s_loss += style_loss(stylized_feat, target_gram)
            
            # Compute total variation loss
            tv_loss = total_variation_loss(stylized_batch)
            
            # Total loss
            total_loss = (CONTENT_WEIGHT * c_loss + 
                         STYLE_WEIGHT * s_loss + 
                         TV_WEIGHT * tv_loss)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
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
            
            # Save checkpoint every 1000 iterations
            if total_iterations % 1000 == 0:
                checkpoint = {
                    'iteration': total_iterations,
                    'model_state_dict': style_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.item(),
                }
                torch.save(checkpoint, f'checkpoint_iter_{total_iterations}.pth')
                
                # Save sample stylized image
                with torch.no_grad():
                    sample_stylized = style_net(content_batch[0:1])
                    save_image = transforms.ToPILImage()(sample_stylized[0].cpu())
                    save_image.save(f'sample_iter_{total_iterations}.jpg')
        
        avg_loss = epoch_loss / len(dataloader)
        avg_content = epoch_content_loss / len(dataloader)
        avg_style = epoch_style_loss / len(dataloader)
        avg_tv = epoch_tv_loss / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Avg Loss: {avg_loss:.4f} "
              f"Content: {avg_content:.4f} "
              f"Style: {avg_style:.6f} "
              f"TV: {avg_tv:.6f}")
        
        if total_iterations >= target_iterations:
            break
    
    # Save final model
    torch.save(style_net.state_dict(), 'style_transfer_final.pth')
    print("Training completed! Model saved as 'style_transfer_final.pth'")

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
    # Train the model
    train_style_transfer()
   
    # test_inference('style_transfer_final.pth', 'test_content.jpg', 'stylized_output.jpg')