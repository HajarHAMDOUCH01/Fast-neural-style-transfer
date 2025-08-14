import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

import sys
sys.path.append('/content/real-time-neural-style-transfer')
from model import StyleTransferNet, VGG16, gram_matrix, content_loss, style_loss, total_variation_loss

BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
STYLE_WEIGHT = 5e6  # Increased style weight
CONTENT_WEIGHT = 1.0
TV_WEIGHT = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def normalize_batch(batch):
    """Normalize batch to [-1, 1] for training"""
    return batch * 2.0 - 1.0

def denormalize_batch(batch):
    """Denormalize from [-1, 1] to [0, 1]"""
    return (batch + 1.0) / 2.0

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
        # Style image should be in [0, 1] for VGG
        style_features = vgg(style_img)
        style_targets = []
        for feat in style_features:
            gram = gram_matrix(feat)
            # Keep batch dimension for easier handling
            style_targets.append(gram)
    return style_targets

class COCODataset(torch.utils.data.Dataset):
    """Custom COCO dataset for content images"""
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

def train_style_transfer():
    """Main training function with fixes"""
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor()  # Already gives [0, 1]
    ])
    
    # Load dataset
    dataset = COCODataset(root='/root/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/train2017', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=2, pin_memory=True)
    
    # Initialize networks
    style_net = StyleTransferNet().to(device)
    vgg = VGG16().to(device)
    vgg.eval()
    
    style_img = load_style_image('/content/picasso.jpg')
    style_targets = get_style_targets(vgg, style_img)
    
    # Print style target statistics for debugging
    for i, target in enumerate(style_targets):
        print(f"Style target {i} - Shape: {target.shape}, Mean: {target.mean():.6f}, Std: {target.std():.6f}")
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(style_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15000, gamma=0.5)
    
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
            
            normalized_content = normalize_batch(content_batch)
            
            # Forward pass through style network
            stylized_batch_normalized = style_net(normalized_content)
            
            stylized_batch = denormalize_batch(stylized_batch_normalized)
            
            # Clamp to valid range to prevent artifacts
            stylized_batch = torch.clamp(stylized_batch, 0, 1)
            
            # Extract features for losses (both inputs in [0, 1])
            content_features = vgg(content_batch)
            stylized_features = vgg(stylized_batch)
            
            # Compute content loss (relu2_2 only)
            c_loss = content_loss(stylized_features[1], content_features[1])
            
            s_loss = 0.0
            for stylized_feat, target_gram in zip(stylized_features, style_targets):
                # Expand target to match batch size
                target_expanded = target_gram.expand(stylized_feat.size(0), -1, -1)
                s_loss += style_loss(stylized_feat, target_expanded)
            
            # Average style loss across layers
            s_loss = s_loss / len(style_targets)
            
            # Compute total variation loss on normalized output
            tv_loss = total_variation_loss(stylized_batch)
            
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
            
            # Print progress with more details
            if total_iterations % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iteration [{total_iterations}/{target_iterations}] "
                      f"LR: {current_lr:.2e} "
                      f"Total Loss: {total_loss.item():.4f} "
                      f"Content: {c_loss.item():.4f} "
                      f"Style: {s_loss.item():.6f} "
                      f"TV: {tv_loss.item():.6f}")
                
                if torch.isnan(total_loss):
                    print("WARNING: NaN detected in loss!")
                    break
            
            # Save checkpoints more frequently
            if total_iterations % 5000 == 0:
                checkpoint = {
                    'iteration': total_iterations,
                    'model_state_dict': style_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': total_loss.item(),
                }
                torch.save(checkpoint, f'checkpoint_iter_{total_iterations}.pth')
                
                # Save sample stylized image
                with torch.no_grad():
                    style_net.eval()
                    sample_normalized = normalize_batch(content_batch[0:1])
                    sample_stylized = style_net(sample_normalized)
                    sample_stylized = denormalize_batch(sample_stylized)
                    sample_stylized = torch.clamp(sample_stylized, 0, 1)
                    
                    save_image = transforms.ToPILImage()(sample_stylized[0].cpu())
                    save_image.save(f'sample_iter_{total_iterations}.jpg')
                    style_net.train()
                    
                    print(f"Checkpoint saved at iteration {total_iterations}")
        
        if total_iterations >= target_iterations:
            break
    
    # Save final model
    torch.save(style_net.state_dict(), 'style_transfer_final.pth')
    print("Training completed! Model saved as 'style_transfer_final.pth'")

if __name__ == '__main__':
    train_style_transfer()