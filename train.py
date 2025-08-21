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
from model import StyleTransferNet, VGG16, gram_matrix, content_loss, style_loss, total_variation_loss


BATCH_SIZE      = 6
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 2

CONTENT_WEIGHT = 1
STYLE_WEIGHT   = 5
TV_WEIGHT      = 1e-4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_style_image(style_path, size=256):
    style_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    
    style_img = Image.open(style_path).convert('RGB')
    style_img = style_transform(style_img) * 255.0  
    return style_img.unsqueeze(0).to(device)

def get_style_targets(vgg, style_img):
    """Fixed style target computation"""
    with torch.no_grad():
        style_features = vgg(style_img)
        style_targets = []
        
        for feat in style_features:
            gram = gram_matrix(feat)
            style_targets.append(gram.squeeze(0))  
    return style_targets

# not necessarly coco dataset , plus you only need a directory that has images , whatever the dataset is 
# in the case of coco dataset only take the tain directory
class COCODataset(torch.utils.data.Dataset): 
    """Custom dataset class for content images"""
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

def save_checkpoint(model, optimizer, scheduler, iteration, loss, filepath):
    """Saving complete checkpoint including optimizer and scheduler state"""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at iteration {iteration}")

def load_checkpoint(model, optimizer, scheduler, filepath):
    """loading checkpoint and returning the iteration to resume from"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_iteration = checkpoint['iteration']
    print(f"Resuming training from iteration {start_iteration}")
    return start_iteration

def train_style_transfer(resume_from_checkpoint=False, checkpoint_path=None):
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)  
    ])
    
    # Load dataset
    dataset = COCODataset(root='/kaggle/input/coco-2017-dataset/coco2017/train2017', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=2, pin_memory=True)
    
    # Initialize networks
    style_net = StyleTransferNet().to(device)
    vgg = VGG16().to(device)
    vgg.eval()
    
    # Load style image and compute targets
    style_img = load_style_image('/content/style.jpg')
    style_targets = get_style_targets(vgg, style_img)
    
    # Print style target shapes for debugging
    print("Style target shapes:")
    for i, target in enumerate(style_targets):
        print(f"Layer {i}: {target.shape}")

    start_iteration = 0
    total_steps = 40000
    
    optimizer = optim.Adam(style_net.parameters(),
                        lr=LEARNING_RATE,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=1e-4)

    if resume_from_checkpoint == True and checkpoint_path is not None and os.path.exists(checkpoint_path):
        temp_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )
        start_iteration = load_checkpoint(style_net, optimizer, temp_scheduler, checkpoint_path)
        remaining_steps = total_steps - start_iteration
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_steps if remaining_steps > 0 else 1,
            eta_min=1e-6
        )
        print(f"Resuming training from iteration {start_iteration}")
    else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
            print(f"Starting training from scratch")
            
    # Training loop
    print("Starting training...")
    style_net.train()
    
    total_iterations = start_iteration

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_content_loss = 0.0
        epoch_style_loss = 0.0
        epoch_tv_loss = 0.0
        
        for batch_idx, content_batch in enumerate(dataloader):
            if total_iterations >= total_steps:
                break
                
            content_batch = content_batch.to(device)
            
            # Forward pass
            stylized_batch = style_net(content_batch)
            
            # Extract features
            content_features = vgg(content_batch)
            stylized_features = vgg(stylized_batch)
            
            c_loss = content_loss(stylized_features, content_features)
            s_loss = style_loss(stylized_features, style_targets)
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
            
            # Print progress
            if total_iterations % 100 == 0:
                print(f"Iteration [{total_iterations}/{total_steps}] "
                      f"Total Loss: {total_loss.item():.4f} "
                      f"Content: {c_loss.item():.4f} "
                      f"Style: {s_loss.item():.6f} "
                      f"TV: {tv_loss.item():.6f}")
            
            # Debug 
            if total_iterations % 1000 == 0:
                print(f"\n=== Debug at iteration {total_iterations} ===")
                with torch.no_grad():
                    for i, (s_feat, target) in enumerate(zip(stylized_features, style_targets)):
                        s_gram = gram_matrix(s_feat)
                        print(f"Layer {i}: stylized_gram range: {s_gram.min():.8f} - {s_gram.max():.8f}")
                        print(f"Layer {i}: target_gram range: {target.min():.8f} - {target.max():.8f}")
                        mse = F.mse_loss(s_gram, target.unsqueeze(0).expand_as(s_gram))
                        print(f"Layer {i}: MSE = {mse:.8f}")
            
            if total_iterations % 10000 == 0 and total_iterations > start_iteration:
                torch.save(style_net.state_dict(), f"/content/drive/MyDrive/style_transfer_final_{total_iterations}.pth")
                save_checkpoint(style_net, optimizer, scheduler, total_iterations, total_loss.item(), f"/content/drive/MyDrive/style_transfer_checkpoint_{total_iterations}.pth")
                
            if total_iterations >= total_steps:
                break
        
        if total_iterations >= total_steps:
            break
    
    # Save final model
    torch.save(style_net.state_dict(), '/content/drive/MyDrive/style_transfer_final.pth')
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
        
    stylized_tensor = stylized_tensor.clamp(0.0, 1.0)
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