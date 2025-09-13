import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import os
import numpy as np

import sys
sys.path.append('/content/real-time-neural-style-transfer')
from losses.losses import gram_matrix, style_loss, total_variation_loss, content_loss
from models.model import StyleTransferNet
from models.vgg19_net import VGG19
from config import training_config, loss_weights_config, vgg_loss_layers, dataset_dir, training_monitor_content_image, style_image
from utils.image_utils import normalize_batch, denormalize_batch
from data.dataset import Dataset
from inference import test_inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_style_targets(vgg, style_img):
    """Extract style targets from style image"""
    vgg.eval()
    with torch.no_grad():
        style_features = vgg(style_img) 
        print("Style features shapes:", [f.shape for f in style_features])
        
        style_targets = []
        for feat in style_features:
            gram = gram_matrix(feat) 
            style_targets.append(gram.squeeze(0).detach())  
            
    return style_targets 

def load_model_from_checkpoint(checkpoint_path, lr, total_steps, ):
    start_iteration = checkpoint_path['iteartion']
    style_transfer_net = StyleTransferNet().to(device)
    model = checkpoint_path['model']
    stn_stat_dict = torch.load(model, map_location=device)
    style_transfer_net.load_state_dict(stn_stat_dict)
    adam_optimizer = optim.Adam(style_transfer_net.parameters(),
                        lr=lr,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=1e-5)  
    optimizer_checkpoint = checkpoint_path['optimizer']
    adam_stat_dict = torch.load(optimizer_checkpoint, map_location=device)
    adam_optimizer.load_state_dict(adam_stat_dict)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        adam_optimizer,
        T_max=total_steps - start_iteration,
        eta_min=1e-6
    )
    scheduler_checkpoint = checkpoint_path['scheduler']
    scheduler_stat_dict = torch.load(scheduler_checkpoint, map_location=device)
    scheduler.load_state_dict(scheduler_stat_dict)
    return start_iteration

def train_style_transfer(
        style_image,
        training_monitor_content_image,
        dataset_dir,
        output_dir,
        content_weight,
        style_weight,
        tv_weight,
        num_epochs,
        batch_size,
        total_steps,
        lr
):
    # Initialize VGG19 for loss calculation
    vgg = VGG19().to(device)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False  # Freeze VGG parameters

    # Use proper ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Transform for training data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
        normalize
    ])

    # Load dataset
    dataset = Dataset(root=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=2, pin_memory=True, drop_last=True)
    
    # Load and preprocess style image
    style_img = Image.open(style_image).convert('RGB')
    style_img = transform(style_img).unsqueeze(0).to(device)
    
    # Extract style targets
    with torch.no_grad():
        style_targets = get_style_targets(vgg, style_img)
    
    # Initialize style transfer network
    style_net = StyleTransferNet().to(device)
    
    optimizer = optim.Adam(style_net.parameters(),
                          lr=lr * 0.1,  
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-7
    )
    
    style_net.train()
    total_iterations = 0
    running_loss = 0.0
    running_content_loss = 0.0
    running_style_loss = 0.0
    running_tv_loss = 0.0
    
    print("Starting training...")
    
    for _ in range(num_epochs):
        for _, content_batch in enumerate(dataloader):
            if total_iterations >= total_steps:
                break
                
            content_batch = content_batch.to(device)
            
            # Generate stylized output
            stylized_batch = style_net(content_batch)
            
            # Clamp output to reasonable range
            stylized_batch = torch.clamp(stylized_batch, -3, 3)
            
            # Extract features for loss calculation
            with torch.no_grad():
                content_features = vgg(content_batch)
            
            stylized_features = vgg(stylized_batch)

            # Calculate losses
            c_loss = content_loss(stylized_features, content_features)
            s_loss = style_loss(stylized_features, style_targets)
            tv_loss = total_variation_loss(stylized_batch)
            
            # Scale losses appropriately
            total_loss = (content_weight * c_loss + 
                         style_weight * s_loss + 
                         tv_weight * tv_loss)
            
            # Check for NaN/inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Invalid loss at iteration {total_iterations}")
                print(f"Content: {c_loss.item():.6f}, Style: {s_loss.item():.6f}, TV: {tv_loss.item():.6f}")
                continue
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(style_net.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update running losses
            running_loss += total_loss.item()
            running_content_loss += c_loss.item()
            running_style_loss += s_loss.item()
            running_tv_loss += tv_loss.item()
            
            total_iterations += 1
            
            # Logging
            if total_iterations % 100 == 0:
                avg_loss = running_loss / 100
                avg_content = running_content_loss / 100
                avg_style = running_style_loss / 100
                avg_tv = running_tv_loss / 100
                
                print(f"Iter [{total_iterations}/{total_steps}] "
                      f"Total: {avg_loss:.4f} | "
                      f"Content: {avg_content:.4f} | "
                      f"Style: {avg_style:.4f} | "
                      f"TV: {avg_tv:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
                
                # Reset running losses
                running_loss = 0.0
                running_content_loss = 0.0
                running_style_loss = 0.0
                running_tv_loss = 0.0
            
            if total_iterations % 10000 == 0:
                content_weight = content_weight / 10
                style_weight = style_weight * 10
              
            # Generate sample images
            if total_iterations % 1000 == 0:
                style_net.eval()
                with torch.no_grad():
                    # Load and preprocess test image
                    test_image = Image.open(training_monitor_content_image).convert("RGB")
                    test_tensor = transform(test_image).unsqueeze(0).to(device)
                    
                    # Generate stylized image
                    stylized_tensor = style_net(test_tensor)
                    
                    # Denormalize and convert to PIL
                    # Reverse the normalization
                    denorm = transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225]
                    )
                    stylized_tensor = denorm(stylized_tensor[0])
                    stylized_tensor = torch.clamp(stylized_tensor, 0, 1)
                    
                    # Convert to PIL and save
                    stylized_img = transforms.ToPILImage()(stylized_tensor.cpu())
                    stylized_img.save(f"{output_dir}/sample_image_{total_iterations}.jpg")
                    print(f"Sample image saved: {total_iterations}")
                
                style_net.train()

            # Save checkpoints
            if total_iterations % 5000 == 0 and total_iterations > 0:
                torch.save({
                    'model_state_dict': style_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iteration': total_iterations,
                    'loss': total_loss.item()
                }, f"{output_dir}/checkpoint_{total_iterations}.pth")
                print(f"Checkpoint saved: {total_iterations}")
                
            if total_iterations >= total_steps:
                break
        
        if total_iterations >= total_steps:
            break
    
    # Save final model
    torch.save(style_net.state_dict(), f"{output_dir}/style_transfer_final.pth")
    print("Training completed!")

