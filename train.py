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

def get_style_targets(vgg, style_img=style_image):
    vgg.eval()
    with torch.no_grad():
        style_features = vgg(style_img) 
        print("style_features shape : ", style_features[0].shape)
        style_targets = []
        
        for feat in style_features:
            gram = gram_matrix(feat) 
            style_targets.append(gram.squeeze(0)) # removing batch dim ! -> check
    print("image style shape after vgg", style_targets[0].shape)
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
    vgg = VGG19().to(device)
    vgg.eval()

    vgg_weights = vgg.vgg_model_weights.IMAGENET1K_V1
    transform = vgg_weights.transforms()

    dataset = Dataset(root=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=2, pin_memory=True)
    
    style_img = Image.open(style_image) 
    style_img = transform(style_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        style_targets = get_style_targets(vgg, style_img) # without batch dim 
        style_targets = [t.detach() for t in style_targets]
    
    style_net = StyleTransferNet().to(device)
    
    optimizer = optim.Adam(style_net.parameters(),
                          lr=lr,
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )
    
    style_net.train()
    total_iterations = 0
    running_loss = 0.0
    
    for epoch in range(num_epochs):
        for batch_idx, content_batch in enumerate(dataloader):
            if total_iterations >= total_steps:
                break
                
            content_batch = content_batch.to(device) 
            # print("content_batch : ",content_batch[0].shape)
            
            # Generate stylized output
            stylized_batch = style_net(content_batch) 
            # print("stylized_batch : ",stylized_batch[0].shape)
             # 224*224
            
            # Convert to [0,1] for VGG processing
            # content_batch_vgg = denormalize_batch(content_batch)
            # stylized_batch_vgg = denormalize_batch(stylized_batch)
            
            # Extract features
            content_features = vgg(content_batch)
            # print("content_features : ",content_features[0].shape)
            stylized_features = vgg(stylized_batch) 
            # print("stylized_features : ",stylized_features[0].shape)

            # Calculate losses
            c_loss = content_loss(stylized_features, content_features)
            s_loss = style_loss(stylized_features, style_targets)
            tv_loss = total_variation_loss(stylized_batch)
            
            total_loss = (content_weight * c_loss + 
                         style_weight * s_loss + 
                         tv_weight * tv_loss)
            
            # Check for NaN
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Invalid loss at iteration {total_iterations}")
                print(f"Content: {c_loss.item():.6f}, Style: {s_loss.item():.6f}, TV: {tv_loss.item():.6f}")
                continue
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(style_net.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += total_loss.item()
            total_iterations += 1
            
            # Logging with better frequency
            if total_iterations % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Iteration [{total_iterations}/{total_steps}] "
                      f"Total: {avg_loss:.4f} "
                      f"Content: {c_loss.item():.4f} "
                      f"Style: {s_loss.item():.4f} "
                      f"TV: {tv_loss.item():.6f} "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0
              
            if total_iterations % 1000 == 0:
                image = Image.open(training_monitor_content_image).convert("RGB")
                content_image = transform(image).unsqueeze(0).to(device)
                # print("image tensor shape : ", content_image[0].shape)
                # print("image tensor to check values : ", content_image)
                stylized_tensor = style_net(content_image)
                # print("output image tensor to check values : ", sample_image)
                stylized_tensor = denormalize_batch(stylized_tensor)
                stylized_tensor = torch.clamp(stylized_tensor * 255, 0, 255)
        
                stylized_img = transforms.ToPILImage()(stylized_tensor[0].to(device))
                stylized_img.save(f"{output_dir}/sample_image_{total_iterations}.jpg")
                print(f"Stylized image saved {total_iterations}")

            # Save checkpoints
            if total_iterations % 5000 == 0 and total_iterations > 0:
                torch.save({
                    'model_state_dict': style_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iteration': total_iterations,
                    'loss': total_loss.item()
                }, f"{output_dir}/checkpoint_{total_iterations}.pth")
                
            if total_iterations >= total_steps:
                break
        
        if total_iterations >= total_steps:
            break
    
    # Save final model
    torch.save(style_net.state_dict(), f"{output_dir}/style_transfer_final.pth")
    print("Training completed!")

