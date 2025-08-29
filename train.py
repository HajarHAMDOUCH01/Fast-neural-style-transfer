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
from model import StyleTransferNet, VGG16, gram_matrix, style_loss, total_variation_loss, content_loss

TOTAL_STEPS = 40000
BATCH_SIZE      = 4  
LEARNING_RATE   = 1e-2
NUM_EPOCHS      = 2

CONTENT_WEIGHT = 1.0
STYLE_WEIGHT   = 3
TV_WEIGHT      = 1e-4

TRAIN_IMAGE_SHAPE = (256, 256)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def normalize_batch(batch):
    """Ensure batch is in [0,1] range for VGG"""
    return torch.clamp(batch, 0.0, 1.0)

def denormalize_batch(batch):
    """Convert from [-1,1] to [0,1] range"""
    return (batch + 1.0) / 2.0  

def load_style_image(style_path, size=TRAIN_IMAGE_SHAPE):
    style_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()  
    ])
    
    style_img = Image.open(style_path)
    style_img = style_transform(style_img)
    return style_img.unsqueeze(0).to(device)

def get_style_targets(vgg, style_img):
    vgg.eval()
    with torch.no_grad():
        style_features = vgg(style_img) 
        print("style_features shape : ", style_features[0].shape)
        style_targets = []
        
        for feat in style_features:
            gram = gram_matrix(feat) 
            style_targets.append(gram.squeeze(0)) # removing batch dim ! -> check
    return style_targets

class Dataset(torch.utils.data.Dataset): 
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
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image) #!!
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.images)))

def load_model_from_checkpoint(checkpoint_path):
    start_iteration = checkpoint_path['iteartion']
    style_transfer_net = StyleTransferNet().to(device)
    model = checkpoint_path['model']
    stn_stat_dict = torch.load(model, map_location=device)
    style_transfer_net.load_state_dict(stn_stat_dict)
    adam_optimizer = optim.Adam(style_transfer_net.parameters(),
                        lr=LEARNING_RATE,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=1e-5)  
    optimizer_checkpoint = checkpoint_path['optimizer']
    adam_stat_dict = torch.load(optimizer_checkpoint, map_location=device)
    adam_optimizer.load_state_dict(adam_stat_dict)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        adam_optimizer,
        T_max=TOTAL_STEPS - start_iteration,
        eta_min=1e-6
    )
    scheduler_checkpoint = checkpoint_path['scheduler']
    scheduler_stat_dict = torch.load(scheduler_checkpoint, map_location=device)
    scheduler.load_state_dict(scheduler_stat_dict)
    return start_iteration

def train_style_transfer():
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SHAPE),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),  
        transforms.Lambda(lambda x: x * 2.0 - 1.0)  # Convert to [-1,1] for model input
    ])
    
    dataset = Dataset(root='/kaggle/input/coco-2017-dataset/coco2017/train2017', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=2, pin_memory=True)
    
    vgg = VGG16().to(device)
    vgg.eval()
    
    # Load and process style image consistently
    style_transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SHAPE),
        transforms.ToTensor()  # Keep in [0,1] for VGG
    ])
    
    style_img = Image.open('/content/style.jpg')
    style_img = style_transform(style_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        style_targets = get_style_targets(vgg, style_img)
        style_targets = [t.detach() for t in style_targets]
    
    style_net = StyleTransferNet().to(device)
    
    optimizer = optim.Adam(style_net.parameters(),
                          lr=LEARNING_RATE,
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_STEPS,
        eta_min=1e-6
    )
    
    style_net.train()
    total_iterations = 0
    running_loss = 0.0
    
    for epoch in range(NUM_EPOCHS):
        for batch_idx, content_batch in enumerate(dataloader):
            if total_iterations >= TOTAL_STEPS:
                break
                
            content_batch = content_batch.to(device)  # [-1,1] range
            
            # Generate stylized output
            stylized_batch = style_net(content_batch)  # [-1,1] range
            
            # Convert to [0,1] for VGG processing
            content_batch_vgg = denormalize_batch(content_batch)
            stylized_batch_vgg = denormalize_batch(stylized_batch)
            
            # Ensure values are in valid range
            content_batch_vgg = normalize_batch(content_batch_vgg)
            stylized_batch_vgg = normalize_batch(stylized_batch_vgg)
            
            # Extract features
            content_features = vgg(content_batch_vgg)
            stylized_features = vgg(stylized_batch_vgg)
            
            # Calculate losses
            c_loss = content_loss(stylized_features, content_features)
            s_loss = style_loss(stylized_features, style_targets)
            tv_loss = total_variation_loss(stylized_batch)
            
            total_loss = (CONTENT_WEIGHT * c_loss + 
                         STYLE_WEIGHT * s_loss + 
                         TV_WEIGHT * tv_loss)
            
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
                print(f"Iteration [{total_iterations}/{TOTAL_STEPS}] "
                      f"Total: {avg_loss:.4f} "
                      f"Content: {c_loss.item():.4f} "
                      f"Style: {s_loss.item():.4f} "
                      f"TV: {tv_loss.item():.6f} "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0
                
                # Save sample output for debugging
                if total_iterations % 1000 == 0:
                    with torch.no_grad():
                        sample_output = denormalize_batch(stylized_batch[0:1])
                        sample_output = torch.clamp(sample_output * 255, 0, 255)
                        sample_img = transforms.ToPILImage()(sample_output[0].cpu())
                        sample_img.save(f"/content/sample_{total_iterations}.jpg")
            
            # Save checkpoints
            if total_iterations % 5000 == 0 and total_iterations > 0:
                torch.save({
                    'model_state_dict': style_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iteration': total_iterations,
                    'loss': total_loss.item()
                }, f"/content/drive/MyDrive/checkpoint_{total_iterations}.pth")
                
            if total_iterations >= TOTAL_STEPS:
                break
        
        if total_iterations >= TOTAL_STEPS:
            break
    
    # Save final model
    torch.save(style_net.state_dict(), '/content/drive/MyDrive/style_transfer_final.pth')
    print("Training completed!")

def test_inference(model_path, content_path, output_path):
    style_net = StyleTransferNet().to(device)
    style_net.load_state_dict(torch.load(model_path, map_location=device))
    style_net.eval()
    
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SHAPE),
        transforms.ToTensor(),  
        transforms.Lambda(lambda x: x * 2.0 - 1.0)  
    ])
    
    content_img = Image.open(content_path)
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stylized_tensor = style_net(content_tensor)  
        stylized_tensor = denormalize_batch(stylized_tensor)
        stylized_tensor = torch.clamp(stylized_tensor * 255, 0, 255)
        
    stylized_img = transforms.ToPILImage()(stylized_tensor[0].cpu())
    stylized_img.save(output_path)
    print(f"Stylized image saved to {output_path}")


if __name__ == '__main__':

    # if training from the start : 
    train_style_transfer()

    # if training from a checkpoint : 
    # train_style_transfer(resume_from_checkpoint=True, checkpoint_path='/content/drive/MyDrive/style_transfer_checkpoint_10000.pth')

    # test_inference('style_transfer_final.pth', 'test_content.jpg', 'stylized_output.jpg')