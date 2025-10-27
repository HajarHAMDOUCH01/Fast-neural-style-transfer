import os
import random
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from functools import lru_cache

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PlacesStreamingDataset(Dataset):
    def __init__(self, root_dir, total_images=150_000, transform=None, seed=42):
        self.transform = transform
        self.paths = self._collect_balanced_subset(root_dir, total_images, seed)
        print(f" Loaded {len(self.paths)} image paths.")

    def _collect_balanced_subset(self, root_dir, total_images, seed):
        random.seed(seed)
        from collections import defaultdict
        class_to_images = defaultdict(list)

        for initial in os.listdir(root_dir):
            init_path = os.path.join(root_dir, initial)
            if not os.path.isdir(init_path): 
                continue
            for cls in os.listdir(init_path):
                cls_path = os.path.join(init_path, cls)
                if not os.path.isdir(cls_path): 
                    continue
                imgs = [os.path.join(cls_path, f)
                        for f in os.listdir(cls_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    class_to_images[cls].extend(imgs)

        n_classes = len(class_to_images)
        n_per_class = total_images // n_classes
        selected = []
        for cls, imgs in class_to_images.items():
            random.shuffle(imgs)
            selected.extend(imgs[:n_per_class])
        random.shuffle(selected)
        return selected

    @lru_cache(maxsize=512)  # small in-RAM cache (~512 images)
    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f" Skipping image {path}: {e}")
            return torch.zeros((3, 256, 256))

    def __getitem__(self, idx):
        path = self.paths[idx]
        return self._load_image(path)

    def __len__(self):
        return len(self.paths)
