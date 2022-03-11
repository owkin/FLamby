import torch 
from PIL import Image
import albumentations
import numpy as np
from dataset_creation_scripts.color_constancy import color_constancy
import os

class MelonamaDataset:
    def __init__(self, image_paths, targets, augmentations=None, cc=False, meta_array=None):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.cc = cc
        self.meta_array = meta_array

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        target = self.targets[idx]

        # color constancy if not preprocessed          
        if self.cc: 
            image = color_constancy(image)
        
        # meta data and augmentations
        if self.meta_array is not None:
            meta  = self.meta_array[idx]
            # gender
            if torch.rand(1)<0.1:
                meta[:2] = torch.zeros(2)      
            # anatom_site
            if torch.rand(1)<0.1:
                meta[2:8] = torch.zeros(6)      
            # age_approx
            if torch.rand(1)<0.1:
                meta[8] = torch.tensor(-0.05)      

        # Image augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)      

        return {
            'image': torch.tensor(image, dtype=torch.float), 
            'target': torch.tensor(target, dtype=torch.long)
        } if self.meta_array is None else {
            'image': torch.tensor(image, dtype=torch.float), 
            'target': torch.tensor(target, dtype=torch.long),
            'meta': torch.tensor(meta, dtype=torch.float)
        }


class MelonamaTTADataset:
    """Only useful for TTA during evaluation"""
    def __init__(self, image_paths, augmentations=None, meta_array=None, nc=None):
        self.image_paths = image_paths
        self.augmentations = augmentations 
        self.meta_array = meta_array
        self.nc = nc
        
    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        # dummy targets
        target = torch.zeros(5 if self.nc==5 else 10)
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.augmentations(image)

        if self.meta_array is not None:
            meta  = self.meta_array[idx]
        
        return {
            'image':image, 
            'target':target
        } if self.meta_array is None else {
            'image': image, 
            'target': target,
            'meta': torch.tensor(meta, dtype=torch.float)
        }