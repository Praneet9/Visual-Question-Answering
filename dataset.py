from torch.utils.data import Dataset
from torch.utils import data as dataloader
from glob import glob
import os
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    
    def __init__(self, dataset_path, image_folder, transform):
        
        self.image_folder = image_folder
        self.dataset_path = dataset_path
        self.image_paths = list(glob(os.path.join(dataset_path, image_folder, '*.jpg')))
        self.transform = transform
        
    def __len__(self):
        
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        img_path = self.image_paths[idx]
        
        image_name = os.path.split(img_path)[-1]
        file_name = os.path.splitext(image_name)[0]
        
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img) # transform and add batch dimension

        return {
            'file_name': file_name,
            'img': tensor
        }

class ViTFeatures(Dataset):
    
    def __init__(self, dataset_path, features_folder):
        
        self.features_folder = features_folder
        self.dataset_path = dataset_path
        self.features_paths = list(glob(os.path.join(dataset_path, features_folder, '*.npy')))
        
    def __len__(self):
        
        return len(self.features_paths)
    
    def __getitem__(self, idx):
        
        feature_path = self.features_paths[idx]
        feature = np.load(feature_path)

        return feature
