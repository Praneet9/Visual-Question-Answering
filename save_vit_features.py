import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from glob import glob
import os
from tqdm import tqdm
import numpy as np


def vit_features_w_gp(dataset_path, folder):
    
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).cuda()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    features_path = os.path.join(dataset_path, 'vit_features_w_global_pool')
    if not os.path.exists(features_path):
        os.mkdir(features_path)
    
    for img_path in tqdm(glob(os.path.join(dataset_path, folder, '*.jpg'))):
        
        image_name = os.path.split(img_path)[-1]
        file_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(features_path, f'{file_name}.npy')
        
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).cuda() # transform and add batch dimension

        with torch.no_grad():
            out = model(tensor)
        
        np.save(output_path, out[0].cpu().numpy())

def vit_features_wo_gp(dataset_path, folder):
    
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, global_pool='').cuda()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    features_path = os.path.join(dataset_path, 'vit_features_raw')
    if not os.path.exists(features_path):
        os.mkdir(features_path)
    
    for img_path in tqdm(glob(os.path.join(dataset_path, folder, '*.jpg'))):
        
        image_name = os.path.split(img_path)[-1]
        file_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(features_path, f'{file_name}.npy')
        
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).cuda() # transform and add batch dimension

        with torch.no_grad():
            out = model(tensor)
        
        np.save(output_path, out[0].cpu().numpy())

if __name__ == "__main__":

    DATASET_PATH = 'dataset/'
    IMAGES_FOLDER = 'val2014'

    # Save ViT features without global_pool
    vit_features_wo_gp(DATASET_PATH, IMAGES_FOLDER)

    # Save ViT features with global_pool
    vit_features_w_gp(DATASET_PATH, IMAGES_FOLDER)
