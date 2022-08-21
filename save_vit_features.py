import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import os
from tqdm import tqdm
import numpy as np
from dataset import ImageDataset
from torch.utils import data as dataloader
import argparse

class SaveViTFeatures():

    def __init__(self, dataset_path, images_folder, with_global_pool=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if with_global_pool:
            self.model = timm.create_model('vit_base_patch16_224', 
                                    pretrained=True, 
                                    num_classes=0).eval().to(self.device)
            config = resolve_data_config({}, model=self.model)
            transform = create_transform(**config)

            self.features_path = os.path.join(dataset_path, 'vit_features_w_global_pool')
            if not os.path.exists(self.features_path):
                os.mkdir(self.features_path)

        else:
            self.model = timm.create_model('vit_base_patch16_224',
                                    pretrained=True, 
                                    num_classes=0,
                                    global_pool='').eval().to(self.device)
            config = resolve_data_config({}, model=self.model)
            transform = create_transform(**config)

            self.features_path = os.path.join(dataset_path, 'vit_features_raw')
            if not os.path.exists(self.features_path):
                os.mkdir(self.features_path)
        
        dataset = ImageDataset(dataset_path, images_folder, transform)
        self.dataset = dataloader.DataLoader(dataset,
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=2)
    
    def save_features(self):
        
        for batch in tqdm(self.dataset):
            with torch.no_grad():
                features = self.model(batch['img'].to(self.device)).cpu()

            for idx, feat in enumerate(features):
                
                if type(batch['file_name']) is list:
                    file_name = batch['file_name'][idx]
                else:
                    file_name = batch['file_name']

                output_path = os.path.join(self.features_path, f'{file_name}.npy')
                np.save(output_path, feat.numpy())

def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Save ViT intermediate features"
    )
    
    parser.add_argument(
        "--dataset", dest="dataset", type=str, required=True,
        default='dataset/', help="Path to the dataset"
    )
    parser.add_argument(
        "--images-path", dest="images_path", type=str, required=True,
        help="Images folder"
    )
    parser.add_argument(
        "--with-global-pool", dest="with_global_pool", action='store_false',
        help="Pretrained ViT with global pooling layer"
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    DATASET_PATH = args.dataset
    IMAGES_FOLDER = args.images_path

    SaveViTFeatures(DATASET_PATH, 
                    IMAGES_FOLDER, 
                    args.with_global_pool).save_features()
