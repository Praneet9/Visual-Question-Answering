import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import os
from tqdm import tqdm
import numpy as np
from dataset import ImageDataset
from torch.utils import data as dataloader
import yaml

class SaveViTFeatures():

    def __init__(self, config, data_type):

        dataset_path = config['dataset_path']
        with_global_pool = config['with_global_pool']
        images_folder = config[data_type]['images_path']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if with_global_pool:
            features_dir = config[data_type]['features_path']
            self.model = timm.create_model('vit_base_patch16_224', 
                                            pretrained=True, 
                                            num_classes=0).eval().to(self.device)
        else:
            features_dir = config[data_type]['raw_features_path']
            self.model = timm.create_model('vit_base_patch16_224',
                                            pretrained=True, 
                                            num_classes=0,
                                            global_pool='').eval().to(self.device)
        
        timm_config = resolve_data_config({}, model=self.model)
        transform = create_transform(**timm_config)

        self.features_path = os.path.join(dataset_path, features_dir)
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


if __name__ == "__main__":

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_type = 'validation_data'

    SaveViTFeatures(config, 
                    data_type).save_features()
