from torch.utils.data import Dataset
from torch.utils import data as dataloader
from glob import glob
import os
from PIL import Image
import numpy as np
from filter_dataset import FilterDataset
import torch
import json
import yaml

class ImageDataset(Dataset):
    
    def __init__(self, config, data_type, transform):
        
        self.image_folder = config[data_type]['dataset_path']
        self.dataset_path = config['dataset_path']
        self.image_paths = list(glob(os.path.join(self.dataset_path, self.image_folder, '*.jpg')))
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

class VQADataset(Dataset):

    def __init__(self, config,
                 data_type,
                 vocab,
                 ans_vocab=None):
        
        self.config = config
        self.data_type = data_type
        self.features_dir = self.config[self.data_type]['features_path']
        self.dataset_path = self.config['dataset_path']
        self.token2idx = {}
        self.idx2token = {}
        self.ans_vocab = ans_vocab
        self.max_seq_length = self.config['max_seq_length']
        for idx, word in enumerate(vocab):
            if self.token2idx.get(word, None) is None:
                self.token2idx[word] = idx
                self.idx2token[idx] = word
        self.samples = self.filter_samples()
        self.sample_ids = list(self.samples.keys())
    
    def __len__(self):

        return len(self.sample_ids)
    
    def filter_samples(self):

        if self.ans_vocab is None:
            save_ans_vocab = True
        else:
            save_ans_vocab = False
        
        data = FilterDataset(self.config, 
                             self.data_type).filter()
        
        if save_ans_vocab:
            ans_vocab_path = os.path.join(self.dataset_path, self.config['ans_vocab_path'])
            with open(ans_vocab_path, 'r') as f:
                self.ans_vocab = json.load(f)
        
        self.ans2idx = {value:int(key) for key, value in self.ans_vocab.items()}

        total_samples = len(data)
        samples = {}
        for key in data.keys():
            if self.token2idx.get(data[key]['answer'], None) is None:
                continue
            samples[key] = data[key].copy()

        skipped = total_samples - len(samples)
        if skipped > 0:
            print(f"{skipped}/{total_samples} samples skipped!")
        
        return samples
    
    def __getitem__(self, idx):

        img_id = self.samples[self.sample_ids[idx]]['image_id']
        question = self.samples[self.sample_ids[idx]]['question']
        answer = self.samples[self.sample_ids[idx]]['answer']
        feature_path = os.path.join(self.dataset_path, self.features_dir, f'{img_id}.npy')

        features = np.zeros((1, 768))
        # features = np.load(feature_path)

        question = [self.token2idx.get(i, self.token2idx['<unk>']) for i in question.split()]
        question += [self.token2idx['<pad>']] * (self.max_seq_length - len(question))

        answer = self.ans2idx[answer]

        return {
            'image': torch.from_numpy(features),
            'question': torch.tensor(question),
            'answer': torch.tensor(answer)
        }


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_type = 'validation_data'
    ans_vocab = None

    vocab_path = os.path.join(config['embeddings_path'], config['vocab'])

    with open(os.path.join(config['dataset_path'], config['ans_vocab_path']), 'r') as f:
        ans_vocab = json.load(f)

    vocab = np.load(vocab_path)

    dataset = VQADataset(config, 
                         data_type,
                         vocab,
                         ans_vocab)
    
    item = next(iter(dataset))
    print(item['image'].shape)
    print(item['question'].shape)
    print(item['answer'].shape)