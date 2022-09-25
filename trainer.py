import torch.nn as nn
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from glob import glob
import os
from tqdm.notebook import tqdm
from torch.utils import data as dataloader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
from dataset import VQADataset
import numpy as np
from model import ModelGP, ModelRaw


class Trainer():
    
    def __init__(self, config):
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config['with_global_pool']:
            # # With Global Pooling
            # self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).eval().to(self.device)
            self.model = ModelGP(71).to(self.device)
        else:
            # # Without Global Pooling
            # self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, global_pool='').eval().to(self.device)
            self.model = ModelRaw(71).to(self.device)
        
        # timm_config = resolve_data_config({}, model=self.vit_model)
        # self.transform = create_transform(**timm_config)
        
        embeddings_path = os.path.join(self.config['embeddings_path'], self.config['embeddings'])
        vocab_path = os.path.join(self.config['embeddings_path'], self.config['vocab'])
        
        embeddings = np.load(embeddings_path)
        self.vocab = np.load(vocab_path)
        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float()).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        
        self.writer = SummaryWriter(
            comment=f"{type(self.model).__name__}"
        )
        self.epochs = self.config['epochs']
    
    def prepare_data(self):
        
        dataset = VQADataset(self.config, 'train_data', self.vocab)
        self.training_data = dataloader.DataLoader(dataset,
                                                batch_size=self.config['batch_size'],
                                                shuffle=True,
                                                num_workers=1)
        
        dataset = VQADataset(self.config, 'validation_data', self.vocab)
        self.validation_data = dataloader.DataLoader(dataset,
                                                batch_size=self.config['batch_size'],
                                                shuffle=False,
                                                num_workers=1)
    
    def train(self):
        
        for epoch in range(1, self.epochs+1):
            print(f"Epoch {epoch}:")
            train_loss = 0.0
            train_acc = 0.0
            
            self.model.train()
            pbar = tqdm(self.training_data)
            for idx, sample in enumerate(pbar):
                
                questions = sample['question'].to(self.device)
                answers = sample['answer'].to(self.device)
                images = sample['image'].to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    embeddings = self.embedding_layer(questions).detach()
                    # images = self.vit_model(images).detach()
                
                outputs = self.model(images, embeddings)
                loss = self.criterion(outputs, answers)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_acc += torch.sum(F.softmax(outputs, dim=1).argmax(dim=1) == answers).item() / outputs.shape[0]
                
                text = f"Batch ({idx+1}/{len(self.training_data)})"
                text += f"\tTraining Loss: {round(train_loss / (idx + 1), 4)}"
                text += f"\tTraining Accuracy: {round(train_acc / (idx + 1), 4)}"

                pbar.set_description(text)
                # print(text, end='\r')
            else:
                
                val_loss, val_acc = self.evaluate(self.validation_data)

                train_loss /= len(self.training_data)
                train_acc /= len(self.training_data)

                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/validation', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/validation', val_acc, epoch)

                text = f"Training Loss:{round(train_loss, 4)} Training Accuracy:{round(train_acc, 4)}"
                text += f"\tValidation Loss: {round(val_loss, 4)} Validation Accuracy:{round(val_acc, 4)}\n"
                print(text)
        
        self.writer.flush()
        self.writer.close()
        
        # Saving best and latest model
        name = f"{type(self.model).__name__}_EPOCHS_{epoch}"
        model_name = f"{name}_VAL_LOSS_{round(val_loss, 5)}.pth"
        self.save_model(model_name)
                
    
    def evaluate(self, test_data):
        
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0

        with torch.no_grad():

            for sample in tqdm(test_data):
                
                questions = sample['question'].to(self.device)
                answers = sample['answer'].to(self.device)
                images = sample['image'].to(self.device)
                
                # Forward pass
                embeddings = self.embedding_layer(questions)
                # images = self.vit_model(images).detach()
                
                outputs = self.model(images, embeddings)
                pred_loss = self.criterion(outputs, answers)

                test_loss += pred_loss.item()
                test_acc += torch.sum(F.softmax(outputs, dim=1).argmax(dim=1) == answers).item() / outputs.shape[0]
                
            test_loss /= len(test_data)
            test_acc /= len(test_data)

        return test_loss, test_acc
    
    def save_model(self, file_path='torch_model.pth'):
        
        torch.save({'state_dict': self.model.state_dict()}, file_path)

        print(f"Model saved to {file_path}")



if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    trainer = Trainer(config)
    trainer.prepare_data()
    trainer.train()