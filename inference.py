import os
import torch
from model import ModelGP, ModelRaw
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
import json
import torch.nn.functional as F


class Inference():

    def __init__(self, config, model_path):

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.total_answers = 71
        self.prepare_vocab()

    def prepare_vocab(self):

        vocab_path = os.path.join(self.config['embeddings_path'], self.config['vocab'])
        self.vocab = np.load(vocab_path)

        ans_vocab_path = os.path.join(self.config['dataset_path'], self.config['ans_vocab_path'])
        with open(ans_vocab_path, 'r') as f:
            self.ans_vocab = json.load(f)
        
        self.token2idx = {}
        self.idx2token = {}
        
        for idx, word in enumerate(self.vocab):
            if self.token2idx.get(word, None) is None:
                self.token2idx[word] = idx
                self.idx2token[idx] = word

    def load_models(self):

        if self.config['with_global_pool']:
            # With Global Pooling
            self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).eval().to(self.device)
            self.model = ModelGP(self.total_answers).eval().to(self.device)
        else:
            # Without Global Pooling
            self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, global_pool='').eval().to(self.device)
            self.model = ModelRaw(self.total_answers).eval().to(self.device)
        timm_config = resolve_data_config({}, model=self.vit_model)
        self.transform = create_transform(**timm_config)

        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict['state_dict'], strict=True)

        embeddings_path = os.path.join(self.config['embeddings_path'], self.config['embeddings'])
        embeddings = np.load(embeddings_path)
        
        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float()).to(self.device)

    def predict(self, image_path, question):

        if type(image_path) is str:
            img = Image.open(image_path).convert('RGB')
        else:
            img = Image.fromarray(image_path)
        
        image = self.transform(img).unsqueeze(0)

        question = [self.token2idx.get(i, self.token2idx['<unk>']) for i in question.split()]
        question += [self.token2idx['<pad>']] * (self.config['max_seq_length'] - len(question))
        question = torch.tensor(question).unsqueeze(0)

        with torch.no_grad():
            image = self.vit_model(image.to(self.device))
            question = self.embedding_layer(question.to(self.device))

            answer = self.model(image, question)

        return F.softmax(answer, dim=1).cpu()
    
    def predict_classes(self, image_path, question, top_n=5):

        predictions = self.predict(image_path, question)

        indices = predictions.argsort(descending=True)[0].numpy().tolist()
        scores = predictions[0].numpy().round(2)

        confidences = []
        classes = []

        for i in indices[:top_n]:
            
            confidences.append(scores[i])
            classes.append(self.ans_vocab[str(i)])
        
        return confidences, classes