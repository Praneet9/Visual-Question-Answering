import torch.nn as nn
import torch


class ModelGP(nn.Module):
    
    def __init__(self, output_vocab, seq_len=20):
        
        super().__init__()
        
        self.lstm_1 = nn.LSTM(300, 96, batch_first=True, num_layers=2, bidirectional=True, dropout=0.5)
        self.fc_1 = nn.Linear(seq_len*96*2, 768)
        
        self.output = nn.Sequential(nn.ReLU(),
                                    nn.Linear(768, 384),
                                    nn.ReLU(),
                                    nn.Linear(384, output_vocab))
        
    def forward(self, image, embs):
        
        embs, _ = self.lstm_1(embs)
        embs = self.fc_1(torch.flatten(embs, start_dim=1))
        x = torch.add(image, embs)
        
        return self.output(x)