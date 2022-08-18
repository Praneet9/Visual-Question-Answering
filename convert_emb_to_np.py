import numpy as np
from tqdm import tqdm
import torch
import os


def convert_emb(embedding_name):

    vocab, embeddings = [], []

    emb_path = os.path.join('embeddings', embedding_name)
    emb_file_name = os.path.splitext(embedding_name)[0]

    with open(emb_path,'rt') as fi:
        full_content = fi.read().strip().split('\n')
    
    for i in tqdm(range(len(full_content))):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
    
    vocab = np.array(vocab)
    embeddings = np.array(embeddings)

    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab = np.insert(vocab, 0, '<pad>')
    vocab = np.insert(vocab, 1, '<unk>')
    print(vocab[:10])

    pad_emb_npa = np.zeros((1,embeddings.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embeddings,axis=0,keepdims=True)    #embedding for '<unk>' token.

    #insert embeddings for pad and unk tokens at top of embs_npa.
    embeddings = np.vstack((pad_emb_npa,unk_emb_npa,embeddings))
    print(embeddings.shape)

    my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())

    assert my_embedding_layer.weight.shape == embeddings.shape
    print(my_embedding_layer.weight.shape)

    vocab_op_path = os.path.join('embeddings', f'vocab_{emb_file_name}.npy')
    emb_op_path = os.path.join('embeddings', f'embeddings_{emb_file_name}.npy')

    with open(vocab_op_path,'wb') as f:
        np.save(f,vocab)

    with open(emb_op_path,'wb') as f:
        np.save(f,embeddings)

if __name__ == '__main__':

    EMBEDDING = 'glove.6B.300d.txt'

    convert_emb(EMBEDDING)