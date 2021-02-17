import os
import pickle
import torch 
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

class MyDataset(Dataset):
    def __init__(self, dirname, p):
        self.filename = os.path.join('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/', dirname, p)
        self.data_num = len(self.filename)
        self.sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        with open(self.filename) as f:
            

        
        image = self.imagevec[self.imagedata[idx]]

        self.sbert_model.encode(text)
        key = self.keyvec[idx]
        ans = self.ansvec[idx]
        label = self.label[idx]
        return image, key, ans, label










