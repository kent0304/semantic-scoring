import pickle
import torch 
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, imagedata, keydata, ansdata, labeldata):
        self.imagevec = imagedata
        self.keyvec = keydata
        self.ansvec = ansdata
        self.label = labeldata
        self.data_num = len(self.imagevec)
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        image = self.imagevec[idx]
        key = self.keyvec[idx]
        ans = self.ansvec[idx]
        label = self.label[idx]
        return image, key, ans, label










