import os
import pickle
import torch 
from torch import nn
import torchtext
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sentence_transformers import SentenceTransformer

device = torch.device('cuda:1')

class MyDataset(Dataset):
    def __init__(self, dirname, p, images, device=None):
        """
        引数:
            dirnameは疑似生成主砲のディレクトリ名
            pは train/valid 
        """
        self.filename = os.path.join('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/', dirname, p, 'output.txt')
        self.df = pd.read_csv(self.filename, header=None, sep='\t')
        self.keys = self.df.iloc[:,1]
        self.anss = self.df.iloc[:,2]
        self.labels = self.df.iloc[:,3]
        self.imgidx = self.df.iloc[:,4]
        self.data_num = len(self.df)
        self.p = p

        # 画像
        self.images = images

        # SBERT
        self.sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')



        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):  
        # 画像
        img = self.images[int(self.imgidx)]

        # 語句
        key = self.sbert_model.encode(self.keys[idx])

        # 解答文
        ans = self.sbert_model.encode(self.anss[idx])

        # 正誤ラベル
        label = torch.tensor(int(self.labels[idx]))
        

        return img, key, ans, label



# device = torch.device('cuda:0')

# dataset = MyDataset(dirname='berteach_wn05', p='train', device=device)
# print(len(dataset))


# from tqdm import tqdm
# i = 0
# for img, key, ans, label in tqdm(dataset, total=len(dataset)):
#     # print(i)
#     if not (torch.equal(label, torch.tensor(0)) or torch.equal(label, torch.tensor(1))):
#         print('これ', i)
#         print(img, key, ans, label)
#     i += 1


# print("Reading val data...")
# with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/{}/imagedata.pkl'.format(ver), 'rb') as f:
#     val_imagedata = pickle.load(f) 




