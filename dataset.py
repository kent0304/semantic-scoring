import os
import pickle
from tqdm import tqdm
import torch 
from torch import nn
import torchtext
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sentence_transformers import SentenceTransformer


class MyDataset(Dataset):
    def __init__(self, dirname, p, images, device='cpu'):
        """
        引数:
            dirnameは疑似生成手法のディレクトリ名
            pは train/valid 
        """
        # テキストファイルロード
        self.filename = os.path.join('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/', dirname, p, 'output.txt')
        self.df = pd.read_csv(self.filename, header=None, sep='\t')
        self.keys = self.df.iloc[:,1]
        # 解答文
        self.anss = self.df.iloc[:,2]
        self.sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self.ansvec = torch.zeros(len(self.df.iloc[:,2]), 768)
        for i in tqdm(range(len(self.df.iloc[:,2])), total=len(self.df.iloc[:,2])):
            self.ansvec[i] = torch.from_numpy(self.sbert_model.encode(self.df.iloc[i,2])).clone()
        with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/{}/{}/ans_tensor.pkl'.format(dirname, p), 'wb') as f:
            pickle.dump(self.ansvec, f)
        # ラベル
        self.labels = self.df.iloc[:,3]
        self.imgidx = self.df.iloc[:,4]
        self.data_num = len(self.df)
        self.p = p
        print('データセットの読み込みが完了しました。')

        # 画像
        self.images = images

        



        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):  
        # 画像
        img = self.images[torch.tensor(self.imgidx[idx], dtype=torch.long)]

        # 語句
        key = self.sbert_model.encode(self.keys[idx])

        # 解答文
        ans = self.sbert_model.encode(self.anss[idx])

        # 正誤ラベル
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        

        return img, key, ans, label



# ver = 'berteach_wn05'

# # 画像のキャッシュをロード
# # with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imgs/train2017_images.pkl', 'rb') as f:
# #     train_imagedata = pickle.load(f) 
# with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imgs/val2017_images.pkl', 'rb') as f:
#     val_imagedata = pickle.load(f) 


# # print('train画像の数：', len(train_imagedata))
# # print('val画像の数：', len(val_imagedata))

# # # train_dataset = MyDataset(dirname=ver, p='train', images=train_imagedata)
# valid_dataset = MyDataset(dirname=ver, p='valid', images=val_imagedata)


# # # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

# for img, key, ans, label in valid_loader:
#     print(img, key, ans, label)