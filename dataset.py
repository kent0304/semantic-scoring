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
        # SBERTロード
        self.sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

        # 語句
        keyvec_path = '/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/{}/{}/key_tensor.pkl'.format(dirname, p)
        if os.path.exists(keyvec_path):
            with open(keyvec_path, 'rb') as f:
                self.keyvec = pickle.load(f)
        else:
            self.keys = self.df.iloc[:,1]
            self.keyvec = torch.zeros(len(self.keys), 768)
            for i in tqdm(range(len(self.keys)), total=len(self.keys)):
                self.keyvec[i] = torch.from_numpy(self.sbert_model.encode(self.df.iloc[i,1])).clone()
            with open(keyvec_path, 'wb') as f:
                pickle.dump(self.keyvec, f, protocol=4)

        # 解答文
        ansvec_path = '/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/{}/{}/ans_tensor.pkl'.format(dirname, p)
        if os.path.exists(ansvec_path):
            with open(ansvec_path, 'rb') as f:
                self.ansvec = pickle.load(f)
        else:
            self.anss = self.df.iloc[:,2]
            self.ansvec = torch.zeros(len(self.anss), 768)
            for i in tqdm(range(len(self.anss)), total=len(self.anss)):
                self.ansvec[i] = torch.from_numpy(self.sbert_model.encode(self.df.iloc[i,2])).clone()
            with open(ansvec_path, 'wb') as f:
                pickle.dump(self.ansvec, f, protocol=4)

        
        # ラベル
        self.labels = self.df.iloc[:,3]
        # 画像
        self.images = images
        # 画像のインデックス
        self.imgidx = self.df.iloc[:,4]
        # データ数
        self.data_num = len(self.df)
        # train or valid
        self.p = p
        print('データセットの読み込みが完了しました。')
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):  
        # 画像
        img = self.images[torch.tensor(self.imgidx[idx], dtype=torch.long)]
        # 語句
        key = self.keyvec[idx]
        # 解答文
        ans = self.ansvec[idx]
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