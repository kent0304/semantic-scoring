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

class MyDataset(Dataset):
    def __init__(self, dirname, p, transforms=transforms, device=None):
        """
        引数:
            dirnameは疑似生成主砲のディレクトリ名
            pは train/valid 
        """
        self.filename = os.path.join('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/', dirname, p, 'output.txt')
        self.df = pd.read_csv(self.filename, header=None, sep='\t')
        self.imageids = self.df.iloc[:,0]
        self.keys = self.df.iloc[:,1]
        self.anss = self.df.iloc[:,2]
        self.labels = self.df.iloc[:,3]
        self.transforms = transforms
        self.transformer = transforms.Compose([
            self.transforms.Resize(256),
            self.transforms.CenterCrop(224),
            self.transforms.ToTensor(),
            self.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_num = len(self.df)
        self.p = p

        # 画像
        self.image_net = models.resnet50(pretrained=True)
        self.image_net.fc = nn.Identity()
        self.image_net.eval()
        self.image_net = self.image_net.to(device)

        # SBERT
        self.sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):  
        # 画像
        if self.p == 'train':
            path = '/mnt/LSTA5/data/tanaka/lang-learn/coco/train2017/'
        else:
            path = '/mnt/LSTA5/data/tanaka/lang-learn/coco/val2017'
        l = len(str(self.imageids[idx]))
        f = (12-l)*'0' + str(self.imageids[idx]) + '.jpg'
        image_path = os.path.join(path,f)
        image = self.transformer(Image.open(image_path).convert('RGB')).to(device)
        #print(torch.unsqueeze(image,0))
        img = self.image_net(torch.unsqueeze(image,0))

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
# print(dataset[0])







