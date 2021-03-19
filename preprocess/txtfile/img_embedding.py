# テキストファイルから画像のembeddingをpickle保存するファイル
import os
import csv
csv.field_size_limit(10000000000)
import pickle
import pandas as pd
import torch 
from torch import nn
import torchtext
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

device = torch.device('cuda:1')


def image2vec(image_net, image_paths):
    # 画像を Tensor に変換
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # stackはミニバッチに対応できる
    images = torch.stack([
        transformer(Image.open(image_path).convert('RGB'))
        for image_path in image_paths
    ])
    images = images.to(device)
    images = image_net(images)
    return images.cpu()

def make_imagedata(imgs, datanum):
    # resnet呼び出し
    image_net = models.resnet50(pretrained=True)
    image_net.fc = nn.Identity()
    image_net.eval()
    image_net = image_net.to(device)

    batch_size = 512

    image_vec = torch.zeros((datanum , 2048))
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            image_paths = [imgs[j] for j in range(i, len(imgs))[:batch_size]]
            images = image2vec(image_net, image_paths)
            image_vec[i:i + batch_size] = images

            # if i >= 10*batch_size:
            #     exit(0)
        
    return image_vec

# with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imageid_list.pkl', 'rb')as f:
#     imageid_list = pickle.load(f)
# print('ファイルに書き込む際に確認した数：', len(set(imageid_list)))

train_filename = '/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/bert_wn05/train/output.txt'
valid_filename = '/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/bert_wn05/valid/output.txt'

train_df = pd.read_csv(train_filename, header=None, sep='\t')
valid_df = pd.read_csv(valid_filename, header=None, sep='\t')
train_imageids = train_df.iloc[:,0]
valid_imageids = valid_df.iloc[:,0]

print('ファイルから読み込んで確認した数',len(set(train_imageids)))
print('ファイルから読み込んで確認した数', len(set(valid_imageids)))


train_imageids_list = []
train_check = []
path = '/mnt/LSTA5/data/tanaka/lang-learn/coco/train2017/'
for id in tqdm(train_imageids, total=len(train_imageids)):
    id = str(id)
    if id in train_check:
        continue
    else:
        l = len(str(id))
        f = path + (12-l)*'0' + id + '.jpg'
        train_imageids_list.append(f)
        train_check.append(id)
# 動詞語句が存在する画像数
train_datanum = len(train_imageids_list)
print('embeddingに変換する画像数', len(train_imageids_list))


valid_imageids_list = []
valid_check = []
path = '/mnt/LSTA5/data/tanaka/lang-learn/coco/val2017/'
for id in tqdm(valid_imageids, total=len(valid_imageids)):
    if id in valid_check:
        continue
    else:
        l = len(str(id))
        f = path + (12-l)*'0' + str(id) + '.jpg'
        valid_imageids_list.append(f)
        valid_check.append(id)
# 動詞語句が存在する画像数
# print('数',len(valid_check))
# print('数',len(valid_imageids_list))
valid_datanum = len(valid_imageids_list)
print('embeddingに変換する画像数',len(valid_imageids_list))




# Make image data 2048 dim
print("画像をベクトル化")
train_images = make_imagedata(train_imageids_list, train_datanum)
valid_images = make_imagedata(valid_imageids_list, valid_datanum)

print(train_images.shape)
print(valid_images.shape)
print("picleで保存")
with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imgs/train2017_images.pkl', 'wb') as f:
    pickle.dump(train_images, f) 
with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imgs/val2017_images.pkl', 'wb') as f:
    pickle.dump(valid_images, f) 

