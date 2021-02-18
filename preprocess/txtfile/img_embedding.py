
import os
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

def make_imagedata(imgs):
    # resnet呼び出し
    image_net = models.resnet50(pretrained=True)
    image_net.fc = nn.Identity()
    image_net.eval()
    image_net = image_net.to(device)
    data_num = len(imgs)
    image_vec = torch.zeros((data_num, 2048))
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            image_paths = [imgs[j] for j in range(i, len(imgs))[:batch_size]]
            images = image2vec(image_net, image_paths)
            image_vec[i:i + batch_size] = images

            # if i >= 10*batch_size:
            #     exit(0)
        
    return image_vec

train_filename = '/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/bert_wn05/train/output.txt'
valid_filename = '/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/bert_wn05/valid/output.txt'
train_df = pd.read_csv(train_filename, header=None, sep='\t')
valid_df = pd.read_csv(valid_filename, header=None, sep='\t')
train_imageids = train_df.iloc[:,0]
valid_imageids = valid_df.iloc[:,0]

train_imageids_list = []
train_check = []
path = '/mnt/LSTA5/data/tanaka/lang-learn/coco/train2017/'
for id in tqdm(train_imageids, total=len(train_imageids)):
    if id in train_check:
        continue
    else:
        l = len(str(id))
        f = path + (12-l)*'0' + str(id) + '.jpg'
        train_imageids_list.append(f)
        train_check.append(id)

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



# Make image data 2048 dim
print("画像をベクトル化")
train_images = make_imagedata(train_imageids_list)
valid_images = make_imagedata(valid_imageids_list)
print("picleで保存")
with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imgs/train2017_images.pkl', 'wb') as f:
    pickle.dump(train_images, f) 
with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imgs/val2017_images.pkl', 'wb') as f:
    pickle.dump(valid_images, f) 

