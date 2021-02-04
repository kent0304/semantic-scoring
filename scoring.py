import pickle 
import yaml
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from tqdm import tqdm

from gensim.models.wrappers import FastText
import nltk
nltk.download('punkt')

from dataset import MyDataset
from model import Model
from PIL import Image
from torchvision import transforms, models
from matplotlib import pyplot as plt
plt.switch_backend('agg')

# GPU対応
device = torch.device('cuda:1')

def load_model():
    # 学習済みモデル読み込み
    model = Model()
    model.load_state_dict(torch.load('model/bert/model_adam_epoch500.pth', map_location=device))
    return model

def load_data():
    with open('src/input_real.yml') as f:
        obj = yaml.safe_load(f)
    return obj

def embedding(obj):
    data_num = len(obj)
    image_paths = []
    keyvec = torch.zeros(data_num, 300)
    ansvec = torch.zeros(data_num, 300)
    # text 
    fasttext_model = load_fasttext()
    for i, sample in enumerate(obj):
        # image
        image_paths.append(sample['IMAGE'])
        # text 
        keyvec[i] = text2vec(sample['KEY'], fasttext_model)
        ansvec[i] = text2vec(sample['ANSWER'], fasttext_model)
    # image
    imagevec = make_imagedata(image_paths, data_num)

    return imagevec, keyvec, ansvec
    
        
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


def make_imagedata(image_paths, data_num):
    # resnet呼び出し
    image_net = models.resnet50(pretrained=True)
    image_net.fc = nn.Identity()
    image_net.eval()
    image_net = image_net.to(device)
    image_vec = torch.zeros((data_num, 2048))
    batch_size = 1

    with torch.no_grad():
        for i in range(0, data_num, batch_size):
            mini_image_paths = [image_paths[j] for j in range(i, data_num)[:batch_size]]
            images = image2vec(image_net, mini_image_paths)
            image_vec[i:i + batch_size] = images
    return image_vec

def load_fasttext():
    with open('/mnt/LSTA5/data/tanaka/fasttext/gensim-vecs.cc.en.300.bin.pkl', mode='rb') as fp:
        model = pickle.load(fp)
    return model

def text2vec(text, fasttext_model):
    morph = nltk.word_tokenize(text)
    vec = torch.zeros(300,)
    cnt = 0
    for token in morph:
        vec += fasttext_model.wv[token]
        cnt += 1
    vec = vec/cnt
    return vec



def eval(model, imagevec, keyvec, ansvec):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        imagevec = imagevec.to(device)
        keyvec = keyvec.to(device)
        ansvec = ansvec.to(device)
        pred = model(imagevec, keyvec, ansvec)

    pred = torch.squeeze(pred)
    print(pred)
    m = nn.Sigmoid()
    print(m(pred))
    pred = pred.to('cpu').detach().numpy().copy()
    pred = np.where(pred>0, 1, 0)

    return pred



def main():
    model = load_model()
    obj = load_data()
    imagevec, keyvec, ansvec = embedding(obj)
    result = eval(model, imagevec, keyvec, ansvec)
    # for score in result:
    #     print(score)
    print(result)

    return 


if __name__ == '__main__':
    main()