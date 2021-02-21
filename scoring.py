import csv
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

from sentence_transformers import SentenceTransformer

# GPU対応
device = torch.device('cuda:1')

# def load_random():
#     # 学習済みモデル読み込み
#     model = Model()
#     model.load_state_dict(torch.load('model/random/model_epoch200.pth', map_location=device))
#     return model

def load_bert_wn025():
    # 学習済みモデル読み込み
    model = Model()
    model.load_state_dict(torch.load('model/bert/wn025/model_epoch7.pth', map_location=device))
    return model

def load_bert_wn05():
    # 学習済みモデル読み込み
    model = Model()
    model.load_state_dict(torch.load('model/bert/wn05/model_epoch7.pth', map_location=device))
    return model

def load_bert_wn075():
    # 学習済みモデル読み込み
    model = Model()
    model.load_state_dict(torch.load('model/bert/wn075/model_epoch9.pth', map_location=device))
    return model


def load_data():
    with open('src/input_real.yml') as f:
        obj = yaml.safe_load(f)
    return obj

def embedding(obj):
    data_num = len(obj)
    image_paths = []
    keyvec = torch.zeros(data_num, 768)
    ansvec = torch.zeros(data_num, 768)
    # text 
    sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    for i, sample in enumerate(obj):
        # image
        image_paths.append(sample['IMAGE'])
        # text 
        keyvec[i] = text2vec(sample['KEY'], sbert_model)
        ansvec[i] = text2vec(sample['ANSWER'], sbert_model)
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


def text2vec(text, sbert_model):
    return torch.tensor(sbert_model.encode(text))



def eval(model, imagevec, keyvec, ansvec):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        imagevec = imagevec.to(device)
        keyvec = keyvec.to(device)
        ansvec = ansvec.to(device)
        pred = model(imagevec, keyvec, ansvec)

    pred = torch.squeeze(pred)
    # print(pred)
    m = nn.Sigmoid()
    # print(m(pred))
    pred = pred.to('cpu').detach().numpy().copy()
    pred = np.where(pred>0, 1, 0)

    return pred



def main():
    # 学習済みモデル
    # random_model = load_random()
    bert_wn025_model = load_bert_wn025()
    bert_wn05model = load_bert_wn05()
    bert_wn075_model = load_bert_wn075()

    obj = load_data()
    imagevec, keyvec, ansvec = embedding(obj)
    # 推論
    # random_result = eval(random_model, imagevec, keyvec, ansvec)
    bert_wn025_score = eval(bert_wn025_model, imagevec, keyvec, ansvec)
    bert_wn05_score = eval(bert_wn05model, imagevec, keyvec, ansvec)
    bert_wn075_score = eval(bert_wn075_model, imagevec, keyvec, ansvec)

    learners = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "B1", "B2","B3", "B4", "B5", "B6", "B7", "B8"]
    questions = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]

    result_wn025 = [questions]
    result_wn05 = [questions]
    result_wn075 = [questions]

    for i, learner in enumerate(learners):
        result_wn025.append(bert_wn025_score[10*i:10*(i+1)])
        result_wn05.append(bert_wn05_score[10*i:10*(i+1)])
        result_wn075.append(bert_wn075_score[10*i:10*(i+1)])


    with open("result/scoring/scoring_wn025.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(result_wn025)
    with open("result/scoring/scoring_wn05.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(result_wn05)
    with open("result/scoring/scoring_wn075.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(result_wn075)

    return 


if __name__ == '__main__':
    main()