# 辞書で管理しているimg2infoをもとにinfoをベクトルに変換
import os
import json 
import random 
import pickle 

import nltk
nltk.download('punkt')
import numpy as np
import torch
from torch import nn
from PIL import Image
from gensim.models import KeyedVectors

from torchvision import transforms, models

from tqdm import tqdm 

from sentence_transformers import SentenceTransformer

def load_img2info():
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/train2017_img2infobert.pkl', 'rb') as f:
        train_img2info = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/val2017_img2infobert.pkl', 'rb') as f:
        val_img2info = pickle.load(f)
    return train_img2info, val_img2info

# Get image paths list
def get_imagepaths():
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/train2017_imagepaths.pkl', 'rb') as f:
        train_imagpaths = pickle.load(f)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/valid2017_imagepaths.pkl', 'rb') as f:
        val_imagpaths = pickle.load(f)
    return train_imagpaths, val_imagpaths


def info2vec(imgpaths, img2info):
    sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    info_vec = torch.zeros(3, len(imgpaths), 10*5, 768)
    for i, img in enumerate(tqdm(imgpaths, total=len(imgpaths))):
        id = int(img[-16:-4])
        for j, (key, caption, noise_caption) in enumerate(zip(img2info[str(id)]['key'], img2info[str(id)]['captions'], img2info[str(id)]['berteach_wn05_noise_captions'])):
            key_vec = text2vec(' '.join(key), sbert_model)
            cap_vec = text2vec(caption, sbert_model)
            for noise in noise_caption:
                noisecap_vec = text2vec(noise, sbert_model)

                info_vec[0][i][j] += key_vec
                info_vec[1][i][j] += cap_vec
                info_vec[2][i][j] += noisecap_vec
    return info_vec


def text2vec(text, sbert_model):
    return sbert_model.encode(text)




def main():
    # image id 2 infomation dictionary
    print("Get img2info dictionary...")
    train_img2info, val_img2info = load_img2info()

    # Get image paths list
    print("Get image paths list")
    train_imagpaths, val_imagpaths = get_imagepaths()

  

    # Get info vector (3, len(imgpaths), 50, 768) tensor
    print("Get info vector (3, len(imgpaths), 50, 768) tensor")
    train_infovec = info2vec(train_imagpaths, train_img2info)
    print("picleで保存")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/each_wn05/train2017_bertinfovec.pkl', 'wb') as f:
        pickle.dump(train_infovec, f, protocol=4) 
    val_infovec = info2vec(val_imagpaths, val_img2info)
    print("picleで保存")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/each_wn05/val2017_bertinfovec.pkl', 'wb') as f:
        pickle.dump(val_infovec, f, protocol=4) 
    print("完了!!")



if __name__ == '__main__':
    main()
