# 辞書で管理しているimg2infoをもとにinfoをベクトルに変換
import os
import json 
import random 
import pickle 
from gensim.models.wrappers import FastText
import nltk
nltk.download('punkt')
import numpy as np
import torch
from torch import nn
from PIL import Image
from gensim.models import KeyedVectors
from gensim.models import FastText
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

# Get image vector 2048dim tensor
def get_imagevec():
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/train2017_images.pkl', 'rb') as f:
        train_imagevec =  pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/val2017_images.pkl', 'rb') as f:
        val_imagevec =  pickle.load(f) 
    return train_imagevec, val_imagevec

# def load_fasttext():
#     # 1回目（modelが保存されていない時）
#     # gensimからfastTextの学習済み学習済み単語ベクトル表現を利用
#     # model = KeyedVectors.load_word2vec_format('/mnt/LSTA5/data/tanaka/fasttext/cc.en.300.vec.gz', binary=False)
#     # with open('/mnt/LSTA5/data/tanaka/fasttext/gensim-vecs.cc.en.300.vec.pkl', mode='wb') as fp:
#     #     pickle.dump(model, fp)
#     # model = FastText.load_fasttext_format('/mnt/LSTA5/data/tanaka/fasttext/cc.en.300.bin.gz')
#     # with open('/mnt/LSTA5/data/tanaka/fasttext/gensim-vecs.cc.en.300.bin.pkl', mode='wb') as fp:
#     #     pickle.dump(model, fp)

#     # 2回目以降（modelが保存されている時）
#     with open('/mnt/LSTA5/data/tanaka/fasttext/gensim-vecs.cc.en.300.bin.pkl', mode='rb') as fp:
#         model = pickle.load(fp)
#     return model

def info2vec(imgpaths, img2info):
    sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    info_vec = torch.zeros(3, len(imgpaths), 10, 768)
    for i, img in enumerate(tqdm(imgpaths, total=len(imgpaths))):
        id = int(img[-16:-4])
        for j, (key, caption, noise_caption) in enumerate(zip(img2info[str(id)]['key'], img2info[str(id)]['captions'], img2info[str(id)]['bert_wn05_noise_captions'])):
            key_vec = text2vec(' '.join(key), sbert_model)
            cap_vec = text2vec(caption, sbert_model)
            noisecap_vec = text2vec(noise_caption, sbert_model)

            info_vec[0][i][j] += key_vec
            info_vec[1][i][j] += cap_vec
            info_vec[2][i][j] += noisecap_vec
    return info_vec

# def fasttext(text, fasttext_model):
#     morph = nltk.word_tokenize(text)
#     vec = torch.zeros(300,)
#     cnt = 0
#     for token in morph:
#         vec += fasttext_model.wv[token]
#         cnt += 1
#     vec = vec/cnt
#     return vec

def text2vec(text, sbert_model):
    return sbert_model.encode(text)




def main():
    # image id 2 infomation dictionary
    print("Get img2info dictionary...")
    train_img2info, val_img2info = load_img2info()

    # Get image paths list
    print("Get image paths list")
    train_imagpaths, val_imagpaths = get_imagepaths()

    # Get image vector 2048dim tensor
    # train_imagevec, val_imagevec = get_imagevec()

    # # Load fasttext
    # print("Load fasttext...")
    # fasttext_model = load_fasttext()

    # Get info vector (3, len(imgpaths), 10, 300) tensor
    print("Get info vector (3, len(imgpaths), 10, 300) tensor")
    train_infovec = info2vec(train_imagpaths, train_img2info)
    print("picleで保存")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/train2017_bertinfovec.pkl', 'wb') as f:
        pickle.dump(train_infovec, f) 
    val_infovec = info2vec(val_imagpaths, val_img2info)
    print("picleで保存")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/val2017_bertinfovec.pkl', 'wb') as f:
        pickle.dump(val_infovec, f) 
    print("完了!!")



if __name__ == '__main__':
    main()