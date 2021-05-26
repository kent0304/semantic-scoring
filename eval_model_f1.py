import csv
import pickle 
import yaml
import argparse
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

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



# GPU対応
device = torch.device('cuda:1')


def load_data(device, ver):
    # 画像のキャッシュをロード
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imgs/val2017_images.pkl', 'rb') as f:
        val_imagedata = pickle.load(f) 
    print('val画像の数：', len(val_imagedata))

    valid_dataset = MyDataset(dirname=ver, p='valid', images=val_imagedata)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

    return valid_dataset, valid_loader


def load_model(ver):
    # 学習済みモデル読み込み
    model = Model()
    model.load_state_dict(torch.load('model/bert/{}/0220model_epoch5.pth'.format(ver), map_location=device))
    return model


def eval(model, test_loader, device):
    model = model.to(device)
    model.eval()
    pred_list = []
    label_list = []
    for i, (image, ans, label) in enumerate(tqdm(test_loader, total=len(test_loader))):
        with torch.no_grad():
            image = image.to(device)
            ans = ans.to(device)
            label = label.to(device)
            pred = model(image, ans)
        
        pred = torch.squeeze(pred)
        
        pred = pred.to('cpu').detach().numpy().copy()
        label = label.to('cpu').detach().numpy().copy()
        pred = np.where(pred >0, 1, 0)

        pred = pred.tolist()
        label = label.tolist()

        pred_list += pred 
        label_list += label

    new_pred_list = []
    for e in pred_list:
        if e == 0:
            new_pred_list.append(1)
        else:
            new_pred_list.append(0)

    new_label_list = []
    for e in label_list:
        if e == 0:
            new_label_list.append(1)
        else:
            new_label_list.append(0)
    return new_pred_list, new_label_list






def main(args):
    output = []

    ver = args.ver
    print("これは{}の疑似生成データを用いて訓練します".format(ver))
    test_dataset, test_loader = load_data(device, ver)
    model = load_model(ver)
    print("データのロード完了")

    pred_list, label_list = eval(model, test_loader, device)

    
    # # ランダム
    # test_dataset, test_loader = load_random_testdata()
    # output.append('テストデータの数' + str(len(test_dataset)))
    # model = load_random_model()
    # output.append('ランダムに疑似生成したデータで訓練したモデル')
    # pred_list, label_list = eval(model, test_loader,  device)
    # cm = confusion_matrix(label_list, pred_list)
    # print("cm:", cm)
    # precision = precision_score(label_list, pred_list)
    # output.append("Precision:"+str(precision))
    # recall = recall_score(label_list, pred_list)
    # output.append("Recall:"+str(recall))
    # f1 = f1_score(label_list, pred_list)
    # output.append("F1score:"+str(f1))

    # output.append('\n')
    # bert

    output.append('テストデータの数' + str(len(test_dataset)))

    cm = confusion_matrix(label_list, pred_list)
    print("cm:", cm)
    precision = precision_score(label_list, pred_list)
    output.append("Precision:"+str(precision))
    recall = recall_score(label_list, pred_list)
    output.append("Recall:"+str(recall))
    f1 = f1_score(label_list, pred_list)
    output.append("F1score:"+str(f1))

    with open('result/eval/bert_wn05_eval.txt', 'w') as f:
        f.write('\n'.join(output))




    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver',
                        help='Select pseudo data version', required=True)
    args = parser.parse_args()
    main(args)