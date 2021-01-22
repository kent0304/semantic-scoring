import pickle

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from dataset import MyDataset
from model import Model
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from model import Model
from dataset import MyDataset

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



# GPU対応
device = torch.device('cuda:1')


def load_model():
    # 学習済みモデル読み込み
    model = Model()
    model.load_state_dict(torch.load('model/model0112_epoch200.pth', map_location=device))
    return model

def load_testdata():
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train_semantic_scoring/imagedata.pkl', 'rb') as f:
        val_imagedata = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train_semantic_scoring/keydata.pkl', 'rb') as f:
        val_keydata = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train_semantic_scoring/ansdata.pkl', 'rb') as f:
        val_ansdata = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train_semantic_scoring/labeldata.pkl', 'rb') as f:
        val_labeldata = pickle.load(f) 
    valid_dataset = MyDataset(val_imagedata, val_keydata, val_ansdata, val_labeldata)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    return valid_dataset, valid_loader

def eval(model, test_loader, device):
    model = model.to(device)
    model.eval()
    pred_list = []
    label_list = []
    for i, (image, key, ans, label) in enumerate(tqdm(test_loader, total=len(test_loader))):
        with torch.no_grad():
            image = image.to(device)
            key = key.to(device)
            ans = ans.to(device)
            label = label.to(device)
            pred = model(image, key, ans)
        
        pred = torch.squeeze(pred)
        
        pred = pred.to('cpu').detach().numpy().copy()
        label = label.to('cpu').detach().numpy().copy()
        pred = np.where(pred >0, 1, 0)

        pred = pred.tolist()
        label = label.tolist()

        pred_list += pred 
        label_list += label
    return pred_list, label_list






def main():
    model = load_model()
    test_dataset, test_loader = load_testdata()
    
    print(len(test_dataset))
    pred_list, label_list = eval(model, test_loader,  device)
    cm = confusion_matrix(label_list, pred_list)
    print("cm:", cm)
    precision = precision_score(label_list, pred_list)
    print("Precision:", precision)
    recall = recall_score(label_list, pred_list)
    print("Recall:", recall)
    f1 = f1_score(label_list, pred_list)
    print("F1score:", f1)




    return 


if __name__ == '__main__':
    main()