import pickle
import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from dataset import MyDataset
from model import Model
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from sentence_transformers import SentenceTransformer


device = torch.device('cuda:1')

# ver = os.getenv("PD")
# ver = args.ver


def load_data(device, ver):
    # 画像のキャッシュをロード
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imgs/train2017_images.pkl', 'rb') as f:
        train_imagedata = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/imgs/val2017_images.pkl', 'rb') as f:
        val_imagedata = pickle.load(f) 
    print('train画像の数：', len(train_imagedata))
    print('val画像の数：', len(val_imagedata))

    train_dataset = MyDataset(dirname=ver, p='train', images=train_imagedata)
    valid_dataset = MyDataset(dirname=ver, p='valid', images=val_imagedata)

    # print(valid_dataset[0][0])
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=True)



    return train_loader, valid_loader

# モデル評価
def eval_net(model, data_loader, loss, device):
    model.eval()
    model = model.to(device)
    outputs = []
    accs = []
    for i, (image, ans, label) in enumerate(data_loader):
        with torch.no_grad():
            # GPU setting
            image = image.to(device)
            # key = key.to(device)
            ans = ans.to(device)
            label = label.to(device)
            pred = model(image, ans)
    
        output = loss(torch.squeeze(pred), label)
        outputs.append(output.item())

        pred = torch.squeeze(pred)
        pred = pred.to('cpu').detach().numpy().copy()
        label = label.to('cpu').detach().numpy().copy()
        pred = np.where(pred > 0, 1, 0)
        acc = (pred == label).sum() / len(label)
        accs.append(acc)

    return sum(outputs) / i , sum(accs)/i
    

# モデルの学習
def train_net(model, train_loader, valid_loader, loss, n_iter, device, ver):
    train_losses = []
    valid_losses = []
    train_accs = []
    val_accs = []
    optimizer = optim.Adam(model.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        model = model.to(device)
        # ネットワーク訓練モード
        model.train()
        accs = []
        for i, (image, ans, label) in enumerate(tqdm(train_loader, total=len(train_loader))):        
            # GPU setting
            image = image.to(device)
            # key = key.to(device)
            ans = ans.to(device)
            label = label.to(device)
            # print(type(label))
            # model
            pred = model(image, ans)
            pred = torch.squeeze(pred)
            # print(type(pred))
            # print(type(label))
            # print(pred.dtype)
            # print(label.dtype)
            # print(torch.squeeze(pred))
            # print(torch.tensor(label,dtype=torch.float))
            output = loss(pred, label)
            # print(output)

            pred = torch.squeeze(pred)
            pred = pred.to('cpu').detach().numpy().copy()
            label = label.to('cpu').detach().numpy().copy()
            pred = np.where(pred > 0, 1, 0)
            acc = (pred == label).sum() / len(label)
            accs.append(acc)
            
   
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            running_loss += output.item()
    
        # 訓練用データでのloss値
        train_losses.append(running_loss / i)
        # 検証用データでのloss値
        pred_valid, val_acc =  eval_net(model, valid_loader, loss, device)
        train_accs.append(sum(accs)/i)
        val_accs.append(val_acc)
        valid_losses.append(pred_valid)
        print('epoch:' +  str(epoch+1), 'train loss:'+ str(train_losses[-1]), 'valid loss:' + str(valid_losses[-1]), 'train acc:' + str(train_accs[-1]),  'val acc:' + str(val_accs[-1]), flush=True)

        # 学習モデル保存
        if (epoch+1)%1==0:
            # 学習させたモデルの保存パス
            model_path =  'model/bert/{}/0220model_epoch{}.pth'.format(ver, str(epoch+1))
            # モデル保存
            torch.save(model.to('cpu').state_dict(), model_path)
            # loss保存
            with open('model/bert/{}/0220train_losses.pkl'.format(ver), 'wb') as f:
                pickle.dump(train_losses, f) 
            with open('model/bert/{}/0220valid_losses.pkl'.format(ver), 'wb') as f:
                pickle.dump(valid_losses, f) 
            # グラフ描画
            my_plot(train_losses, valid_losses, ver)
    return train_losses, valid_losses

def my_plot(train_losses, valid_losses, ver):
    # グラフの描画先の準備
    fig = plt.figure()
    # 画像描画
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    #グラフタイトル
    plt.title('Triplet Margin Loss')
    #グラフの軸
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #グラフの凡例
    plt.legend()
    # グラフ画像保存
    fig.savefig("result/lossimag/0220{}_loss.png".format(ver))

def select_epoch(valid_losses):
    min_loss = min(valid_losses)
    return valid_losses.index(min_loss) + 1



def main(args):
    ver = args.ver
    print("これは{}の疑似生成データを用いて訓練します".format(ver))
    train_loader, valid_loader = load_data(device, ver)
    loss = nn.BCEWithLogitsLoss()
    model = Model()
    print("データのロード完了")

    print("訓練開始")
    train_losses, valid_losses = train_net(model=model, 
                                           train_loader=train_loader, 
                                           valid_loader=valid_loader,  
                                           loss=loss, 
                                           n_iter=100, 
                                           device=device,
                                           ver=ver)
    best_epoch = select_epoch(valid_losses)
    print(f'{ver}の結果です')
    print(f'{best_epoch}epochのモデルが最もvalid lossが下がった。')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver',
                        help='Select pseudo data version', required=True)
    args = parser.parse_args()
    main(args)

