# 正誤判定モデル
# 入力は画像(2048)、語句(300)、正答文(300)、誤答文(300)
# 出力はラベル(0, 1)
import torch
from torch import nn, optim
import torch.nn.functional as F
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
from PIL import Image

import matplotlib.pyplot as plt




class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_fc1 = nn.Linear(2048, 1536)
        self.image_fc2 = nn.Linear(1536, 1024)
        self.image_fc3 = nn.Linear(1024, 768)

        # self.key_fc1 = nn.Linear(768, 768)
        # self.key_fc2 = nn.Linear(768, 768)
        # self.key_fc3 = nn.Linear(446, 512)

        self.ans_fc1 = nn.Linear(768, 768)
        self.ans_fc2 = nn.Linear(768, 768)
        # self.ans_fc3 = nn.Linear(600, 512)

        self.total_fc1 = nn.Linear(1536, 1000)
        self.total_fc2 = nn.Linear(1000, 500)
        self.total_fc3 = nn.Linear(500, 1)



    def forward(self, image, ans):
        image = F.relu(self.image_fc1(image))
        image = F.relu(self.image_fc2(image))
        image = self.image_fc3(image)
        # print(list(image))

        # key = F.relu(self.key_fc1(key))
        # key = self.key_fc2(key)

        ans = F.relu(self.ans_fc1(ans))
        ans = self.ans_fc2(ans)
        # print(list(ans))

        # input_feature = torch.cat([torch.squeeze(image), key, ans], axis=1)
        input_feature = torch.cat([torch.squeeze(image), ans], axis=1)
        # print(torch.squeeze(image))
        # print(torch.squeeze(image).size())

        # print(key)
        # print(key.size())
        # print(ans)
        # print(ans.size())


        output = F.relu(self.total_fc1(input_feature))
        # print(self.total_fc1(input_feature)[0].cpu().data.numpy())

        #  # グラフの描画先の準備
        # fig = plt.figure()
        # # 画像描画
        # plt.plot(output[0].cpu().data.numpy())

        # # グラフ画像保存
        # fig.savefig("result/test.png")
        

        # print(output.size())
        output = F.relu(self.total_fc2(output))
        output = self.total_fc3(output)

        return output
