# ベクトルに変換したinfovecを用いてpytorchから呼び出す形に揃えてpickle保存
import pickle 
import numpy as np
import torch
from tqdm import tqdm 
# import joblib


def load_infovec():
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/each_wn025/train2017_bertinfovec.pkl', 'rb') as f:
        train_infovec = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/each_wn025/val2017_bertinfovec.pkl', 'rb') as f:
        val_infovec = pickle.load(f) 
    return train_infovec, val_infovec

# Get image paths list
def get_imagepaths():
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/train2017_imagepaths.pkl', 'rb') as f:
        train_imagpaths = pickle.load(f)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/valid2017_imagepaths.pkl', 'rb') as f:
        val_imagpaths = pickle.load(f)
    return train_imagpaths, val_imagpaths

# Get image vector 2048dim tensor
# def get_imagevec():
#     with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/train2017_images.pkl', 'rb') as f:
#         train_imagevec =  pickle.load(f) 
#     with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/val2017_images.pkl', 'rb') as f:
#         val_imagevec =  pickle.load(f) 
#     return train_imagevec, val_imagevec

def make_dataset(imagepath, infovec):
    data_num = len(infovec[0]) * len(infovec[0][0])
    imagedata = torch.zeros(data_num*2)
    keydata = torch.zeros(data_num*2,768)
    ansdata = torch.zeros(data_num*2,768)
    labeldata = torch.zeros(data_num*2,)

    idx = 0
    for i in tqdm(range(len(imagepath)), total=len(imagepath)):
        for j, (key, caption, noise_caption) in enumerate(zip(infovec[0][i], infovec[1][i], infovec[2][i])): 
            if (not torch.equal(key, torch.zeros(768)) and not torch.isnan(key).any()) and (not torch.equal(caption, torch.zeros(768)) and not torch.isnan(caption).any()) and (not torch.equal(noise_caption, torch.zeros(768)) and not torch.isnan(noise_caption).any()):
                # 先に正解文
                if j == 0:
                    pre_caption = torch.zeros(768)
                
                # 前のキャプションと今回のキャプションが異なれば、新しいキャプション格納
                if not torch.equal(pre_caption, caption):
                    imagedata[idx] = i # 画像はインデックス入れる 
                    keydata[idx] = key
                    ansdata[idx] = caption
                    # captionを1ターン記憶
                    pre_caption =caption
                    labeldata[idx] = 1
                    idx += 1

                # 次に誤答文
                imagedata[idx] = i # 画像はインデックス入れる 
                keydata[idx] = key
                ansdata[idx] = noise_caption
                labeldata[idx] = 0
                idx += 1
            else:
                continue
    
    return imagedata[:idx], keydata[:idx], ansdata[:idx], labeldata[:idx]


def main():
    print("Get info vector (3, len(imgpaths), 50, 768) tensor")
    train_infovec, val_infovec = load_infovec()

    # print("Get image vector 2048dim tensor")
    # train_imagevec, val_imagevec = get_imagevec()

    print("Get Image Paths")
    train_imagpaths, val_imagpaths = get_imagepaths()

    print("Integrate total features into one")

    # train_dataset = make_dataset(train_imagevec, train_infovec)
    imagedata, keydata, ansdata, labeldata = make_dataset(train_imagpaths, train_infovec)
    print("保存中...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/each_wn025/imagedata.pkl', 'wb') as f:
       pickle.dump(imagedata, f, protocol=4)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/each_wn025/keydata.pkl', 'wb') as f:
       pickle.dump(keydata, f, protocol=4) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/each_wn025/ansdata.pkl', 'wb') as f:
       pickle.dump(ansdata, f, protocol=4) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/each_wn025/labeldata.pkl', 'wb') as f:
       pickle.dump(labeldata, f, protocol=4) 
    print("完了")

    # val_dataset = make_dataset(val_imagpaths, val_infovec)
    imagedata, keydata, ansdata, labeldata = make_dataset(val_imagpaths, val_infovec)
    print("保存中...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/each_wn025/imagedata.pkl', 'wb') as f:
        pickle.dump(imagedata, f, protocol=4)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/each_wn025/keydata.pkl', 'wb') as f:
        pickle.dump(keydata, f, protocol=4) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/each_wn025/ansdata.pkl', 'wb') as f:
        pickle.dump(ansdata, f, protocol=4) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/each_wn025/labeldata.pkl', 'wb') as f:
        pickle.dump(labeldata, f, protocol=4) 

    print("eachwn025のdataset完了")





if __name__ == '__main__':
    main()
