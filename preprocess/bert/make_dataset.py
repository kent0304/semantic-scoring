# ベクトルに変換したinfovecを用いてpytorchから呼び出す形に揃えてpickle保存
import pickle 
import numpy as np
import torch
from tqdm import tqdm 


def load_infovec():
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/train2017_bertinfovec.pkl', 'rb') as f:
        train_infovec = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/val2017_bertinfovec.pkl', 'rb') as f:
        val_infovec = pickle.load(f) 
    return train_infovec, val_infovec

# Get image vector 2048dim tensor
def get_imagevec():
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/train2017_images.pkl', 'rb') as f:
        train_imagevec =  pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/images/val2017_images.pkl', 'rb') as f:
        val_imagevec =  pickle.load(f) 
    return train_imagevec, val_imagevec

def make_dataset(imagevec, infovec):
    data_num = len(infovec[0]) * len(infovec[0][0])
    imagedata = torch.zeros(data_num*2,2048)
    keydata = torch.zeros(data_num*2,768)
    ansdata = torch.zeros(data_num*2,768)
    labeldata = torch.zeros(data_num*2,)

    idx = 0
    for i, image in enumerate(tqdm(imagevec, total=len(imagevec))):
        for key, caption, noise_caption in zip(infovec[0][i], infovec[1][i], infovec[2][i]): 
            if (not torch.equal(key, torch.zeros(768)) and not torch.isnan(key).any()) and (not torch.equal(caption, torch.zeros(768)) and not torch.isnan(caption).any()) and (not torch.equal(noise_caption, torch.zeros(768)) and not torch.isnan(noise_caption).any()):
                # 先に正解文
                imagedata[idx] = image 
                keydata[idx] = key
                ansdata[idx] = caption
                labeldata[idx] = 1
                # 次に誤答文
                idx += 1
                imagedata[idx] = image 
                keydata[idx] = key
                ansdata[idx] = noise_caption
                labeldata[idx] = 0

                idx += 1
            else:
                continue
    
    return imagedata[:idx], keydata[:idx], ansdata[:idx], labeldata[:idx]


def main():
    # Get info vector (3, len(imgpaths), 10, 768) tensor
    print("Get info vector (3, len(imgpaths), 10, 768) tensor")
    train_infovec, val_infovec = load_infovec()

    # Get image vector 2048dim tensor
    print("Get image vector 2048dim tensor")
    train_imagevec, val_imagevec = get_imagevec()

    # Integrate total features into one
    print("Integrate total features into one")
    # train_dataset = make_dataset(train_imagevec, train_infovec)
    imagedata, keydata, ansdata, labeldata = make_dataset(train_imagevec, train_infovec)
    print("保存中...")

    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/imagedata.pkl', 'wb') as f:
        pickle.dump(imagedata, f, protocol=4)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/keydata.pkl', 'wb') as f:
        pickle.dump(keydata, f, protocol=4) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/ansdata.pkl', 'wb') as f:
        pickle.dump(ansdata, f, protocol=4) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/labeldata.pkl', 'wb') as f:
        pickle.dump(labeldata, f, protocol=4) 
    print("完了")

    # val_dataset = make_dataset(val_imagevec, val_infovec)
    imagedata, keydata, ansdata, labeldata = make_dataset(val_imagevec, val_infovec)
    print("保存中...")

    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/imagedata.pkl', 'wb') as f:
        pickle.dump(imagedata, f, protocol=4)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/keydata.pkl', 'wb') as f:
        pickle.dump(keydata, f, protocol=4) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/ansdata.pkl', 'wb') as f:
        pickle.dump(ansdata, f, protocol=4) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/labeldata.pkl', 'wb') as f:
        pickle.dump(labeldata, f, protocol=4) 
    print("完了")





if __name__ == '__main__':
    main()
