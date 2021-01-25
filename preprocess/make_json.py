import pickle
import json

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def main():

    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_img2info.pkl', 'rb') as f:
        train_img2info = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_img2info.pkl', 'rb') as f:
        val_img2info = pickle.load(f) 

    
    # 辞書をjsonとして書き込み
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/train_img2info.json', 'w') as f:
        json.dump(train_img2info, f, indent=4, default=set_default)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/val_img2info.json', 'w') as f:
        json.dump(val_img2info, f, indent=4, default=set_default)

    # # 辞書をjsonとして書き込み
    # with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/train_img2info.json', 'r') as f:
    #     train_img2info = json.load(f)
    # with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/val_img2info.json', 'r') as f:
    #     val_img2info = json.load(f)


if __name__ == '__main__':
    main()