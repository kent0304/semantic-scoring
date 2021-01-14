# 疑似誤り文（ノイズ）を生成し、img2infoの辞書で保管。
import json 
import random
import pickle
import nltk 
# word_tokenize
nltk.download('punkt')
# pos_tag
nltk.download('averaged_perceptron_tagger')

from tqdm import tqdm


def load_corpus(json_obj):
    # 全ての文を管理するコーパス
    corpus = []
    # 名詞を管理するリスト
    nn = []
    for dic in tqdm(json_obj["annotations"], total=len(json_obj["annotations"])):
        # コーパス構築
        caption = dic['caption']
        corpus.append(caption)  
        # 形態素解析
        morph = nltk.word_tokenize(dic["caption"])
        pos = nltk.pos_tag(morph)
        for token in pos:
            # 名詞句はリストで収集
            if token[1] == 'NN' or token[1] == 'NNS':
                nn.append(token[0])
    
    return corpus, nn


def build_img2info(json_obj, corpus, nn):
    # 画像のidをkey (key, caption, noise caption)をvalue
    img2info = {}
    idx=0
    for dic in tqdm(json_obj["annotations"], total=len(json_obj["annotations"])):
        # keyのリスト 
        k = []
        # 形態素解析
        morph = nltk.word_tokenize(dic["caption"])
        pos = nltk.pos_tag(morph)
        # ノイズキャプション生成
        noise_caption = []
        for token in pos:
            # 名詞句はランダム置換
            if token[1] == 'NN' or token[1] == 'NNS':
                neg_nn = random.sample(nn, 1)[0]
                while neg_nn == token[0]:
                    neg_nn = random.sample(nn, 1)[0] # random.choiceはリストで出力するのでインデックス付与
                noise_caption.append(neg_nn)
            else:
                noise_caption.append(token[0])
                
            # 動詞句はそのキャプションに対応するkeyにする
            if token[1][:2] == 'VB':
                k.append(token[0])
        # keyの重複削除
        k = set(k)
        # ノイズキャプションはstringに
        noise_caption = ' '.join(noise_caption)
        # print("キャプション：", dic["caption"])
        # print("ノイズキャプション:", noise_caption)
        # print("語句:", k)

            

        if dic["image_id"] in img2info:   
            # keyのタプル追加
            img2info[dic["image_id"]]["key"].append(k)
            # キャプション追加
            img2info[dic["image_id"]]["captions"].append(dic["caption"])
            # ノイズキャプション追加
            img2info[dic["image_id"]]["noise_captions"].append(noise_caption)
        else: # 初めて登場する画像
            # キャプションのリストを作って格納
            captions = [dic["caption"]]
            # ノイズキャプションもリスト
            noise_captions = [noise_caption]
            # keyもリスト
            keys = [k]
            # infoリストにはその画像のkey（配列）、captions（配列）、noise_captions（配列）が入っている
            info = {"key": keys, "captions": captions, "noise_captions": noise_captions}
            img2info[dic["image_id"]] = info
        
    return img2info
            


        



def main():
    # open file
    json_train2017 = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/annotations/captions_train2017.json', 'r')
    json_val2017 = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/annotations/captions_val2017.json', 'r')
    
    # read as json
    train2017 = json.load(json_train2017)
    val2017 = json.load(json_val2017)

    train_corpus, train_nn = load_corpus(train2017)
    val_corpus, val_nn = load_corpus(val2017)
    # 文がリストで管理されたコーパス
    corpus = train_corpus + val_corpus
    c = '\n'.join(corpus)
    # コーパスファイル書き込み
    # with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/annotations/corpus.txt', 'w') as f:
    #     f.write(c)
    # 名詞のリストはnnで管理
    nn = train_nn + val_nn
    print("nnのリストの要素数は", len(nn))
    nn = set(nn)
    print("nnのsetの要素数は", len(nn))

    # 画像のidをkey {key, captions, noise_captions}をvalueにした辞書
    train_img2info = build_img2info(train2017, corpus, nn)
    val_img2info = build_img2info(val2017, corpus, nn)
    # with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_img2info.pkl', 'wb') as f:
    #     pickle.dump(train_img2info, f) 
    # with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_img2info.pkl', 'wb') as f:
    #     pickle.dump(val_img2info, f) 





if __name__ == '__main__':
    main()