import torch 
from transformers import BertTokenizer, BertForPreTraining

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

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


def build_img2info(json_obj):
    # 画像のidをkey (key, caption, noise caption)をvalue
    img2info = {}
    idx=0
    print(type(json_obj))
    for dic in tqdm(json_obj.values(), total=len(json_obj)):
        new_noise = []
        for caption in dic['captions']:
            # 形態素解析
            morph = nltk.word_tokenize(caption)
            pos = nltk.pos_tag(morph)
            morph = ["[CLS]"] + morph + ["[SEP]"]
            final_noise_caption = morph.copy()
        
            for i, token in enumerate(pos):
                # 名詞句はランダム置換
                if token[1] == 'NN' or token[1] == 'NNS':
                    # ノイズキャプション生成
                    noise_caption = morph.copy()
                    noise_caption[i+1] = '[MASK]'
                    # print(noise_caption)
                    # BERTで予測
                    ids = tokenizer.convert_tokens_to_ids(noise_caption)
                    ids = torch.tensor(ids).reshape(1,-1)  # バッチサイズ1の形に整形

                    with torch.no_grad():
                        outputs = model(ids)
                    predictions = outputs.prediction_logits[0]

                    _, predicted_indexes = torch.topk(predictions[i+1], k=5)

                    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
                    # print(predicted_tokens)
                    for j, (v, pos) in enumerate(nltk.pos_tag(predicted_tokens)):
                        if v != morph[i+1] and v != '[UNK]'  and (pos == 'NN' or pos == 'NNS'):
                            # print(v)
                            final_noise_caption[i+1] = v
                            break      
            final_noise_caption = final_noise_caption[1:-1]   
            final_noise_caption = ' '.join(final_noise_caption)          
            # print('最終形態')    
            # print(final_noise_caption)
            # 名詞句をbert言語モデルで尤もらしい名詞句に変換
            new_noise.append(final_noise_caption)
        # 更新
        dic['bert_noise_captions'] = new_noise
        
    return json_obj
            


        



def main():
    # open file
    train_img2info = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/train_img2info.json', 'r')
    val_img2info = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/val_img2info.json', 'r')
    
    # # read as json
    train_img2info = json.load(train_img2info)
    val_img2info = json.load(val_img2info)


    # 画像のidをkey {key, captions, noise_captions}をvalueにした辞書
    train_img2infobert = build_img2info(train_img2info)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/train2017_img2infobert.pkl', 'wb') as f:
        pickle.dump(train_img2infobert, f) 
        
    val_img2infobert = build_img2info(val_img2info)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/val2017_img2infobert.pkl', 'wb') as f:
        pickle.dump(val_img2infobert, f) 





if __name__ == '__main__':
    main()