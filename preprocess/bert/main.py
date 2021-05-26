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
# wordnet
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def build_img2info(json_obj, sim_value):
    # 画像のidをkey (key, caption, noise caption)をvalue
    img2info = {}
    idx=0
    for dic in tqdm(json_obj.values(), total=len(json_obj)):
        new_noise = []
        for caption in dic['captions']:
            noise_captions = []
            # 形態素解析
            morph = nltk.word_tokenize(caption.lower())
            pos = nltk.pos_tag(morph)
            morph = ["[CLS]"] + morph + ["[SEP]"]
            # print('トークンに分割-------------------------------------------------')
            # print('形態素解析結果:', morph)
            final_noise_caption = morph.copy()
        
            for i, token in enumerate(pos):
                # 名詞句はランダム置換
                if token[1] == 'NN' or token[1] == 'NNS': # 名詞 or 名詞（複数形） 
                    # ノイズキャプション生成
                    noise_caption = morph.copy()
                    noise_caption[i+1] = '[MASK]'
                    # print("これがnoise caption")
                    # print(noise_caption)
                    # BERTで予測
                    
                    ids = tokenizer.convert_tokens_to_ids(noise_caption)
                    ids = torch.tensor(ids).reshape(1,-1)  # バッチサイズ1の形に整形

                    with torch.no_grad():
                        outputs = model(ids)
                    predictions = outputs.prediction_logits[0]

                    _, predicted_indexes = torch.topk(predictions[i+1], k=5)

                    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
                    # print('今回対象のトークン:', token[0])
                    # print('BERTで予測した置き換えられる語:', predicted_tokens)
                    for j, word in enumerate(predicted_tokens):
                        # if word != morph[i+1] and word != '[UNK]':
                        #     noise_caption[i+1] = word
                        #     final_noise_caption = noise_caption[1:-1]
                        #     final_noise_caption = ' '.join(final_noise_caption)
                        #     noise_captions.append(final_noise_caption)
                        #     break

                        if word != morph[i+1] and word != '[UNK]': # 置き換える語と同じでない
                            try:
                                w1 = wn.synset(token[0] + '.n.01')
                            except nltk.corpus.reader.wordnet.WordNetError: # wordnetに存在しない場合
                                # print('そもそもwordnetに存在しないので比較できない', token[0])
                                # print(word, 'で決定')
                                # final_noise_caption[i+1] = word
                                noise_caption[i+1] = word
                                break
                            # 置換対象との類似度を比較
                            try: 
                                w2 = wn.synset(word + '.n.01')
                                if w1.wup_similarity(w2) < sim_value: # 類似度閾値未満なら採用
                                    # print(word, 'で決定')
                                    # final_noise_caption[i+1] = word
                                    # print("適用前")
                                    # print(noise_caption)
                                    noise_caption[i+1] = word
                                    final_noise_caption = noise_caption[1:-1]
                                    final_noise_caption = ' '.join(final_noise_caption)
                                    noise_captions.append(final_noise_caption)
                                    # print(noise_captions)
                                    # print("適用後")
                                    # print(noise_caption)
                                    break
                                else: # 類似度閾値以上なら次の候補へ
                                    continue
                            except nltk.corpus.reader.wordnet.WordNetError: 
                                continue 
  

            # final_noise_caption = final_noise_caption[1:-1]   
            # final_noise_caption = ' '.join(final_noise_caption)  
            # final_noise_captions.append()        
            # print('最終形態')    
            # print(final_noise_caption)
            # 名詞句をbert言語モデルで尤もらしい名詞句に変換
            new_noise.append(noise_captions)
        # 更新
        dic['berteach_noise_captions'] = new_noise

        
    return json_obj


# def check(word):
#     """
#     入れ替えれる単語が好ましくなければFalse
#     """
#     spaces = ['\u200b', '\u200e', '\u202a', '\u2009', '\u2028', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u3000', '\x10', '\x7f', '\x9d', '\xad',
#             '\x97', '\x9c', '\x8b', '\x81', '\x80', '\x8c', '\x85', '\x92', '\x88', '\x8d', '\x80', '\x8e', '\x9a', '\x94', '\xa0', 
#             '\x8f', '\x82', '\x8a', '\x93', '\x90', '\x83', '\x96', '\x9b', '\x9e', '\x99', '\x87', '\x84', '\x9f',
#             ]
#     puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
#             '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',
#             '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
#             '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
#             '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '（', '）', '～',
#             '➡', '％', '⇒', '▶', '「', '➄', '➆',  '➊', '➋', '➌', '➍', '⓪', '①', '②', '③', '④', '⑤', '⑰', '❶', '❷', '❸', '❹', '❺', '❻', '❼', '❽',  
#             '＝', '※', '㈱', '､', '△', '℮', 'ⅼ', '‐', '｣', '┝', '↳', '◉', '／', '＋', '○',
#             '【', '】', '✅', '☑', '➤', 'ﾞ', '↳', '〶', '☛', '｢', '⁺', '『', '≫',
#             ] 
#     if word in spaces:
#         return False 
#     elif word in puncts:
#         return False
#     else:
#         return True

 

def main():
    # open file
    train_img2info = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/train_img2infobert.json', 'r')
    val_img2info = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/val_img2infobert.json', 'r')
    
    # read as json
    train_img2info = json.load(train_img2info)
    val_img2info = json.load(val_img2info)

    # 閾値 0.25 / 0.5 / 0.75
    sim_value = 0.5



    # 辞書をそのままpickleで保存
    # 画像のidをkey {key, captions, noise_captions}をvalueにした辞書
    train_img2infobert = build_img2info(train_img2info, sim_value)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/train2017_img2infobert.pkl', 'wb') as f:
        pickle.dump(train_img2infobert, f)   
    val_img2infobert = build_img2info(val_img2info, sim_value)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/val2017_img2infobert.pkl', 'wb') as f:
        pickle.dump(val_img2infobert, f) 

    
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/train_semantic_scoring/train2017_img2infobert.pkl', 'rb') as f:
        train_img2infobert = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/bert/val_semantic_scoring/val2017_img2infobert.pkl', 'rb') as f:
        val_img2infobert = pickle.load(f) 

    
    # 辞書をjsonとして書き込み
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/train_img2infobert.json', 'w') as f:
        json.dump(train_img2infobert, f, indent=4)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/val_img2infobert.json', 'w') as f:
        json.dump(val_img2infobert, f, indent=4)


    





if __name__ == '__main__':
    main()
