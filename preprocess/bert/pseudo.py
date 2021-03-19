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

caption = "A giraffe looks down at a zebra on the field."

# 形態素解析
morph = nltk.word_tokenize(caption.lower())
pos = nltk.pos_tag(morph)
morph = ["[CLS]"] + morph + ["[SEP]"]

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

        _, predicted_indexes = torch.topk(predictions[i+1], k=20)

        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
        print('今回対象のトークン:', token[0])
        print('BERTで予測した置き換えられる語:', predicted_tokens)
        for j, word in enumerate(predicted_tokens):
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
                    if w1.wup_similarity(w2) < 0.5: # 類似度0.5未満なら採用
                        # print(word, 'で決定')
                        # final_noise_caption[i+1] = word
                        print("適用前")
                        print(noise_caption)
                        noise_caption[i+1] = word
                        final_noise_caption = noise_caption[1:-1]
                        final_noise_caption = ' '.join(final_noise_caption)
                    
                   
                        print("適用後")
                        print(noise_caption)
                        break
                    else: # 類似度0.5以上なら次の候補へ
                        continue
                except nltk.corpus.reader.wordnet.WordNetError: 
                    continue 