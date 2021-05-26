# mscoco jsonからテキストファイル準備
# txtfileは画像のid、語句、正しいキャプション文、正誤ラベル、画像のインデックスの順に文字列保存
import json 
import pickle
import itertools
from tqdm import tqdm
from nltk import stem
lemmatizer = stem.WordNetLemmatizer()


def each_write_txt(json_obj, ver):
    # 出力するファイルをstringで管理
    output = ''
    image_idx = -1
    pre_imageid = ''
    for image_set in tqdm(json_obj.items(), total=len(json_obj)):
        # タプル要素1つ目に画像のid
        imageid = image_set[0]
        # タプル要素2つ目に画像以外のメタ情報
        meta = image_set[1]
        for i, k in enumerate(meta['key']):
            # 語句が存在するかどうか
            if k != []:
                # 語句は原型に変換
                k = ' '.join([lemmatizer.lemmatize(elm, pos='v') for elm in k])
                if pre_imageid != imageid:
                    image_idx += 1
                cap = meta['captions'][i].strip().replace('\n', ' ')
                # 画像のid、語句、正しいキャプション文、正誤ラベル、画像のインデックスの順に文字列保存
                row = imageid + '\t' + k + '\t' + cap + '\t' + str(1) + '\t' + str(image_idx) + '\n'
                pre_imageid = imageid
                output += row
                for n_cap in meta[ver + '_noise_captions'][i]:
                    row = imageid + '\t' + k + '\t' + n_cap.strip().replace('\n', ' ')+ '\t' + str(0) + '\t' + str(image_idx) + '\n'
                    output += row

    return output

def write_txt(json_obj, ver):
    # 出力するファイルをstringで管理
    output = ''
    image_idx = -1
    pre_imageid = ''
    for image_set in tqdm(json_obj.items(), total=len(json_obj)):
        imageid = image_set[0]
        meta = image_set[1]
        for i, k in enumerate(meta['key']):
            
            # 語句が存在するかどうか
            if k != []:
                # 原型に変換
                k = ' '.join([lemmatizer.lemmatize(elm, pos='v') for elm in k])
                if pre_imageid != imageid:
                    image_idx += 1
                cap = meta['captions'][i].strip().replace('\n', ' ').replace('\t', '')
                row = imageid + '\t' + k + '\t' + cap + '\t' + str(1) + '\t' + str(image_idx) + '\n'
                pre_imageid = imageid
                output += row
                n_cap = meta[ver+'_noise_captions'][i].strip().replace('\n', ' ').replace('\t', '').replace('"', '')
                row = imageid + '\t' + k + '\t' + n_cap + '\t' + str(0) + '\t' + str(image_idx) + '\n'
                pre_imageid = imageid
                output += row

    return output


def main():
    # open mscoco json file
    train_img2info = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/train_img2infobert.json', 'r')
    val_img2info = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/val_img2infobert.json', 'r')
    
    # read as json
    train_img2info = json.load(train_img2info)
    val_img2info = json.load(val_img2info)

    ver_list = ['berteach_wn025', 'berteach_wn05', 'berteach_wn075', 'bert_wn025', 'bert_wn05', 'bert_wn075']

    for ver in ver_list:
        if 'each' in ver:
            output = each_write_txt(train_img2info, ver)
        else:
            output = write_txt(train_img2info, ver)
        # 書き込み
        with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/{}/train/output.txt'.format(ver), 'w') as f:
            f.write(output)

        if 'each' in ver:
            output = each_write_txt(val_img2info, ver)
        else:
            output = write_txt(val_img2info, ver)
        # 書き込み
        with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/{}/valid/output.txt'.format(ver), 'w') as f:
            f.write(output)

    return

if __name__ == '__main__':
    main()