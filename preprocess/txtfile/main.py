import json 
from tqdm import tqdm
from nltk import stem
lemmatizer = stem.WordNetLemmatizer()


def each_write_txt(json_obj, ver):
    # 出力するファイルをstringで管理
    output = ''
    image_idx = 0
    for image_set in tqdm(json_obj.items(), total=len(json_obj)):
        imageid = image_set[0]
        meta = image_set[1]
        for i, k in enumerate(meta['key']):
            # 語句が存在するかどうか
            if k != []:
                # 原型に変換
                k = ' '.join([lemmatizer.lemmatize(elm, pos='v') for elm in k])
                cap = meta['captions'][i].strip().replace('\n', ' ')
                row = imageid + '\t' + k + '\t' + cap + '\t' + str(1) + '\t' + str(image_idx) + '\n'
                output += row
                for n_cap in meta[ver+'_noise_captions'][i]:
                    row = imageid + '\t' + k + '\t' + n_cap.strip().replace('\n', ' ')+ '\t' + str(0) + '\t' + str(image_idx) + '\n'
                    output += row
        image_idx += 1
    return output

def write_txt(json_obj, ver):
    # 出力するファイルをstringで管理
    output = ''
    image_idx = 0
    for image_set in tqdm(json_obj.items(), total=len(json_obj)):
        imageid = image_set[0]
        meta = image_set[1]
        for i, k in enumerate(meta['key']):
            # 語句が存在するかどうか
            if k != []:
                # 原型に変換
                k = ' '.join([lemmatizer.lemmatize(elm, pos='v') for elm in k])
                cap = meta['captions'][i].strip().replace('\n', ' ')
                row = imageid + '\t' + k + '\t' + cap + '\t' + str(1) + '\t' + str(image_idx) + '\n'
                output += row
                n_cap = meta[ver+'_noise_captions'][i].strip().replace('\n', ' ')
                row = imageid + '\t' + k + '\t' + n_cap + '\t' + str(0) + '\t' + str(image_idx) + '\n'
                output += row
        image_idx += 1
    return output

def random_write_txt(json_obj):
    # 出力するファイルをstringで管理
    output = ''
    image_idx = 0
    for image_set in tqdm(json_obj.items(), total=len(json_obj)):
        imageid = image_set[0]
        meta = image_set[1]
        for i, k in enumerate(meta['key']):
            # 語句が存在するかどうか
            if k != []:
                # 原型に変換
                k = ' '.join([lemmatizer.lemmatize(elm, pos='v') for elm in k])
                cap = meta['captions'][i].strip().replace('\n', ' ')
                row = imageid + '\t' + k + '\t' + cap + '\t' + str(1) + '\t' + str(image_idx) + '\n'
                output += row
                n_cap = meta['noise_captions'][i].strip().replace('\n', ' ')
                row = imageid + '\t' + k + '\t' + n_cap + '\t' + str(0) + '\t' + str(image_idx) + '\n'
                output += row
        image_idx += 1
    return output

def main():
    # open file
    train_img2info = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/train_img2infobert.json', 'r')
    val_img2info = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/val_img2infobert.json', 'r')
    
    # read as json
    train_img2info = json.load(train_img2info)
    val_img2info = json.load(val_img2info)

    ver_list = ['berteach_wn025', 'berteach_wn05', 'berteach_wn075', 'bert_wn025', 'bert_wn05', 'bert_wn075', 'random']

    for ver in ver_list:
        if 'each' in ver:
            output = each_write_txt(train_img2info, ver)
        elif ver == 'random':
            output = random_write_txt(train_img2info)
        else:
            output = write_txt(train_img2info, ver)
        with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/{}/train/output.txt'.format(ver), 'w') as f:
            f.write(output)

        if 'each' in ver:
            output = each_write_txt(val_img2info, ver)
        elif ver == 'random':
            output = random_write_txt(val_img2info)
        else:
            output = write_txt(val_img2info, ver)
        with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/txtfile/{}/valid/output.txt'.format(ver), 'w') as f:
            f.write(output)

    return

if __name__ == '__main__':
    main()