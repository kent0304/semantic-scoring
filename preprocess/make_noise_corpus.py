# 誤り文のみのコーパス作成
import pickle

def load_img2info():
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_img2info.pkl', 'rb') as f:
        train_img2info = pickle.load(f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_img2info.pkl', 'rb') as f:
        val_img2info = pickle.load(f) 
    return train_img2info, val_img2info

def parse_noisecaptions(img2info, noise_corpus):
    for dic in img2info.values():
        noise_corpus += dic['noise_captions']
    return noise_corpus


def main():
    noise_corpus = []
    train_img2info, val_img2info = load_img2info()
    noise_corpus = parse_noisecaptions(train_img2info, noise_corpus)
    noise_corpus = parse_noisecaptions(val_img2info, noise_corpus)
    n = '\n'.join(noise_corpus)
    # コーパスファイル書き込み
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/annotations/noise_corpus.txt', 'w') as f:
        f.write(n)
    return

if __name__ == '__main__':
    main()