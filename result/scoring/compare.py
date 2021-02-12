import csv 
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# 人手採点
with open('human.csv') as f:
    reader = csv.reader(f)
    human = [row for row in reader]

# bertで類義語カットしたデータで学習させたモデルによる採点
with open('scoring_wn025.csv') as f:
    reader = csv.reader(f)
    wn025 = [row for row in reader]
with open('scoring_wn05.csv') as f:
    reader = csv.reader(f)
    wn05 = [row for row in reader]
with open('scoring_wn075.csv') as f:
    reader = csv.reader(f)
    wn075 = [row for row in reader]

def cal(human, target):
    human = list(itertools.chain.from_iterable(human[1:]))
    target = list(itertools.chain.from_iterable(target[1:]))
    human = [int(elm) for elm in human]
    target = [int(elm) for elm in target]
    cm = confusion_matrix(human, target)
    print(cm)
    return f1_score(human, target)


def main():
    wn025_f1 = cal(human, wn025)
    wn05_f1 = cal(human, wn05)
    wn075_f1 = cal(human, wn075)

    result = 'wn025: ' + str(wn025_f1) + '\n'
    result += 'wn05: ' + str(wn05_f1) + '\n'
    result += 'wn075: ' + str(wn075_f1) + '\n'

    with open('scoring_compared.txt', 'w') as f:
        f.write(result)

    return 


if __name__ == '__main__':
    main()