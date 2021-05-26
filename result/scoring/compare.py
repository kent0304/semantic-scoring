import csv 
import itertools
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix

# 人手採点
with open('human.csv') as f:
    reader = csv.reader(f)
    human = [row for row in reader]
with open('random.csv') as f:
    reader = csv.reader(f)
    random = [row for row in reader]

# bertで類義語カットしたデータで学習させたモデルによる採点
with open('scoring_0220wn025.csv') as f:
    reader = csv.reader(f)
    wn025 = [row for row in reader]
with open('scoring_0220wn05.csv') as f:
    reader = csv.reader(f)
    wn05 = [row for row in reader]
with open('scoring_0220wn075.csv') as f:
    reader = csv.reader(f)
    wn075 = [row for row in reader]


# 負例1対多
with open('scoring_0220eachwn025.csv') as f:
    reader = csv.reader(f)
    eachwn025 = [row for row in reader]
with open('scoring_0220eachwn05.csv') as f:
    reader = csv.reader(f)
    eachwn05 = [row for row in reader]
with open('scoring_0220eachwn075.csv') as f:
    reader = csv.reader(f)
    eachwn075 = [row for row in reader]

# lxmert
with open('pretrained_lxmert.csv') as f:
    reader = csv.reader(f)
    lxmert = [row for row in reader]

def cal(human, target):
    human = list(itertools.chain.from_iterable(human[1:]))
    target = list(itertools.chain.from_iterable(target[1:]))
    human = [int(elm) for elm in human]
    new_human = []
    for e in human:
        if e == 0:
            new_human.append(1)
        else:
            new_human.append(0)

    target = [int(elm) for elm in target]
    new_target = []
    for e in target:
        if e == 0:
            new_target.append(1)
        else:
            new_target.append(0)
    cm = confusion_matrix(new_human, new_target)
    # print(cm)
    # print(new_human)
    # print(new_target)
    return precision_score(new_human, new_target), recall_score(new_human, new_target), f1_score(new_human, new_target)


def main():
    # wn025_f1 = cal(human, wn025)
    # print('wn025_f1', wn025_f1)
    # wn05_f1 = cal(human, wn05)
    # print('wn05_f1', wn05_f1)
    # wn075_f1 = cal(human, wn075)
    # print('wn075_f1', wn075_f1)
    random_p, random_r, random_f1 =  cal(human, random)
    # print('ランダム', cal(human, random))
    eachwn025_p, eachwn025_r, eachwn025_f1 = cal(human, eachwn025)
    # print('eachwn025_f1', eachwn025_f1)
    eachwn05_p, eachwn05_r, eachwn05_f1= cal(human, eachwn05)
    # print('eachwn05_f1', eachwn05_f1)
    eachwn075_p, eachwn075_r, eachwn075_f1= cal(human, eachwn075)
    # print('eachwn075_f1', eachwn075_f1)
    lxmert_p, lxmert_r, lxmert_f1= cal(human, lxmert)
    # print('lxmert_f1', lxmert_f1)

    # result = 'wn025: ' + str(wn025_f1) + '\n'
    # result += 'wn05: ' + str(wn05_f1) + '\n'
    # result += 'wn075: ' + str(wn075_f1) + '\n'
    result = 'ランダム: ' + str(random_p) +'\t' + str(random_r)+'\t' + str(random_f1) + '\n'
    result += 'eachwn025: ' + str(eachwn025_p) +'\t' + str(eachwn025_r)+'\t' + str(eachwn025_f1) + '\n'
    result += 'eachwn05: ' + str(eachwn05_p) +'\t' + str(eachwn05_r) +'\t'+ str(eachwn05_f1) + '\n'
    result += 'eachwn075: ' + str(eachwn075_p) +'\t' + str(eachwn075_r)+'\t' + str(eachwn075_f1) + '\n'
    result += 'lxmert_f1: ' + str(lxmert_p) +'\t' + str(lxmert_r)+'\t' + str(lxmert_f1) + '\n'

    with open('scoring_compared_new.txt', 'w') as f:
        f.write(result)

    return 


if __name__ == '__main__':
    main()