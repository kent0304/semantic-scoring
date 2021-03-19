import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
plt.switch_backend('agg')

font = {"family":"IPAexGothic"}
mpl.rc('font', **font)

# グラフの描画先の準備
fig = plt.figure()

## 写真描画問題全体の誤り分析
total = np.array([16, 73, 14])
total_label = ["画像に関連していない", "文法誤りがある", "語句を使用していない"]

plt.pie(total, labels=total_label, explode=[0.1, 0, 0], autopct="%1.1f%%")
plt.axis('equal')
plt.savefig('total_annalyse.png')


# グラフの描画先の準備
fig = plt.figure()

## Based on Imageに着目した誤り分析
image = np.array([14, 3, 3])
image_label = ["Content Words", "Opinions", "The Number of Objects"]

plt.pie(image, labels=image_label, autopct="%1.1f%%")
plt.axis('equal')
plt.savefig('image_annalyse.png')