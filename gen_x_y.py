import pickle
from collections import Counter
#from keras.utils import to_categorical
import numpy as np

with open("rand_la_900", "rb") as f:
    la = pickle.load(f)
    f.close()

y = []
x = []
j = 0
i = 0
while i < len(la):
    # tagbytes = la[i][1].encode("gb18030")
    # tag = tagbytes[0] * 256 + tagbytes[1]
    # taglist.append(tag)
    curtag = la[i][1]
    cur_y = np.zeros(30)
    cur_y[j] = 1
    while la[i][1] == curtag:
        x.append(la[i][2])
        y.append(cur_y)
        i += 1
        if i == len(la):
            break
    j += 1

z = sorted(zip(x, y), key=lambda l:np.size(l[0], 0))
result = zip(*z)

x, y = [list(i) for i in result]
print(len(y))
#_, _taglist = np.unique(taglist, return_inverse=True)

