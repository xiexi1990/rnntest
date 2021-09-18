import pickle
import numpy as np
import random

lb_len = 3
nclass = 30

with open("dump_lb_" + str(lb_len), "rb") as f:
    lb = pickle.load(f)

#print(np.unique(np.array([r[0] for r in lb])))

lb2 = random.sample(list(filter(lambda r: r[0] == 1001, lb)), nclass)

y = []
x = []
i = 0

while i < nclass:
    for
    # tagbytes = la[i][1].encode("gb18030")
    # tag = tagbytes[0] * 256 + tagbytes[1]
    # taglist.append(tag)
    curtag = la[i][1]
    cur_y = np.zeros(nclass)
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

with open("x_y_" + str(la_len), "wb") as f:
    pickle.dump((x, y), f)

#_, _taglist = np.unique(taglist, return_inverse=True)

