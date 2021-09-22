import pickle
import numpy as np
import random

lb_len = 3
nclass = 10
drop = 0.1
repeat = 10

with open("dump_lb_" + str(lb_len), "rb") as f:
    lb = pickle.load(f)

#print(np.unique(np.array([r[0] for r in lb])))

lb2 = random.sample(list(filter(lambda r: r[0] == 1001, lb)), nclass)

y = []
x = []
i = 0

for i in range(nclass):
    char = lb2[i]
    for j in range(repeat):
        cx = [np.append(np.zeros(5), i)]
        k = 0
        first = True
        while k < np.size(char[2], 0) - 1:
            if random.random() < drop:
                k += 1
                continue
            p = [np.append(np.append(char[2][k], char[3][k]), i)]
            if first:
                first = False
                cy = p
            else:
                cy = np.append(cy, p, 0)
            cx = np.append(cx, p, 0)
            k += 1
        cy = np.append(cy, [np.append(np.append(char[2][k], char[3][k]), i)], 0)
        x.append(cx)
        y.append(cy)


z = sorted(zip(x, y), key=lambda l:np.size(l[0], 0))
result = zip(*z)

x, y = [list(i) for i in result]

batchx = []
batchy = []
while i < x.__len__():
    xb = np.expand_dims(x[i], 0)
    yb = np.expand_dims(y[i], 0)
    j = i + 1
    while j < x.__len__():
        if np.size(x[j], 0) == np.size(x[i], 0):
            xb = np.append(xb, np.expand_dims(x[j], 0), 0)
            yb = np.append(yb, np.expand_dims(y[j], 0), 0)
            j += 1
        else:
            break
    i = j
    batchx.append(xb)
    batchy.append(yb)


with open("x_y_lb_" + str(nclass) + "_repeat_" + str(repeat), "wb") as f:
    pickle.dump((batchx, batchy), f)

#_, _taglist = np.unique(taglist, return_inverse=True)

