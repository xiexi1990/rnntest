import pickle
import numpy as np
from collections import Counter
import random

with open("dump_la_10", "rb") as f:
    la_10 = pickle.load(f)

la_10.sort(key=lambda l:l[1])

la_group = []
i = 0
while i < len(la_10):
    la_item = []
    tag = la_10[i][1]
    while tag == la_10[i][1]:
        la_item.append(la_10[i])
        i += 1;
        if i == len(la_10):
            break
    la_group.append(la_item)

rand_group = random.sample(la_group, 10)

rand_la = []
for i in range(0, len(rand_group)):
    for j in range(0, len(rand_group[i])):
        rand_la.append(rand_group[i][j])

with open("rand_la_100", "wb") as f:
    pickle.dump(rand_la, f)

print(len(rand_la))