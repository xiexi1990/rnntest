import pickle
import numpy as np
from collections import Counter
import random

with open("dump_la_30", "rb") as f:
    la_30 = pickle.load(f)
    f.close()

la_30.sort(key=lambda l:l[1])
la_group = []
i = 0
while i < len(la_30):
    la_item = []
    tag = la_30[i][1]
    while tag == la_30[i][1]:
        la_item.append(la_30[i])
        i += 1;
        if i == len(la_30):
            break
    la_group.append(la_item)

rand_group = random.sample(la_group, 30)

rand_la = []
for i in range(0, len(rand_group)):
    for j in range(0, len(rand_group[i])):
        rand_la.append(rand_group[i][j])

with open("rand_la_900", "wb") as f:
    pickle.dump(rand_la, f)
    f.close()

print(len(rand_la))