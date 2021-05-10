import matplotlib.pyplot as plt
import numpy as np

file = "E:\\tf_projects\\Handwriting-Dataset3\\chinese_english\\chinese1\\baixuerui.dta"
begin = False
_X = []
_Y = []
_S = []
with open(file) as f:
    stroke = 0
    cnt = 0
    button_change = False
    for _ in range(0, 2000000):
        l = f.readline().strip()
        if begin:
            infos = l.split()
            if infos[0] == "End":
                break
            if int(infos[1]) == 1:
                if not button_change:
                    button_change = True
                    stroke += 1
                _X.append(float(infos[4]))
                _Y.append(float(infos[5]))
                _S.append(stroke)
            if int(infos[1]) == 0:
                button_change = False
        if not begin and len(l) > 0 and l.split()[0] == "TIME":
            begin = True
X = np.array(_X)
Y = np.array(_Y)
S = np.array(_S)

print("total points = " + str(len(X)))


def remove_point_dist(_x, _y, _dist):
    i = 1
    while i < len(_x):
        if np.sqrt((_x[i] - _x[i - 1]) ** 2 + (_y[i] - _y[i - 1]) ** 2) < _dist:
            _x = np.delete(_x, [i])
            _y = np.delete(_y, [i])
            continue
        i += 1
    return _x, _y

def remove_point_ang(_x, _y, _tang):
    i = 1
    while i < len(_x) - 1:
        if ((_x[i]-_x[i-1])*(_x[i+1]-_x[i])+(_y[i]-_y[i-1])*(_y[i+1]-_y[i]))/np.sqrt(((_x[i]-_x[i-1])**2+(_y[i]-_y[i-1])**2)*((_x[i+1]-_x[i])**2+(_y[i+1]-_y[i])**2)) > Tang:
            _x = np.delete(_x, [i])
            _y = np.delete(_y, [i])
            continue
        i += 1
    return _x, _y

Dist = 20
Tang = 0.98
stroke_x = []
stroke_y = []
stroke_s = []
for i in range(1, np.max(S) + 1):
    point_cnt = np.count_nonzero(S == i)
    if point_cnt <= 2:
        continue
    sx = X[S == i]
    sy = Y[S == i]
    sx, sy = remove_point_dist(sx, sy, Dist)
    if len(sx) >= 3:
        sx, sy = remove_point_ang(sx, sy, Tang)
    stroke_x.append(sx)
    stroke_y.append(sy)
    stroke_s.append(np.array([i for _ in range(0, len(sx))]))

removed_total = 0

chars_stroke = [0,9,9,1,4,7,6,1,6,15,9,7,1]
_sum = 0
sum_stroke = np.zeros(len(chars_stroke), dtype=int)
for i in range(0,len(chars_stroke)):
    _sum += chars_stroke[i]
    sum_stroke[i] = _sum

colors = ["red", "orange", "green", "blue", "black"]

begin_char = 0
end_char = 0
cur_char = begin_char
# for i in range(sum_stroke[begin_char], sum_stroke[end_char + 1]):
#     if i >= sum_stroke[cur_char + 1]:
#         cur_char += 1
#     plt.plot(stroke_x[i], stroke_y[i], color=colors[cur_char % 5])
#     removed_total += len(stroke_x[i])

char_x = stroke_x[sum_stroke[begin_char] : sum_stroke[end_char + 1]]
char_y = stroke_y[sum_stroke[begin_char] : sum_stroke[end_char + 1]]

lenl = []
pxl = []
pyl = []
for i in range(0, sum_stroke[end_char + 1] - sum_stroke[begin_char]):
    for j in range(0, len(char_x[i]) - 1):
        _lenl = np.sqrt((char_x[i][j+1] - char_x[i][j])**2 + (char_y[i][j+1] - char_y[i][j])**2)
        lenl.append(_lenl)
        pxl.append(_lenl * (char_x[i][j+1] + char_x[i][j]) / 2)
        pyl.append(_lenl * (char_y[i][j+1] + char_y[i][j]) / 2)

ux = np.sum(pxl) / np.sum(lenl)
uy = np.sum(pyl) / np.sum(lenl)

dxl = []
k = 0
for i in range(0, sum_stroke[end_char + 1] - sum_stroke[begin_char]):
    for j in range(0, len(char_x[i]) - 1):
        dxl.append(1/3*lenl[k]*((char_x[i][j+1]-ux)**2+(char_x[i][j]-ux)**2+(char_x[i][j+1]-ux)*(char_x[i][j]-ux)))
        k += 1

thx = np.sqrt(np.sum(dxl) / np.sum(lenl))

# fig_before_preprocess = plt.figure()
# fig_before_preprocess_plt = fig1.add_subplot(1,1,1)
#
# for i in range(sum_stroke[begin_char], sum_stroke[end_char + 1]):
#     fig_before_preprocess_plt.plot(stroke_x[i], stroke_y[i], color="black")

for i in range(0, sum_stroke[end_char + 1] - sum_stroke[begin_char]):
    for j in range(0, len(char_x[i])):
        char_x[i][j] = (char_x[i][j] - ux) / thx
        char_y[i][j] = (char_y[i][j] - uy) / thx

# fig_after_preprocess = plt.figure()
# fig_after_preprocess_plt = fig_after_preprocess.add_subplot(1,1,1)
#
# for i in range(sum_stroke[begin_char], sum_stroke[end_char + 1]):
#     fig_after_preprocess_plt.plot(stroke_x[i], stroke_y[i], color="black")

#print("after remove, total points = " + str(removed_total))
#plt.show()

L = []
for i in range(0, len(char_x)):
    for j in range(0, len(char_x[i])):
        Li = np.zeros(6, dtype=float)
        if j != len(char_x[i]) - 1:
            Li[0] = char_x[i][j]
            Li[1] = char_y[i][j]
            Li[2] = char_x[i][j+1] - char_x[i][j]
            Li[3] = char_y[i][j+1] - char_y[i][j]
            Li[4] = 1
            Li[5] = 0
            L.append(Li)
        else:
            if i != len(char_x) - 1:
                Li[0] = char_x[i][j]
                Li[1] = char_y[i][j]
                Li[2] = char_x[i+1][0] - char_x[i][j]
                Li[3] = char_y[i+1][0] - char_y[i][j]
                Li[4] = 0
                Li[5] = 1
                L.append(Li)

