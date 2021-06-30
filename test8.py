import pickle
import numpy as np
from collections import Counter
#from keras.utils import to_categorical
import xlwt

with open("dump_la", "rb") as f:
    la = pickle.load(f)
    f.close()

la.sort(key=lambda l: np.size(l[2], 0))

filelist = []
taglist = []
strokeslist = []
lenlist = []

book = xlwt.Workbook()
sheet = book.add_sheet("sheeta")

for i in range(0, len(la)):
    # filelist.append(la[i][0])
    tagbytes = la[i][1].encode("gb18030")
    tag = tagbytes[0] * 256 + tagbytes[1]
    # taglist.append(tag)
    # strokeslist.append(la[i][2])
    # lenlist.append(np.size(la[i][2], 0))
    sheet.write(i, 0, tag)

book.save("chars2.xls")