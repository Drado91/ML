import numpy as np
#quad = lambda x: (np.sum(x ** 2), x * 2)
def read_data(fname):
    data = []
    for line in fname:
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

f = open('train', 'r')
#data=read_data(f)
TRAIN = [(l,text_to_bigrams(data)) for l,data in read_data(f)]
#for i in TRAIN:
   # print(TRAIN[1])
