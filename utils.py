# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
def read_data(fname): #create 2d array of data (list); [first char , rest sentence]; for each sentence in file; size = 2*LINES
    data = []
    for line in fname:
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data(open('train', 'r'))]
DEV   = [(l,text_to_bigrams(t)) for l,t in read_data(open('dev', 'r'))]

def vocabu(dataset): #train/test/dev
    from collections import Counter
    fc = Counter()
    for l,feats in dataset:
        fc.update(feats)
    vocab = set([x for x, c in fc.most_common(600)])
    return vocab
# 600 most common bigrams in the training set.


# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
# feature strings (bigrams) to IDs

F2I = {f:i for i,f in enumerate(list(sorted(vocabu(TRAIN))))}


