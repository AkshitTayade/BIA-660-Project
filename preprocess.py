#!/usr/bin/python

import sys
import os
import numpy as np
import pandas as pd
# import classifier.py as C

# this should be the filename to some saved TFIDF document
filepath = sys.argv[1]
topN = int(sys.argv[2])
f = open(filepath)
df = pd.read_csv(filepath)
tfidf = df.to_numpy()
pf = pd.read_csv("./project.csv", header=False)
labels = df2.to_numpy()
prepro = np.argsort(tfidf, axis=1)[:,:topN]
f.close()
os.system("python classifier.py")
voc = ['youdidntincludethevocab']*prepro.shape[1]
gc = GenreClassifier(prepro,None,voc,labels,14,topN)
answer = np.array(gc.KMeans())
np.save_txt("./output.txt",answer)
