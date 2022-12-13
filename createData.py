#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk, re, json, string
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import numpy as np  
import pandas as pd
import string
from nltk.corpus import stopwords


novel_data = []

for i in os.listdir('Dataset/'):
    with open('Dataset/'+i, encoding="utf8", errors="ignore") as file:
        content = file.read().rstrip().replace("\n", "")
        
        novel_data.append(content)

print(len(novel_data))


# Define the function

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return(wordnet.ADJ)
    elif nltk_tag.startswith('V'):
        return(wordnet.VERB)
    elif nltk_tag.startswith('N'):
        return(wordnet.NOUN)
    elif nltk_tag.startswith('R'):
        return(wordnet.ADV)
    else:          
        return None

def tokenize(doc):
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    word_token = nltk.word_tokenize(doc)

    unigram = list(ngrams(word_token, 1))
    unigram = [i[0] for i in unigram]

    unigram = [i for i in unigram if i not in stop_words]
    unigram = [i for i in unigram if i not in string.punctuation]
    unigram = [i for i in unigram if i!= ('—' or '``' or "''")]
    unigram = [i.lower() for i in unigram]
    
    pos_tag = nltk.pos_tag(unigram)

    clean_unigram = []
    
    for i in pos_tag:
        try:
            clean_unigram.append(lemmatizer.lemmatize(i[0], nltk_pos_tagger(i[1])))
        except:
            clean_unigram.append(i[0])
       
    return(clean_unigram)
     


def compute_tfidf(docs):
    
    smoothed_tf_idf = None
    
    def get_doc_tokens(i):
            tokens = tokenize(i)
            token_count=nltk.FreqDist(tokens)
            return token_count
        
    docs_tokens={idx:get_doc_tokens(doc) for idx,doc in enumerate(docs)}
   
    # put words as columns
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index")
    dtm = dtm.sort_index(axis = 0)
    dtm=dtm.fillna(0)
    tf=dtm.values
    
    # sum of each rows
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf, doc_len[:,None])

    # find freq of each term in all docs
    df=np.where(tf>0,1,0)
       
    idf=np.log(np.divide(len(docs), np.sum(df, axis=0)))+1
  
    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1)+1)
   
    s = tf*idf
    tf_idf=normalize(tf*idf)   
    smoothed_tf_idf = normalize(tf*smoothed_idf)
        
    return smoothed_tf_idf



tf_idf_novel = compute_tfidf(novel_data)



df = pd.DataFrame(tf_idf_novel)



df.to_csv('./giantNP.csv', header=False)
