# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 23:01:24 2018

@author: Eigenaar
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:55:33 2018

@author: Edwin Lima, Efi Athieniti
"""

'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''

import matplotlib.pyplot as plt
from scipy.stats import norm


from nltk import sent_tokenize
from collections import defaultdict
import keras

import numpy as np
import csv
from nltk.corpus import stopwords
from time import strftime, gmtime
from collections import Counter
import os
import sys

window_sz = 5 #five words left, five words right
stopwords = set(stopwords.words('english'))
dir = 'C://Users//Eigenaar//Documents//GitHub//sa//aclImdb//train/pos'
sfile_path = ''

def read_input(dir, most_common=None):
    """ Read a corpus file and create a list of sentences and vocabulary
    
    @param fn         : filename
    @param most_common: number of most frequent words to keep
    
    @return: 
        word2idx
        idx2word
        sentences_tokens: list of tokenized sentences
        
    """
    
    print(strftime("%H:%M:%S", gmtime()), " Reading input sentences..")

    for fn in os.listdir(dir):
        with open(dir +'//' +'18_7.txt', 'r') as content_file:
            content = content_file.read()
            #print(content)
        sentences = sent_tokenize(content)
        print(sentences)
        sys.exit()
        
        punctuation = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '\n']
        
        sentences_tokens = []
        corpus = []
        reserved = ['<null>' ,'<unk>']
        print(strftime("%H:%M:%S", gmtime()), "File read")
    
        for sentence in sentences:
            #s= [w for w in sentence.split() if w not in punctuation]
            s=[]
            for w in sentence.split():
                if w not in punctuation:
                    if w in stopwords:
                        w=reserved[1]
                    s.append(w)
    
            sentences_tokens.append(s)
            corpus = corpus + s
        counts = Counter(corpus)
        print('len corpus=', len(set(corpus)))
        print(strftime(" %H:%M:%S", gmtime()), "created corpus")
    
        if most_common:
            corpus = set(map(lambda x: x[0], counts.most_common(most_common)))
        else:
            corpus = set(corpus)
        corpus = corpus.union(set(reserved))
        print("updated corpus len", len(corpus))
        word2idx, idx2word=encode_corpus(corpus)

    print(strftime(" %H:%M:%S", gmtime()),"Finished reading input sentences")

    return word2idx, idx2word, sentences_tokens, corpus



def save_embeddings(embeddings_file, embeddings, idx2word):
    """
    Write embeddings to file
    @param embeddings     : numpy array of the embeddings
                            shape(vocab_size, emb_sz)
    @param embeddings_file: filename to write embeddings to
    @param idx2word       : dictionary with word index as key
    """
    
    with open(embeddings_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        print(embeddings.shape)
        writer.writerow([embeddings.shape[1], embeddings.shape[0]])

        for i in range(embeddings.shape[1]):
            word = idx2word[i]
            embedding = embeddings[:,i]
            embedding = list(embedding)
            line = [word] + embedding
            writer.writerow(line)

def main():
    read_input(dir)
    
if __name__ == "__main__":           
    main()
