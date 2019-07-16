from __future__ import unicode_literals, print_function, division
from io import open

import string
import re
import random
import numpy as np
import os




def read_sentences(file):
    
    with open('../../dataset/style_transfer/' + file + '.txt', 'r') as f:
        text = f.read()
    text = text.split('\n')[:-1]
    
    return text


def input_data(Glove):
    formal_train = read_sentences('formal-train')
    informal_train = read_sentences('informal-train')
    formal_val = read_sentences('formal-val')
    informal_val = read_sentences('formal-val')
    
    MAX_LEN = 30
    
    src_train = []
    src_train_lens = []
    tgt_train= []


    for i in range(len(formal_train)):

        seq, length = Glove.sentence2seq(informal_train[i], MAX_LEN, i)
        src_train.append(seq)
        src_train_lens.append(length)

        seq, length = Glove.sentence2seq(formal_train[i], MAX_LEN, i)
        tgt_train.append(seq)

    src_train = np.asarray(src_train)
    src_train_lens = np.asarray(src_train_lens)
    tgt_train = np.asarray(tgt_train)
    
    print('src_train', src_train.shape)
    print('src_train_lens', src_train_lens.shape)
    print('tgt_train', tgt_train.shape)    
    
    src_val = []
    src_val_lens = []
    tgt_val = []    

    
    
    for i in range(len(formal_val)):

        seq, length = Glove.sentence2seq(informal_val[i], MAX_LEN, i)
        src_val.append(seq)
        src_val_lens.append(length)

        seq, length = Glove.sentence2seq(formal_val[i], MAX_LEN, i)
        tgt_val.append(seq)

    src_val = np.asarray(src_val)
    src_val_lens = np.asarray(src_val_lens)
    tgt_val = np.asarray(tgt_val) 
    
    print('src_val', src_val.shape)
    print('src_val_lens', src_val_lens.shape)
    print('tgt_val', tgt_val.shape)   
    
    return src_train, src_train_lens, tgt_train, src_val, src_val_lens, tgt_val

    
    
    
    
    
