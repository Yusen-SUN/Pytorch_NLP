from __future__ import unicode_literals, print_function, division
from io import open

import string
import re
import random
import numpy as np
import os




def read_sentences(file):
    
    with open('../../dataset/GYAFC_Corpus/' + file + '.txt', 'r') as f:
        text = f.read()
    text = text.split('\n')[:-1]
    text = [i for i in text if i]
    return text

def fill_padding(seq_list, max_len, pad_id):
    seq_array = []
    for i in seq_list:
        if len(i) < max_len:
            seq_array.append(i + [pad_id]*(max_len-len(i) ))
            
        elif len(i) >= max_len:
            seq_array.append(i[:max_len])
                             
    seq_array = np.asarray(seq_array)
                             
    return seq_array


def input_data(sp_user, MAX):
    
    
    formal_train = read_sentences('formal-train')
    informal_train = read_sentences('informal-train')
    formal_val = read_sentences('formal-val')
    informal_val = read_sentences('informal-val')
        
    src_train = []
    src_train_lens = []
    tgt_train= []
    tgt_train_lens= []


    for i in range(len(formal_train)):

        seq_1 = sp_user.encode_as_ids(formal_train[i])
        seq_2 = sp_user.encode_as_ids(informal_train[i])
        
        if len(seq_1) <=5 or len(seq_2) <=5:
            continue
        src_train.append(seq_1)
        
        if len(seq_1)<MAX:
            src_train_lens.append(len(seq_1))
        else:
            src_train_lens.append(MAX)
    
        tgt_train.append(seq_2)
        
        if len(seq_2)<MAX:
            tgt_train_lens.append(len(seq_2))
        else:
             tgt_train_lens.append(MAX)
                                   
    src_train = fill_padding(src_train, MAX, sp_user.pad_id())
    src_train_lens = np.asarray(src_train_lens)
    tgt_train = fill_padding(tgt_train, MAX, sp_user.pad_id())
    tgt_train_lens = np.asarray(tgt_train_lens)
    
    print('src_train', src_train.shape)
    print('src_train_lens', src_train_lens.shape)
    print('tgt_train', tgt_train.shape)   
    print('tgt_train_lens', tgt_train_lens.shape)
    
    src_val = []
    src_val_lens = []
    tgt_val= []
    tgt_val_lens= []


    for i in range(len(formal_val)):


        seq_1 = sp_user.encode_as_ids(formal_val[i])
        seq_2 = sp_user.encode_as_ids(informal_val[i])
        
        if len(seq_1) <=5 or len(seq_2) <=5:
            continue
            
        src_val.append(seq_1)
        if len(seq_1)<MAX:
            src_val_lens.append(len(seq_1))
        else:
            src_val_lens.append(MAX)                       
                                   
        tgt_val.append(seq_2)
        if len(seq_2)<MAX:
            tgt_val_lens.append(len(seq_2))
        else:
            tgt_val_lens.append(MAX)                       

    src_val = fill_padding(src_val, MAX, sp_user.pad_id())
    src_val_lens = np.asarray(src_val_lens)
    tgt_val = fill_padding(tgt_val, MAX, sp_user.pad_id())
    tgt_val_lens = np.asarray(tgt_val_lens)
    
    print('src_val', src_val.shape)
    print('src_val_lens', src_val_lens.shape)
    print('tgt_val', tgt_val.shape)   
    print('tgt_val_lens', tgt_val_lens.shape)
    
    return src_train, src_train_lens, tgt_train, tgt_train_lens, src_val, src_val_lens, tgt_val, tgt_val_lens

    
    
    
    
    
