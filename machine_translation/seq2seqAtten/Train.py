from __future__ import unicode_literals, print_function, division
from io import open

import string
import re
import random
import numpy as np
import os
import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from EncoderRNN import EncoderRNN
from Attention import Attn
from DecoderRNN import DecoderRNN
from Pretrained_embedding import pre_embedding


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)


def trainBatch(input_tensors, input_lengths, target_tensors, embedding, 
               encoder, decoder, mode, criterion=None,
               encoder_optimizer=None, decoder_optimizer=None ):
    
    # input_tensors (batch, seq)
    # target_tensors (batch, seq)
    # input_lengths (batch)
    # embedding (vocab, embedding_dim)
    
    loss = 0
    
    if mode=='train':
        encoder.train()
        decoder.train()
            
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
    elif mode=='val':
        encoder.eval()
        decoder.eval()
    
    else:
        print('wrong mode')
        
    
    input_tensors = input_tensors.t()

    target_length, batch_size= input_tensors.size()

    # encoder_outputs: seq, batch, hidden
    encoder_outputs, h, c = encoder.forward(input_tensors, input_lengths, embedding)
    
    # decoder_outputs: batch, seq, vocab
    # output_seq: batch, seq
    # atten: batch, seq, seq
    decoder_outputs, output_seq, atten = decoder.forward(embedding, h, c, encoder_outputs)
    
    output_len = decoder_outputs.size()[1]
        
    for i in range(batch_size):
        
        loss += criterion(decoder_outputs[i], target_tensors[i,:output_len])
        #print(loss.item())
        
    if mode=='train':
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        
        return loss.item()
    
    else:
        return loss.item(), output_seq.cpu().numpy(), atten.cpu().numpy()
    
    
    
def trainEpoch(input_train_tensors, input_train_lengths_tensors, output_train_tensors, embedding, 
               encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
               epoches, batch_size):
    
    # input_train_tensors: batch, seq
    
    train_size = len(input_train_tensors)
    orders = np.arange(train_size)
    np.random.shuffle(orders)
    total_loss = 0

    for i in range(0, train_size, batch_size):
        temp = i+batch_size

        if temp >= train_size: 
            batch = orders[i:train_size]
        else:
            batch = orders[i:i+batch_size]

        input_train_tensor = input_train_tensors[batch]
        input_train_length_tensor = input_train_lengths_tensors[batch]
        output_train_tensor = output_train_tensors[batch]


        input_train_length_tensor, indices = torch.sort(input_train_length_tensor, descending=True)
        input_train_tensor = input_train_tensor[indices,:]
        output_train_tensor = output_train_tensor[indices,:]


        loss = trainBatch(input_train_tensor, input_train_length_tensor, output_train_tensor, embedding,
                          encoder, decoder, 'train', criterion, encoder_optimizer, decoder_optimizer)

        print('\r' + str(i) + '/' + str(train_size)+', loss:' + str(loss/len(batch)), end='')
        
        total_loss += loss
        
    return total_loss/train_size

def evaluate(input_tensors, input_lengths_tensors, output_tensors, embedding, 
             encoder, decoder, criterion):
    

    batch_size = 32
    total_loss = 0
    train_size, _ = input_tensors.size()
    
    outputs = []
    attns = []
        
    orders = np.arange(train_size)
    
    with torch.no_grad():

        for i in range(0, train_size, batch_size):
            temp = i+batch_size

            if temp >= train_size: 
                batch = orders[i:train_size]

            else:
                batch = orders[i:i+batch_size]

            input_tensor = input_tensors[batch]
            input_length_tensor = input_lengths_tensors[batch]
            output_tensor = output_tensors[batch]


            input_length_tensor, indices = torch.sort(input_length_tensor, descending=True)
            input_tensor = input_tensor[indices,:]
            output_tensor = output_tensor[indices,:]


            loss, output_seq, attn = trainBatch(input_tensor, input_length_tensor, output_tensor, embedding,
                                                encoder, decoder, 'val', criterion)

            print('\r' + 'Evalidation: '+ str(i) + '/' + str(train_size)+', loss:' + str(loss/len(batch)), end='')

            total_loss += loss

            outputs.append(output_seq)
            attns.append(attn)
        
            
    return total_loss/train_size, np.asarray(outputs), np.asarray(attns)



def train(input_train_tensors, input_train_lengths_tensors, output_train_tensors, 
          input_val_tensors, input_val_lengths_tensors, output_val_tensors, 
          embedding, encoder, decoder, 
          epoches=20, batch_size=64, print_every=1, plot_every=1, learning_rate=0.001,
          patience = 3, decay_rate=0.5, early_stop=10):
    
    # input_val, input_val_lengths, output_val,
              
    start = time.time()
    plot_train_loss = []
    plot_val_loss = []
    
    patience_count = 0
    early_stop_count = 0

    encoder_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(),lr=learning_rate)

    criterion = nn.NLLLoss()
    
    train_size = len(input_train_tensors)
    
    best_val_loss = 100
    
    for iter in range(1, epoches + 1):
        print('iter', iter)
            
        train_loss = trainEpoch(input_train_tensors, input_train_lengths_tensors, output_train_tensors, embedding, 
                          encoder, decoder, encoder_optimizer, decoder_optimizer, 
                          criterion, epoches, batch_size)

        val_loss, output_seq, attn = evaluate(input_val_tensors, input_val_lengths_tensors, output_val_tensors, embedding, 
                                              encoder, decoder, criterion)

        if iter % print_every == 0:
            print()
            print('\r' + timeSince(start, iter / epoches) + 
                  ' train loss: %5.3f.  val loss: %5.3f.'
                  % (train_loss, val_loss))
            

        if iter % plot_every == 0:
            plot_train_loss.append(train_loss)
            plot_val_loss.append(val_loss)
            
        
        if val_loss<best_val_loss:
            print('val loss decreases from %5.3f to %5.3f, save models'% (best_val_loss, val_loss))
            torch.save(encoder, './saved_models/encoder.pt')
            torch.save(decoder, './saved_models/decoder.pt')
            torch.save(encoder_optimizer, './saved_models/encoder_optimizer.pt')
            torch.save(decoder_optimizer, './saved_models/decoder_optimizer.pt')
            torch.save(decoder.attn, './saved_models/attn.pt')

            best_val_loss = val_loss

        #showAttention(input_val_tensors[100],output_seq[100], attn[100])
        
        # early stop
        if early_stop_count>= early_stop:
            print()
            print('Early Etop')
            break

        print()  
        
    the_encoder = torch.load('./saved_models/encoder.pt')
    the_decoder = torch.load('./saved_models/decoder.pt')
    the_encoder_optimizer = torch.load('./saved_models/encoder_optimizer.pt')
    the_decoder_optimizer = torch.load('./saved_models/decoder_optimizer.pt')
    the_attn = torch.load('./saved_models/attn.pt')
    
    torch.save('./saved_models/encoder' + str(best_val_loss) + '.pt')
    torch.save('./saved_models/decoder' + str(best_val_loss) + '.pt')
    torch.save('./saved_models/encoder_optimizer' + str(best_val_loss) + '.pt')
    torch.save('./saved_models/decoder_optimizer' + str(best_val_loss) + '.pt')
    torch.save('./saved_models/attn' + str(best_val_loss) + '.pt')
    

    showPlot(plot_train_loss, plot_val_loss)
    
    
    
def asMinutes(s):
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)

def timeSince(since, precent):
    now = time.time()
    s = now - since
    es = s / precent
    rs = es - s 
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(train, val, name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=1)
    ax.xaxis.set_major_locator(loc)
    plt.plot(train, label='train')
    plt.plot(val, label='val')
    plt.legend()
    plt.ylabel(name)
    plt.xlabel("Epoch")


    
def showAttention(input_sentence, output_sentence, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_sentence)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    
def adjust_learning_rate(optimizer, ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] *  ratio

    



