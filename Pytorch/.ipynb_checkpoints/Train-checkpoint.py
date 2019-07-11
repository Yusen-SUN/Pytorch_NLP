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


def trainBatch(input_tensors, target_tensors,
               model, mode, criterion=None,
               optimizer=None):
    
    # input_tensors (batch, seq)
    # input_lengths (batch)

    loss = 0
    
    if mode=='train':
        model.train()
            
        optimizer.zero_grad()
        
    elif mode=='val':
        model.eval()
    
    else:
        print('wrong mode')
        
    
    # model_outputs: batch, num_labels
    model_outputs = model.forward(input_tensors)
    _, predicted_labels = torch.max(model_outputs, 1)

    
    batch_size = input_tensors.size()[0]
        
    for i in range(batch_size):
        
        loss += criterion(model_outputs, target_tensors)
        
    if mode=='train':
        loss.backward()
        optimizer.step()        
        return loss.item(), predicted_labels
    
    else:
        return loss.item(), predicted_labels.detach()
    
    
    
def trainEpoch(input_train_tensors, output_train_tensors, model, optimizer, criterion, epoches, batch_size):
    
    # input_train_tensors: batch, seq
    
    train_size = len(input_train_tensors)

    orders = np.arange(train_size)
    np.random.shuffle(orders)
    total_loss = 0
    predicted_labels = []

    for i in range(0, train_size, batch_size):
        temp = i+batch_size

        if temp >= train_size: 
            batch = orders[i:train_size]
        else:
            batch = orders[i:i+batch_size]

        input_train_tensor = input_train_tensors[batch]
        output_train_tensor = output_train_tensors[batch]

        loss, predicted_label = trainBatch(input_train_tensor, output_train_tensor, model, 'train', criterion, optimizer)
        
        predicted_labels.append(predicted_label)

        print('\r' + str(i) + '/' + str(train_size)+', loss:' + str(loss/len(batch)), end='')
        
        total_loss += loss
        
    predicted_labels = torch.cat(predicted_labels)
    
    acc = (predicted_labels == output_train_tensors[orders]).sum().item() / len(output_train_tensors)
    
    return total_loss/train_size, acc

def evaluate(input_tensors, output_tensors, model, criterion):
    

    batch_size = 32
    total_loss = 0
    train_size, _ = input_tensors.size()
    predicted_labels = []

    orders = np.arange(train_size)
    
    with torch.no_grad():
        for i in range(0, train_size, batch_size):
            temp = i+batch_size

            if temp >= train_size: 
                batch = orders[i:train_size]

            else:
                batch = orders[i:i+batch_size]

            input_tensor = input_tensors[batch]
            output_tensor = output_tensors[batch]


            loss, predicted_label = trainBatch(input_tensor, output_tensor, model, 'val', criterion)

            print('\r' + 'Evalidation: '+ str(i) + '/' + str(train_size)+', loss:' + str(loss/len(batch)), end='')
            
            predicted_labels.append(predicted_label)

            total_loss += loss
            
    predicted_labels = torch.cat(predicted_labels)
    
    acc = (predicted_labels == output_tensors[orders]).sum().item() / len(output_tensors)
            
    return total_loss/train_size, acc



def train(input_train_tensors, output_train_tensors, input_val_tensors, output_val_tensors,
          model, epoches=20, batch_size=50, print_every=1, plot_every=1, learning_rate=0.001, patience = 3, decay_rate=0.5, early_stop=10):
    
    # input_val, input_val_lengths, output_val,
              
    start = time.time()
    plot_train_loss = []
    plot_val_loss = []
    plot_train_acc = []
    plot_val_acc = []

    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay = 1e-3)
    criterion = nn.NLLLoss()
    
    train_size = len(input_train_tensors) 
    best_val_acc = 0
    patience_count = 0
    early_stop_count = 0
   
    for iter in range(1, epoches + 1):
        print('iter', iter, '/', epoches )
            
        train_loss, train_acc = trainEpoch(input_train_tensors, output_train_tensors, model, optimizer, criterion, epoches, batch_size)

        val_loss, val_acc = evaluate(input_val_tensors, output_val_tensors, model, criterion)

        if iter % print_every == 0:
            print('\r' + timeSince(start, iter / epoches) + 
                  ' train loss: %5.3f.  val loss: %5.3f.  train acc: %5.3f.  val acc: %5.3f'
                  % (train_loss, val_loss, train_acc, val_acc))
            
        if iter % plot_every == 0:
            plot_train_loss.append(train_loss)
            plot_val_loss.append(val_loss)
            plot_train_acc.append(train_acc)
            plot_val_acc.append(val_acc)
            
        # save best model
        if val_acc>best_val_acc:
            print('val acc increase from %5.3f to %5.3f'% (best_val_acc, val_acc))
            print('save models')
            torch.save(model, './saved_models/cnn_model.pt')
            torch.save(optimizer, './saved_models/cnn_optimizer.pt')
            
            best_val_acc = val_acc
            
        else:
            patience_count+=1
            early_stop_count +=1
            
        # adjust learning rate
        if patience_count>=patience:
            adjust_learning_rate(optimizer, decay_rate)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print('learning rate deceases from ' + str(lr) + ' to ' + str(lr*decay_rate))
            
            patience_count=0
        
        # early stop
        if early_stop_count>-= early_stop:
            break
            
        
            
        print()
    
    the_model = torch.load('./saved_models/cnn_model.pt')
    the_optimizer = torch.load('./saved_models/cnn_optimizer.pt')
    
    torch.save(the_model, './saved_models/cnn_model' + str(best_val_acc) + '.pt')
    torch.save(the_optimizer, './saved_models/cnn_optimizer' + str(best_val_acc) + '.pt')
    
        
    showPlot(plot_train_loss, plot_val_loss, 'loss')
    showPlot(plot_train_acc, plot_val_acc, 'accuracy')
    
    
    
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
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    plt.plot(train, label='train')
    plt.plot(val, label='val')
    plt.legend()
    plt.ylabel(name)
    plt.xlabel("Epoch")
    #plt.title(name)

def adjust_learning_rate(optimizer, ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] *  ratio
        
        
    



