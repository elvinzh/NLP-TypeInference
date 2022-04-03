# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:07:17 2021

@author: longq
"""


import string
import re
import random
import numpy as np

import os

import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN,TimeDistributed
from keras.layers import Embedding
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras import activations
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.layers import Bidirectional
from keras.models import load_model



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



# f = open("0100.ml", "r")
f = open("0107.ml", "r")
s = f.read()


# parts1 = s.split()
# parts2 = re.split('\s+', s)

# print(f.read())

# f.close()


# def label

# def span_to_words(span, s):
#     sl = s.splitlines()
#     sp = re.findall(r'\d+', span)
#     span_ = [int(d)-1 for d in sp]
#     row1, col1, row2, col2 = span_
#     # words = []
#     if row1 == row2:
        
#         words = sl[row1][col1:col2]
#     if row1 < row2:
#         diff = row2 - row1
#         words.append
        


def proccessing1(s, err = 'type error'):
    sl = s.splitlines()
    l_len = []
    prog = ''
    for i in sl:
        if "(*" in i:
            break
        l_len.append(len(i))
        prog += i
    start = 0
    end = 0
    for i in range(len(sl)):
        if err in sl[i]:
            start = i
    cut = sl[start+1:]
    for i in range(len(cut)):
        if '*)' in cut[i]:
            end = i
            break
    cut_s = cut[:end]
    lidx = []
    for span in cut_s:
        sp = re.findall(r'\d+', span)
        span_ = [int(d) for d in sp]
        row1, col1, row2, col2 = span_
        s = convert_idx((row1, col1),l_len)
        e = convert_idx((row2, col2),l_len)
        lidx.append([s,e])
    merged = merge_intervals(lidx)
    return prog, merged
    


def proccessing2(s, err = 'changed spans'):
    sl = s.splitlines()
    l_len = []
    prog = ''
    for i in sl:
        if "(*" in i:
            break
        l_len.append(len(i))
        prog += i
    start = 0
    end = 0
    for i in range(len(sl)):
        if err in sl[i]:
            start = i
    cut = sl[start+1:]
    for i in range(len(cut)):
        if '*)' in cut[i]:
            end = i
            break
    cut_s = cut[:end]
    lidx = []
    for span in cut_s:
        sp = re.findall(r'\d+', span)
        span_ = [int(d) for d in sp]
        row1, col1, row2, col2 = span_
        s = convert_idx((row1, col1),l_len)
        e = convert_idx((row2, col2),l_len)
        lidx.append([s,e])
    merged = merge_intervals(lidx)
    return prog, merged





def proccessing(s):
    prog, error = proccessing1(s, err = 'type error')
    prog, fix = proccessing2(s, err = 'changed spans')
    return prog, error, fix





def convert_idx(t,l):
    r,c = t
    idx = sum(l[:r-1]) + c-1
    return idx
    
    
def merge_intervals(temp_tuple):
    temp_tuple.sort(key=lambda interval: interval[0])
    merged = [temp_tuple[0]]
    for current in temp_tuple:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged


def merge(intervals):
    starts = intervals[:,0]
    ends = np.maximum.accumulate(intervals[:,1])
    valid = np.zeros(len(intervals) + 1, dtype=np.bool)
    valid[0] = True
    valid[-1] = True
    valid[1:-1] = starts[1:] >= ends[:-1]
    return np.vstack((starts[:][valid[:-1]], ends[:][valid[1:]])).T

# #example
# a=[]
# a.append([1,3])
# a.append([4,10])
# a.append([5,12])
# a.append([6,8])
# a.append([20,33])
# a.append([30,35])

# b = np.array(a)

# print("intervals")
# print(b)
# print()
# print("merged intervals")
# print(merge_intervals(b))
    


# def filter(lines):
#     # word = re.split('\s+', lines)
#     # return word
#     sentences = re.split(r' *[\.][\'"\)\]]* *', lines)
#     for stuff in sentences:
#         x = re.search('^\w+([-]?\w+)*\w+([\.()-]?\w+)*\w+([-]?\w+)+$',stuff)
#         if x:
#           res = x
#         else:
#           stuff = re.sub('([.,!?()])', r' \1 ', stuff)
#           stuff = re.sub('\txt{2,}', ' ', stuff)
#           res = stuff
#     return res
    

def word_filter0(w):
    ins = []
    if w[-1] == 't' and w[-2] == 'e' and w[-3] == 't':
        ins.append(w[:-3])
        ins.append('let')
    else:
        ins = [w]
    return ins





def word_filter1(w):
    # print(w)
    ins = []
    if len(w) >= 2:
        if w != '':
            if w[0] == '(':
                ins += ['(']
                if w[1] == '(':
                    ins += word_filter1(w[1:])
                else:
                    ins += [w[1:]]
            else:
                ins = [w]
        # if w[-1] == ')':
        #     ins += [')']
        #     if w[-2] == ')':
        #         ins += word_filter(w[:-1])
        #     else:
        #         ins += [w[:-1]]
        else:
            ins = [w]
    else:
        ins = [w]
    return ins


def word_filter2(w):
    def word_f2(w):
        ins = []
        if len(w) >= 2:
            if w[-1] == ')':
                ins += [')']
                if w[-2] == ')':
                    ins += word_f2(w[:-1])
                else:
                    ins += [w[:-1]]
        else:
            ins = [w]
        return ins
    if w != '':
        if w[-1] == ')': 
            ins = word_f2(w)[::-1]
        else:
            ins = [w]
    else:
        ins = [w]
    return ins


def word_filter3(w):
    ins = []
    if w[-1] == ';' and w[-2] == ';':
        ins.append(w[:-2])
        ins.append(';;')
    else:
        ins = [w]
    return ins




def filter_middle(w):
    ins = []
    if ',' in w:
        i = w.index(',')
        ins.append(w[:i])
        ins.append(',')
        ins.append(w[i+1:])
        return ins
    if '::' in w:
        i = w.index(':')
        ins.append(w[:i])
        ins.append('::')
        ins.append(w[i+2:])
        return ins
    if ';;' in w:
        i = w.index(';')
        ins.append(w[:i])
        ins.append(';;')
        ins.append(w[i+2:])
        return ins
    else:
        return [w]
    
# def filter_semicolon(w):
#     ins = []
#     for i in range(w):
                


def filter0(words):
    filtered = []
    for w in words:
        filtered += filter_middle(w)
    return filtered


def filter1(words):
    filtered = []
    for w in words:
        filtered += word_filter1(w)
    return filtered
    
def filter2(words):
    filtered = []
    for w in words:
        filtered += word_filter2(w)
    return filtered



def filter_(words):
    return filter2(filter1(filter0(words)))


def get_words(prog, merged):
    parts = []
    tracker = []
    sp = 0
    for i in range(len(merged)):
        tup = merged[i]
        parts.append(prog[sp:tup[0]])
        tracker.append(0)
        parts.append(prog[tup[0]:tup[1]])
        tracker.append(1)
        sp = tup[1]
    parts.append(prog[sp:])
    tracker.append(0)
    
    words = []
    label = []
    for i in range(len(parts)):
        # ps = filter(parts[i])
        ps = parts[i]
        wo = re.split('\s+', ps)
        # re.split('\s+', s)
        word = filter_(wo)
        words += word
        if tracker[i] == 0:
            label += [0.0 for w in range(len(word))]
        if tracker[i] == 1:
            label += [1.0 for w in range(len(word))]
            
    
    words_ = []
    label_ = []        
    for i in range(len(words)):
        if '' != words[i]:
            words_.append(words[i])
            label_.append(label[i])
            
    return words_,label_
        

#keywordpath1 = "C:/Users/longq/OneDrive/Desktop/keywords1.txt"
#keywordpath2 = "C:/Users/longq/OneDrive/Desktop/keywords2.txt"
keywordpath1 = "keywords1.txt"
keywordpath2 = "keywords2.txt"


with open(keywordpath1, 'r') as f:
    keyword1 = f.read()
    key1 = keyword1.split()
    f.close()
    
with open(keywordpath2, 'r') as f:
   keyword2 = f.read()
   key2 = keyword2.split()
   f.close()   

key = key1 + key2



def keywords_feat(words, key):
    ft = []
    for w in words:
        if w in key:
            ft.append(1)
        else:
            ft.append(0)
    return ft



len_keep = 75
keep_prob = 0.75

train1 = []
label1 = []
err1 = []
ft1 = []
path = "/data/sp14"
try:
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as f:
            s = f.read()
            prog, error, fix = proccessing(s)
            words_,label_ = get_words(prog, fix)
            words_,error_ = get_words(prog, error)
            ft_ = keywords_feat(words_, key)
            if len(words_) < len_keep:
                if random.random() < keep_prob: 
                    train1.append(words_)
                    err1.append(error_)
                    label1.append(label_)
                    ft1.append(ft_)
                    
            else:
                train1.append(words_)
                err1.append(error_)
                label1.append(label_)
                ft1.append(ft_)
            f.close()
except:
    pass
       

train2 = []
label2 = []
err2 = []
path = "/data/fa15"

try:
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as f:
            s = f.read()
            prog, error, fix = proccessing(s)
            words_,label_ = get_words(prog, fix)
            words_,error_ = get_words(prog, error)
            if len(words_) < len_keep:
                if random.random() < keep_prob: 
                    train2.append(words_)
                    err2.append(error_)
                    label2.append(label_)
                    
            else:
                train2.append(words_)
                err2.append(error_)
                label2.append(label_)
                    
            f.close()      
except:
    pass



# statistics

def statistics(label):
    type_error_percentage = []
    len_ = []
    for l in label:
        len_.append(len(l))
        per = round(sum(l)/len(l),2)
        type_error_percentage.append(per)
    return type_error_percentage, len_
        
t1,l1 = statistics(label1)
t2,l2 = statistics(label2)     
        
n_bins = 20


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)

ax1.hist(t1, bins=n_bins, color = "pink")
ax1.set_ylabel("Count")
ax1.set_xlabel("Percentage")
ax1.set_title('sp14')

ax2.hist(t2, bins=n_bins, color = "pink")
ax2.set_xlabel("Percentage")
ax2.set_title('fa15')
fig.suptitle('Type errors percentage')


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)

ax1.hist(l1, bins=n_bins, color = "skyblue")
ax1.set_ylabel("Count")
ax1.set_xlabel("Length")
ax1.set_title('sp14')

ax2.hist(l2, bins=n_bins, color = "skyblue")
ax2.set_xlabel("Length")
ax2.set_title('fa15')
fig.suptitle('Porgram length')





# train = train1 + train2
# label = label1 + label2


train = train1
err = ft1
label = label1

# train = train2
# err = err2
# label = label2


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 5000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 150
# This is fixed.
EMBEDDING_DIM = 100



tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='',)
tokenizer.fit_on_texts(train)
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(train)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
print('Shape of data tensor:', X.shape)


E = pad_sequences(err, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


len_ = len(label)
Y = pad_sequences(label, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
print('Shape of label tensor:', Y.shape)

tf.cast(Y, tf.float32)


from sklearn.model_selection import train_test_split

X_train, X_test, E_train, E_test, Y_train, Y_test = train_test_split(X, E, Y, test_size = 0.10, random_state = 23)

# order_  = list(range(len(train)))
# random.shuffle(order_)


print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)







optimizer = Adam(clipnorm=.5)
# optimizer = Adam()

# MAX_NB_WORDS = 50000
# Max number of words in each complaint.

# This is fixed.
# EMBEDDING_DIM = 100

# model = Sequential()
# model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
# model.add(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True))
# model.add(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation ='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# epochs = 10
# batch_size = 128


# class_weight = {0: 1.,
#                 1: 1.}

# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weight, validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])




# from sklearn.utils import class_weight

# weights = class_weight.compute_class_weight('balanced',
#                                             np.unique(Y_train),
#                                             Y_train)

# # Add the class weights to the training                                         
# model.fit(x_train, y_train, epochs=10, batch_size=32, class_weight=weights)



import keras


from keras import backend as K
def weighted_binary_crossentropy( y_true, y_pred, weight=1. ):
    y_true = tf.cast(y_true, tf.float32)
    logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
    return K.mean( logloss, axis=-1)
                                
def dyn_weighted_bincrossentropy(true, pred):
    """
    Calculates weighted binary cross entropy. The weights are determined dynamically
    by the balance of each category. This weight is calculated for each batch.
    
    The weights are calculted by determining the number of 'pos' and 'neg' classes 
    in the true labels, then dividing by the number of total predictions.
    
    For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
    These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
    1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.
    
    This can be useful for unbalanced catagories.
    """
    # get the total number of inputs
    num_pred = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) + keras.backend.sum(true)
    
    # get weight of values in 'pos' category
    zero_weight =  keras.backend.sum(true)/ num_pred +  keras.backend.epsilon() 
    
    # get weight of values in 'false' category
    one_weight = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) / num_pred +  keras.backend.epsilon()

    # calculate the weight vector
    weights =  (1.0 - true) * zero_weight +  true * one_weight 
    
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return keras.backend.mean(weighted_bin_crossentropy)


def weighted_bincrossentropy(true, pred, weight_zero = 0.25, weight_one = 1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.
        
    This can be useful for unbalanced catagories.
    
    Adjust the weights here depending on what is required.
    
    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
        will be penalize 10 times as much as false negatives.
    """
  
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    # apply the weights
    
    # weights = np.float32(true) * weight_one + (1. - true) * weight_zero
    # weighted_bin_crossentropy = weights * bin_crossentropy 

    # return keras.backend.mean(weighted_bin_crossentropy)
    return bin_crossentropy





epochs = 30
batch_size = 64


#  So obviously, LSTM itself does not need a TimeDistributed wrapper.

# rnn
optimizer = Adam(clipnorm=.5)

MAX_NB_WORDS = 5000
EMBEDDING_DIM = 100

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SimpleRNN(256, activation = 'relu', dropout=0.1, return_sequences=True))
# model.add(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation ='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# history11 = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
history21 = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

model.save('rnn1.h5')  
# model.save('rnn2.h5')  




# lstm 
optimizer = Adam(clipnorm=.5)
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True))
# model.add(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation ='sigmoid'))
model.compile(loss= 'binary_crossentropy' , optimizer=optimizer, metrics=['accuracy'])

# history12 = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

history22 = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

model.save('lstm1.h5')  
# model.save('lstm2.h5') 





# Add Bidirectional
optimizer = Adam(clipnorm=.5)
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(Bidirectional(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation ='sigmoid'))
model.compile(loss= 'binary_crossentropy' , optimizer=optimizer, metrics=['accuracy'])


# history13 = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

history23 = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# history3 = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
model.save('bidir_lstm1.h5')
# model.save('bidir_lstm2.h5')




# Add the class weights to the training     





    
# Compile the model

# optimizer = Adam(clipnorm=.5)
# model = Sequential()
# model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
# model.add(Bidirectional(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation ='sigmoid'))
# model.compile(loss=weighted_binary_crossentropy , optimizer=optimizer, metrics=['accuracy'])

# epochs = 20
# batch_size = 128


# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])





# evaluation

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))




pred_ = model.predict(X_test)

# idd = 8
# idd = 55


# idd = 66

idd = 18



# idd = 9
predi = pred_[idd]
preds = ((predi > 0.5) * 1).reshape(MAX_SEQUENCE_LENGTH,)
print(preds)

print(Y_test[idd])


def predonid(model, X_test, Y_test, id = 18):
    pred_ = model.predict(X_test)
    predi = pred_[idd]
    preds = ((predi > 0.5) * 1).reshape(MAX_SEQUENCE_LENGTH,)
    return preds,Y_test[idd]
    


def intersection_union(ipt):
    try:
        pred,label = ipt
        s_ = pred + label
        i_ = sum(s_==2)
        u_ = sum(s_==1)
    except:
        print(i_)
        print(u_)
    return i_ , u_




# acc_id = intersection_union(predonid(model, X_test, Y_test, id = 18))

def acc_average(model, X_test, Y_test):
    l_ = len(X_test)
    sum_ = 0
    pred_ = model.predict(X_test)
    # l1 = []
    # l2 = []
    # l3 = []
    for i in range(len(X_test)):
        predi = pred_[i]
        preds = ((predi > 0.5) * 1).reshape(MAX_SEQUENCE_LENGTH,)
        i_ , u_ = intersection_union((preds,Y_test[i]))
        if u_ == 0:
            sum_ += 0
        else:
            sum_ += (i_/(i_+u_) > 0.5)*1
    return sum_/l_

    
def acc_average(model, X_test, E_test, Y_test):
    l_ = len(X_test)
    sum_ = 0
    pred_ = model.predict([X_test, E_test])
    # l1 = []
    # l2 = []
    # l3 = []
    for i in range(len(X_test)):
        predi = pred_[i]
        preds = ((predi > 0.5) * 1).reshape(MAX_SEQUENCE_LENGTH,)
        i_ , u_ = intersection_union((preds,Y_test[i]))
        if u_ == 0:
            sum_ += 0
        else:
            sum_ += (i_/(i_+u_) > 0.5)*1
    return sum_/l_







def acc_list(model, X_test, Y_test):
    li = []
    pred_ = model.predict(X_test)
    # l1 = []
    # l2 = []
    # l3 = []
    for i in range(len(X_test)):
        predi = pred_[i]
        preds = ((predi > 0.5) * 1).reshape(MAX_SEQUENCE_LENGTH,)
        i_ , u_ = intersection_union((preds,Y_test[i]))
        if u_ == 0:
            li.append(0) 
        else:
            li.append(i_/(i_+u_))
    return li 

acc_ = acc_average(model, X_test, Y_test)

acc2 = acc_average(model, X_test,E_test, Y_test)

acc3 = acc_average(model, X_test,E_test, Y_test)


model_rnn1 = load_model('rnn1.h5')
model_lstm1 = load_model('lstm1.h5')
model_bidir1 = load_model('bidir_lstm1.h5')

acc_rnn1 = acc_average(model_rnn1 , X_test, Y_test)
acc_lstm1 = acc_average(model_lstm1 , X_test, Y_test)
acc_bidir1 = acc_average(model_bidir1 , X_test, Y_test)



li_ = acc_list(model_bidir1, X_test, Y_test)

# plt.title('Loss')
# plt.plot(history1.history['loss'], label='train')
# plt.plot(history1.history['val_loss'], label='test')
# plt.legend()
# plt.show();


plt.title('Accuracy for sp14')
plt.plot(history11.history['loss'], label='rnn')
# plt.plot(history11.history['val_accuracy'], label='test')
plt.plot(history12.history['loss'], label='lstm')
# plt.plot(history12.history['val_accuracy'], label='test')
plt.plot(history13.history['loss'], label='bidir_lstm')
# plt.plot(history13.history['val_accuracy'], label='test')
plt.legend()
plt.show();






model_rnn2 = load_model('rnn2.h5')
model_lstm2 = load_model('lstm2.h5')
model_bidir2 = load_model('bidir_lstm2.h5')

acc_rnn2 = acc_average(model_rnn2 , X_test, Y_test)
acc_lstm2 = acc_average(model_lstm2 , X_test, Y_test)
acc_bidir2 = acc_average(model_bidir2 , X_test, Y_test)

plt.title('Accuracy for fa15')
plt.plot(history21.history['accuracy'], label='rnn')
# plt.plot(history11.history['val_accuracy'], label='test')
plt.plot(history22.history['accuracy'], label='lstm')
# plt.plot(history12.history['val_accuracy'], label='test')
plt.plot(history23.history['accuracy'], label='bidir_lstm')
# plt.plot(history13.history['val_accuracy'], label='test')
plt.legend()
plt.show();











fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)

ax1.plot(history11.history['loss'], label='rnn')
# plt.plot(history11.history['val_accuracy'], label='test')
ax1.plot(history12.history['loss'], label='lstm')
# plt.plot(history12.history['val_accuracy'], label='test')
ax1.plot(history13.history['loss'], label='bidir_lstm')
ax1.legend()
ax1.set_ylabel("Loss")
ax1.set_xlabel("Epochs")
ax1.set_title('sp14')

ax2.plot(history21.history['loss'], label='rnn')
# plt.plot(history11.history['val_accuracy'], label='test')
ax2.plot(history22.history['loss'], label='lstm')
# plt.plot(history12.history['val_accuracy'], label='test')
ax2.plot(history23.history['loss'], label='bidir_lstm')
ax2.legend()
ax2.set_xlabel("Epochs")
ax2.set_title('fa15')

fig.suptitle('Learning curves')












from keras.models import Model

# ip1 = layers.Input((9000))
# ip2 = layers.Input((9000))
# word_embed = layers.Embedding(90000, 100, input_length=9000)(ip1)
# feat_embed = layers.Embedding(90000, 8, input_length=9000)(ip2)

# layerlist = [x_embed, feat_embed]
# concat = layers.Concatenate(axis = -1)(layerlist)

# model = models.Model([ip1, ip2], concat)
# model.summary()




optimizer = Adam(clipnorm=.5)
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 100
word_ = layers.Input((MAX_SEQUENCE_LENGTH))
feat_ = layers.Input((MAX_SEQUENCE_LENGTH))
word_embed = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH )(word_)
feat_embed = Embedding(MAX_NB_WORDS, 1, input_length=MAX_SEQUENCE_LENGTH )(feat_)
layerlist = [word_embed, feat_embed]
concat = layers.Concatenate(axis = -1)(layerlist)
l1 = SimpleRNN(256, activation = 'relu', dropout=0.1, return_sequences=True)(concat)
# model.add(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True))
l2 = Dense(64, activation='relu')(l1)
l3 = Dense(1, activation='sigmoid')(l2)
model = Model([word_,feat_],l3)


model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history1 = model.fit( [X_train, E_train], Y_train, epochs=epochs, batch_size=batch_size)


model.save('rnn1.h5')  
model = load_model('rnn1.h5')





optimizer = Adam(clipnorm=.5)
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 100
word_ = layers.Input((MAX_SEQUENCE_LENGTH))
feat_ = layers.Input((MAX_SEQUENCE_LENGTH))
word_embed = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH )(word_)
feat_embed = Embedding(MAX_NB_WORDS, 1, input_length=MAX_SEQUENCE_LENGTH )(feat_)
layerlist = [word_embed, feat_embed]
concat = layers.Concatenate(axis = -1)(layerlist)
l1 = LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True)(concat)
# model.add(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True))
l2 = Dense(64, activation='relu')(l1)
l3 = Dense(1, activation='sigmoid')(l2)
model = Model([word_,feat_],l3)


model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history1 = model.fit( [X_train, E_train], Y_train, epochs=epochs, batch_size=batch_size)


model.save('LSTM1.h5')  
model = load_model('LSTM1.h5')



optimizer = Adam(clipnorm=.5)
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 100
word_ = layers.Input((MAX_SEQUENCE_LENGTH))
feat_ = layers.Input((MAX_SEQUENCE_LENGTH))
word_embed = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH )(word_)
feat_embed = Embedding(MAX_NB_WORDS, 1, input_length=MAX_SEQUENCE_LENGTH )(feat_)
layerlist = [word_embed, feat_embed]
concat = layers.Concatenate(axis = -1)(layerlist)

l0 = Bidirectional(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True))(concat)
l1 = LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True)(l0)
# model.add(LSTM(256, activation = 'relu', dropout=0.1, return_sequences=True))
l2 = Dense(64, activation='relu')(l1)
l3 = Dense(1, activation='sigmoid')(l2)
model = Model([word_,feat_],l3)


model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history1 = model.fit( [X_train, E_train], Y_train, epochs=epochs, batch_size=batch_size)


model.save('bi_LSTM_ft.h5')  
model = load_model('bi_LSTM_ft.h5')



# model.save('bi_LSTM_slice.h5')  
# model = load_model('bi_LSTM_slice.h5')










import matplotlib.pyplot as plt
import seaborn as sns


idx = 45


ex = pred_[idx : idx+10,:,:]
ex = ex.reshape(10,150)
sns.heatmap(ex)
plt.ylabel("Sample")
plt.xlabel("Token")
plt.title("The prediction heatmap")
plt.show()



lex = Y_test[idx : idx+10,:]
lex  = lex .reshape(10,150)
sns.heatmap(lex )
plt.ylabel("Sample")
plt.xlabel("Token")
plt.title("The label heatmap")
plt.show()











from matplotlib.pyplot import figure

labels = ['Rnn', 'Lstm', 'Bidir','OCamel', 'SHErrLoc', 'Logistic', 'Tree', 'Forest', 'MLP-500']
sp14 = [acc_rnn1, acc_lstm1, acc_bidir1, 0.42,0.55,0.62,0.695,0.705,0.72]
fa15 = [acc_rnn2, acc_lstm2, acc_bidir2, 0.39,0.56,0.6,0.685,0.695,0.7]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

figure(figsize=(18, 6))
fig, ax = plt.subplots(figsize=(18,6))

rects1 = ax.bar(x - width/2, sp14, width, label='sp14')
rects2 = ax.bar(x + width/2, fa15, width, label='sp15')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy comparsion')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()







pred_ = model_bidir1.predict(X_test)

# idd = 8
# idd = 55


# idd = 66

idd = 18



# idd = 9
predi = pred_[idd]
preds = ((predi > 0.5) * 1).reshape(MAX_SEQUENCE_LENGTH,)
print(preds)

print(Y_test[idd])




#keywordpath1 = "C:/Users/longq/OneDrive/Desktop/keywords1.txt"
#keywordpath2 = "C:/Users/longq/OneDrive/Desktop/keywords2.txt"
keywordpath1 = "keywords1.txt"
keywordpath2 = "keywords2.txt"


with open(keywordpath1, 'r') as f:
    keyword1 = f.read()
    key1 = keyword1.split()
    f.close()
    
with open(keywordpath2, 'r') as f:
   keyword2 = f.read()
   key2 = keyword2.split()
   f.close()   

key = key1 + key2

