import string
import re
import random
import numpy as np

import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import transformers
from transformers import BertForTokenClassification, AdamW, BertConfig
from seqeval.metrics import f1_score, accuracy_score
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup




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

def keywords_feat(words, key):
    ft = []
    for w in words:
        if w in key:
            ft.append(1)
        else:
            ft.append(0)
    return ft


def loadData():
    len_keep = 75
    keep_prob = 0.75

    train1 = []
    label1 = []
    err1 = []
    ft1 = []
    path = data_path + "/sp14"
    try:
        for filename in os.listdir(path):
            if '.ml' not in filename:
                continue
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
    path = data_path + "/fa15"

    try:
        for filename in os.listdir(path):
            if '.ml' not in filename:
                continue
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
    return train1, label1, train2, label2


def statistics(label):
    type_error_percentage = []
    len_ = []
    for l in label:
        len_.append(len(l))
        per = round(sum(l)/len(l),2)
        type_error_percentage.append(per)
    return type_error_percentage, len_


def visualization():
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
    
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

data_path = "./data"
print(data_path)

keywordpath1 = data_path + "/keywords1.txt"
keywordpath2 = data_path + "/keywords2.txt"


with open(keywordpath1, 'r') as f:
    keyword1 = f.read()
    key1 = keyword1.split()
    f.close()
    
with open(keywordpath2, 'r') as f:
   keyword2 = f.read()
   key2 = keyword2.split()
   f.close()   

key = key1 + key2

train1, label1, train2, label2 = loadData()

MAX_LEN = 150
bs = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(train1, label1)
]

tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

input_ids[0]

tag_values = list(set(label1[0]))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)

# from transformers import RobertaForTokenClassification
# model = RobertaForTokenClassification.from_pretrained(
#     "microsoft/codebert-base",
#     num_labels=len(tag2idx),
#     output_attentions = False,
#     output_hidden_states = False
# )


model.cuda();

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

epochs = 40
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    
    pred = np.array(predictions)
    y = np.array(true_labels)
    pad_mask = np.array(true_labels)
    pad_mask[pad_mask==0] = 1
    pad_mask[pad_mask==2] = 0
    pred = pred & pad_mask
    y[y==2]=0
    
    intersection = pred & y
    union = pred | y
    
    nonzero_intersection = torch.count_nonzero(torch.from_numpy(intersection), dim=1)
    nonzero_union = torch.count_nonzero(torch.from_numpy(union), dim=1)
    nonzero_intersection = torch.where(nonzero_union == 0, 1, nonzero_intersection)
    nonzero_union[nonzero_union==0] = 1

    acc = nonzero_intersection / nonzero_union
    
    print("Test Acc = ", torch.mean(acc))
    
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
#     print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()