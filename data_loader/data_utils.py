import matplotlib.pyplot as plt
import os
import numpy as np
from bitarray import bitarray
from os.path import exists
import os
import torch
import itertools
import random
lst = list(itertools.product([0, 1], repeat=8))
bits2token = {}
token2bits = {}
for i in lst:
    token2bits[len(bits2token)] = tuple(i)
    bits2token[tuple(i)] = len(bits2token)

def transform_bits(file):
    ba = bitarray()
    ba.fromfile(file)
    img_bytes = np.array(list(ba)).reshape(-1,8)
    tmp=[]
    for j in img_bytes:
        tmp.append(bits2token[tuple(j)])
    return tmp
def transform_tokens(tokens):
    tmp = []
    for i in tokens:
        tmp = tmp+list(token2bits[i])
    return tmp
def load_data(path,sample_num):
    data = []
    total = 0
    count = 0
    len_datas=[]
    for j in os.listdir(path):
        if ".ipynb" in j:
            continue
        tmp = os.path.join(path,j)
        total += os.path.getsize(tmp)
        f = open(tmp, mode="rb")
        f=transform_bits(f)
        len_datas.append(len(f))
        data.append(np.array(f))
        count+=1
        if count==sample_num:
            break
    
    
    return data,total/1024/1024,len_datas
def tokenize(tokenizer,datas, vocab_size):
    data_for_tokenzier = np.concatenate(datas,axis = 0).tolist()

    tokenizer.train(data_for_tokenzier, vocab_size,verbose =True)
    for i in range(len(datas)):
        datas[i] = np.array(tokenizer.encode(datas[i]))
    return datas
def removeNestings(l):
    lis = []
    for i in l:
        if type(i) == list:
            lis+=removeNestings(i)
        else:
            lis.append(i)
    return lis

# processing training data
def padding(datas, value):
    len_datas = []
    max_len = 0
    for i in datas:
        len_datas.append(len(i))
        max_len = max(max_len,len_datas[-1])

    for i in range(len(datas)):
        datas[i] = np.pad(datas[i],(0,max_len-len(datas[i])),mode = "constant",constant_values =value)

    return np.stack(datas,axis=0),max_len

def split(datas,split_rate = 0.8):
    l = int(len(datas)*(split_rate))
    return datas[:l], datas[l:]
def generate_dataset(data,block_size, repeat_time = 2):
    input = []
    target = []
    for i in range(len(data)):
        for j in range(repeat_time):
            index = random.randint(0,len(data[i])-block_size-2)
            input.append(torch.from_numpy(np.array(data[i][index:index+block_size])))
            target.append(torch.from_numpy(np.array(data[i][index:index+block_size])))
    return input, target
def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')