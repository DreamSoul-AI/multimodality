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
embedding_dim = 64
hidden_size = 32
num_layers = 1
learning_rate = 0.01
epochs = 50
max_iters = 5000
block_size = 128
eval_iters = 200
eval_interval = 100
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
def load_data(tokenizer, path, total_amount):
    data = []
    total = 0
    count = 0
    for j in os.listdir(path):
        if ".ipynb" in j:
            continue
        tmp = os.path.join(path,j)
        total += os.path.getsize(tmp)
        f = open(tmp, mode="rb")
        f=transform_bits(f)
        data.append(np.array(f))
        count+=1
        if count==total_amount:
            break
    return data,total

def removeNestings(l):
    lis = []
    for i in l:
        if type(i) == list:
            lis+=removeNestings(i)
        else:
            lis.append(i)
    return lis



# processing training data

def split(datas,len_datas,split_rate = 0.8):
    l = int(len(datas)*(split_rate))
    return datas[:l], len_datas[:l],datas[l:], len_datas[l:]
def generate_dataset(data,block_size, repeat_time = 2):
    input = []
    target = []
    for i in range(len(data)):
        for j in range(repeat_time):
            index = random.randint(0,len(data[i])-block_size-2)
            input.append(torch.from_numpy(np.array(data[i][index:index+block_size])))
            target.append(torch.from_numpy(np.array(data[i][index:index+block_size])))
    return input, target