import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import base64
from bitarray import bitarray
from os.path import exists
import os
import torch
import itertools
import random

root_dataset_path = 'dataset/'

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

# Downloading Mnist as image files
def load_mnist():
    if not os.path.exists(root_dataset_path):
        os.makedirs(root_dataset_path)
        train_data = torchvision.datasets.MNIST("./", train=True, transform=torchvision.transforms.ToTensor(), download=True)
        for i,data in enumerate(train_data):
            img = data[0]
            label = data[1]
            son_path = root_dataset_path+str(label)+"/"
            if not os.path.exists(son_path):
                os.makedirs(son_path)
            plt.imsave(son_path+str(len(os.listdir(son_path)))+'.png',img.reshape(28,28), cmap="gray")


# BPE encoding on binary

def transform_bits(file):
    ba = bitarray()
    ba.fromfile(file)
    img_bytes = np.array(list(ba)).reshape(-1,8)
    tmp=[]
    for j in img_bytes:
        tmp.append(bits2token[tuple(j)])
    return tmp
def load_data(tokenizer):
    data = []
    for i in range(1):
        path = 'dataset/'+str(i)+'/'
        for j in os.listdir(path):
            if ".ipynb" in j:
                continue
            tmp = os.path.join(path,j)
            f = open(tmp, mode="rb")
            f=transform_bits(f)
            data.append(np.array(f))
    return data

def removeNestings(l):
    lis = []
    for i in l:
        if type(i) == list:
            lis+=removeNestings(i)
        else:
            lis.append(i)
    return lis



# processing training data

def split(data,split_rate = 0.8):
    l = int(len(data)*(split_rate))
    return data[:l], data[l:]
def generate_dataset(data,block_size, repeat_time = 2):
    input = []
    target = []
    for i in range(len(data)):
        for j in range(repeat_time):
            index = random.randint(0,len(data[i])-block_size-2)
            input.append(torch.from_numpy(np.array(data[i][index:index+block_size])))
            target.append(torch.from_numpy(np.array(data[i][index:index+block_size])))
    return input, target



# seq = Sequence(vocab_size,embedding_dim,hidden_size,num_layers,batch_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(seq.parameters(), lr=0.01)
# for i in range(4000-batch_size):
#     inputs = torch.stack(input[i:i+batch_size]) # fetch words for one seq length  
#     targets = torch.stack(target[i:i+batch_size])# shifted by one word from inputs
#     outputs = seq(inputs)
#     loss = criterion(outputs, targets.reshape(-1))

#     seq.zero_grad()
#     loss.backward()
        
#     #The gradients are clipped in the range [-clip_value, clip_value]. This is to prevent the exploding gradient problem
#     optimizer.step()
#     if i % 100 == 0:
#         print(f'Epoch {i}, Loss: {loss.item()}')

# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             i = random.randint(0,len(input)-batch_size-1)
#             X = torch.stack(input[i:i+batch_size])
#             Y = torch.stack(target[i:i+batch_size])
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out
# model = BigramLanguageModel()
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# for iter in range(max_iters):

#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     # sample a batch of data
#     i = random.randint(0,len(input)-batch_size-1)

#     xb = torch.stack(input[i:i+batch_size])
#     yb = torch.stack(target[i:i+batch_size])
#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long)