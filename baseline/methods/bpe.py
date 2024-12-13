import os
import numpy as np
import torch
import torch.nn.functional as F
from data_loader.data_utils import removeNestings, transform_tokens,tokenize,padding,bitstring_to_bytes
import Arithmeticcoding.arithmeticcoding_fast as arithmeticcoding_fast
from BPE import BasicTokenizer
import struct
import shutil
from models import train
from data_loader.data_utils import split
from torch.utils.data import DataLoader
class CompressionBPE:
   
    def __init__(self):
        self.tokenizer = BasicTokenizer()
        self.file_length={}
    def encode(self,data, compress_file, FLAGS, model):
        f = open(compress_file,'wb')
        self.file_length[compress_file]=len(data)
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
            
        
        cumul = np.zeros((FLAGS.vocab_size+2), dtype=np.uint64)
        prob = np.ones(FLAGS.vocab_size+1)/FLAGS.vocab_size
        cumul[1:] = np.cumsum(prob*10000000 + 1)
        for k in range(FLAGS.seq_len):
            enc.write(cumul,data[k])
        cumul_batch  = np.zeros((len(data)-FLAGS.seq_len,FLAGS.vocab_size+2), dtype = np.uint64)
        train_batch = torch.LongTensor(data[:-1]).reshape(1,len(data[:-1]))
        logits = model.forward(torch.tensor(train_batch))
        y = data[FLAGS.seq_len:]
        prob = logits[0,FLAGS.seq_len-1:,:]
        
        prob = F.softmax(prob, dim=-1).detach().cpu().numpy()
        cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
        for j in range(len(y)):

            enc.write(cumul_batch[j,:],y[j])


        enc.finish()
        bitout.close()
        f.close()
        return os.path.getsize(compress_file)

    def decode(self,compress_file, decompress_file, FLAGS, method,decompress_extensions ):

    
        '''seq_len == block_size, like size of the sliding window'''
        self.model.eval()
        f = open(compress_file,'rb') 
        len_data=self.file_length[compress_file]
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(FLAGS.vocab_size+1)/FLAGS.vocab_size
        cumul = np.zeros(FLAGS.vocab_size+2, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)
        series_1d = np.zeros((len_data), dtype = np.uint8).astype('int')
        
        for j in range(FLAGS.seq_len):
            series_1d[j] = dec.read(cumul, FLAGS.vocab_size)
        for j in range(0,len_data-FLAGS.seq_len):
            cumul = np.zeros(FLAGS.vocab_size+2, dtype=np.uint64)
            train_batch = series_1d[:j+FLAGS.seq_len]

            train_batch = torch.LongTensor(train_batch).reshape(1,len(train_batch))
        
            logits= self.model.forward(train_batch)

            prob = logits[0,-1,:]
            prob = F.softmax(prob,dim=-1).detach().cpu().numpy()
            
            cumul[1:] = np.cumsum(prob*10000000 + 1, axis=0)
            
            series_1d[j+FLAGS.seq_len] = dec.read(cumul[:], FLAGS.vocab_size)
        
        bitin.close()
        f.close()
        series1d=self.tokenizer.decode(series_1d)
        series1d = transform_tokens(removeNestings(series1d))
        series1d=bitstring_to_bytes(''.join([str(x) for x in series1d]))
        method(series1d,decompress_file+"hello"+decompress_extensions)
    def train(self,FLAGS,datas,model):
        datas = tokenize(self.tokenizer,datas,FLAGS.vocab_size)
        datas,max_len=padding(datas,FLAGS.vocab_size)
        self.train_data,self.test_data = split(datas, FLAGS.test_split)
        self.model = model(FLAGS.n_heads,FLAGS.n_layers,0,FLAGS.vocab_size+1,FLAGS.hidden_dim,max_len, FLAGS.seq_len)
        train_dataloader = DataLoader(self.train_data, batch_size=FLAGS.batch_size, shuffle=True)
        train(train_dataloader,FLAGS,self.model)

        
    def compress(self,FLAGS, compress_path,datas):
        if not os.path.exists(compress_path):
            os.mkdir(compress_path)
        for i in range(len(datas)):
            encode_size = self.encode(datas[i],compress_path+'_'+str(i)+'.bin', FLAGS, self.model)
        #Combined compressed results
        return encode_size/1024/1024

    def decompress(self,FLAGS, compress_path,decompress_path,method,decompress_extentions):
        if not os.path.exists(decompress_path):
            os.mkdir(decompress_path)
        c=0
        for i in os.listdir(compress_path):
            if i=='.DS_Store':
                continue
            self.decode(compress_path+i,decompress_path+str(c),FLAGS,method,decompress_extentions)
            c+=1
        
def var_int_encode(byte_str_len, f):
  while True:
    this_byte = byte_str_len&127
    byte_str_len >>= 7
    if byte_str_len == 0:
      f.write(struct.pack('B',this_byte))
      break
    f.write(struct.pack('B',this_byte|128))
    byte_str_len -= 1

def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        a = f.read(1)
        if a==b'':
          break
        this_byte = struct.unpack('B', a)[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len