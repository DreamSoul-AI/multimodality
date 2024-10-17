
import os
import time
import BPE.utils as utils
import struct
from absl import app
from absl import flags
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from BPE import BasicTokenizer
from dataloader.mnist import load_mnist
from data_utils import load_data, split, generate_dataset, removeNestings, transform_tokens
import Arithmeticcoding.arithmeticcoding_fast as arithmeticcoding_fast
from models import BigramLanguageModel,lstm
import base64
from torch.utils.data import DataLoader

torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
os.environ["_DEVICE_ORDER"]="PCI_BUS_ID"
torch.set_printoptions(profile="full") 
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('batch_size', 50, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Adam Optimizer learning rate.')
flags.DEFINE_integer('hidden_dim', 64, 'Feature dimension.')
flags.DEFINE_integer('vocab_dim', 259, 'Feature dimension.')
flags.DEFINE_integer('n_layers', 4, 'Number of Attention layers.')
flags.DEFINE_integer('ffn_dim', 128, 'MLP dimension in model.')
flags.DEFINE_integer('n_heads', 4, 'Number of heads for attention.')
flags.DEFINE_integer('block_size',16, 'the length of a sequence in a sample')
flags.DEFINE_string(
    'feature_type', 'sqr',
    'Nonlinearity function for feature. Can be relu, elu+1, sqr, favor+, or favor+{int}.'
)
flags.DEFINE_enum(
    'compute_type', 'iter', ['iter', 'ps', 'parallel_ps'],
    'Which type of method to compute: iter = iterative algorithm, ps = implementation using torch.cumsum, parallel_ps = implementation using custom log prefix sum implementation.'
)
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay for regularization.')

# Training parameters
flags.DEFINE_string('gpu_id', '0', 'ID of GPU.')
flags.DEFINE_integer('print_iter', 20, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('batch_iter', 5, 'Number of time to run a batch training.')
# Dataset parameters
flags.DEFINE_integer('starter_seqlen', 16, 'Starter sequence to encode sequence')
flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size of data.')
flags.DEFINE_string('input_dir', 'aaa', 'input data dir')
flags.DEFINE_string('prefix', 'text8', 'output dir')


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

def padding(datas):

  len_datas = []
  max_len = 0
  for i in datas:
    len_datas.append(len(i))
    max_len = max(max_len,len_datas[-1])

  for i in range(len(datas)):
    datas[i] = np.pad(datas[i],(0,max_len-len(datas[i])),mode = "constant",constant_values =FLAGS.vocab_size)

  return np.stack(datas,axis=0), len_datas, max_len

def decode(temp_dir, compressed_file, FLAGS,model, len_datas, tokenizer ):

  
  '''seq_len == block_size, like size of the sliding window'''
  model.eval()
  decoded_data = []
  idx = 0
  for i in range(len(len_datas)):
    f = open(temp_dir+"/"+compressed_file+'.'+str(idx),'rb') 
    bitin = arithmeticcoding_fast.BitInputStream(f)
    dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
    prob = np.ones(FLAGS.vocab_size+1)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+2, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)
    series_1d = np.zeros((len_datas[i]), dtype = np.uint8).astype('int')
    
    for j in range(FLAGS.starter_seqlen):
      series_1d[j] = dec.read(cumul, FLAGS.vocab_size+1)
    for j in range(0,len_datas[i]-FLAGS.starter_seqlen):
      cumul = np.zeros(FLAGS.vocab_size+2, dtype=np.uint64)
      train_batch = series_1d[:j+FLAGS.starter_seqlen]
      train_batch = torch.LongTensor(train_batch).reshape(1,len(train_batch))
      logits= model.forward(train_batch)
      prob = logits[:,-1,:]

      prob = F.softmax(prob,dim=-1).detach().cpu().numpy()

      cumul[1:] = np.cumsum(prob*10000000 + 1, axis=1)
      
      series_1d[j+FLAGS.starter_seqlen] = dec.read(cumul[:], FLAGS.vocab_size+1)
    
    bitin.close()
    f.close()
    idx+=1
    tokenized=tokenizer.decode(series_1d)
    bits_array = transform_tokens(removeNestings(tokenized))
    out_bytes = np.packbits(bits_array)
    out_bytes.tofile(f'hello_level{i}.png')
  

def train(dataloader,FLAGS,model):
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))

  idx = 0
  batch_index = 0
  for epoch in range(FLAGS.batch_iter):
    for X_batch in dataloader:
      cumul_batch  = np.zeros((len(X_batch), FLAGS.vocab_size+2), dtype = np.uint64)
      model.train()
      train_loss, logits = model.full_loss(X_batch, with_grad=True)
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      idx += 1
    if epoch%FLAGS.print_iter == 0:
        print(f"{epoch}th epoch", ":", train_loss.item())
  
  

def encode(data, temp_dir, compressed_file, FLAGS, model):
  size=0
  idx = 0
  f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'wb') for i in range(len(data))]
  bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(len(data))]
  enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(len(data))]
    
  
  cumul = np.zeros((len(data),FLAGS.vocab_size+2), dtype=np.uint64)
  prob = np.ones(FLAGS.vocab_size+1)/FLAGS.vocab_size
  cumul[:,1:] = np.cumsum(prob*10000000 + 1)
  for j in range(len(data)):
      for k in range(FLAGS.starter_seqlen):
          enc[j].write(cumul[j],data[j,k])
  cumul_batch  = np.zeros((len(data), len(data[0])-FLAGS.starter_seqlen,FLAGS.vocab_size+2), dtype = np.uint64)
  logits = model.forward(torch.tensor(data[:,:-1]))
  y = data[:,FLAGS.starter_seqlen:]
  
  prob = logits[:, FLAGS.starter_seqlen-1:,:]
  prob = F.softmax(prob, dim=-1).detach().cpu().numpy()

  cumul_batch[:,:,1:] = np.cumsum(prob*10000000 + 1, axis = 2)
  for j in range(len(y)):
    for k in range(len(y[j])):
        if y[j,k]==FLAGS.vocab_size:
          break
        enc[j].write(cumul_batch[j,k,:],y[j,k])

  for j in range(len(data)):

      enc[j].finish()
      bitout[j].close()
      f[j].close()
      size += os.path.getsize(temp_dir+"/"+compressed_file+'.'+str(j+idx*FLAGS.batch_size))
  return len(data), size
    
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

def main(_):
  os.environ['_VISIBLE_DEVICES'] = FLAGS.gpu_id
  temp_dir = "{}_{}_{}_{}_bs{}_{}_seq{}_temp".format(FLAGS.prefix, FLAGS.vocab_dim, FLAGS.hidden_dim, FLAGS.ffn_dim, FLAGS.batch_size, FLAGS.n_layers, FLAGS.starter_seqlen)
  compressed_file = temp_dir.replace("_temp", ".compressed")
  if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)
  load_mnist()
  tokenizer = BasicTokenizer()
  data,org_size = load_data(tokenizer,"dataset/mnist/0",20)
  data_for_tokenzier = np.concatenate(data,axis = 0).tolist()
  tokenizer.train(data_for_tokenzier, FLAGS.vocab_size,verbose =False)
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  
  # model = lstm(FLAGS.vocab_size,FLAGS.hidden_dim,FLAGS.ffn_dim,FLAGS.n_layers,FLAGS.block_size)
  '''data: a 2D array data[n]: the nth data. In this case, the nth tokenized image
  len_datas: size of each tokenzied img data
  '''


  for i in range(len(data)):
    data[i] = np.array(tokenizer.encode(data[i]))

  datas, len_datas, max_len = padding(data)
  train_data,train_data_lens, test_data, test_data_lens = split(datas, len_datas,0.8)
  model = BigramLanguageModel(FLAGS.n_heads,FLAGS.n_layers,0,FLAGS.vocab_size+1,FLAGS.hidden_dim,max_len, FLAGS.starter_seqlen,device)
  dataloader = DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True)
  train(dataloader,FLAGS,model)
  file_num, enc_size = encode(test_data,temp_dir, compressed_file, FLAGS, model)
  #Combined compressed results
  f = open(compressed_file+'.combined','wb')
  for i in range(file_num):
    f_in = open(temp_dir+'/'+compressed_file+'.'+str(i),'rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  
  
  

  #Remove temp file
  shutil.rmtree(temp_dir)
  
  #Decode
  os.mkdir(temp_dir)
  
  #Split compressed file
  
  f = open(compressed_file+'.combined','rb')
  for i in range(file_num):
    f_out = open(temp_dir+'/'+compressed_file+'.'+str(i),'wb')
    byte_str_len = var_int_decode(f) 
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()

  # decoded_data = decode(temp_dir, compressed_file, FLAGS, model,test_data_lens, tokenizer)
  print(enc_size)
  print(org_size)
  print(org_size/enc_size)





if __name__ == '__main__':
  app.run(main)
