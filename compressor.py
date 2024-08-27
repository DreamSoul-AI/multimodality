import collections
import gzip
import os
import time
import utils
import struct
from absl import app
from absl import flags
from absl import logging
import shutil
import random
import numpy as np
import torch
import torch.nn.functional as F
from basic import BasicTokenizer
from mnist import load_mnist, load_data, split, generate_dataset
import arithmeticcoding_fast
from model import BigramLanguageModel
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["_DEVICE_ORDER"]="PCI_BUS_ID"
torch.set_printoptions(profile="full") 
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Adam Optimizer learning rate.')
flags.DEFINE_integer('hidden_dim', 64, 'Feature dimension.')
flags.DEFINE_integer('vocab_dim', 286, 'Feature dimension.')
flags.DEFINE_integer('n_layers', 4, 'Number of Attention layers.')
flags.DEFINE_integer('ffn_dim', 512, 'MLP dimension in model.')
flags.DEFINE_integer('n_heads', 4, 'Number of heads for attention.')
flags.DEFINE_integer('block_size',32, 'the length of a sequence in a sample')
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
flags.DEFINE_integer('random_seed', 0, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('print_step', 1000, 'Interval to print metrics.')
# Dataset parameters
flags.DEFINE_integer('seq_len', 8, 'Maximum sequence length (L).')
flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size of data.')
flags.DEFINE_string('input_dir', 'aaa', 'input data dir')
flags.DEFINE_string('prefix', 'text8', 'output dir')


def decode(temp_dir, compressed_file, FLAGS, len_series, last):
  
  bs = FLAGS.batch_size

  iter_num = (len_series - FLAGS.seq_len) // FLAGS.batch_size
  
  ind = np.array(range(bs))*iter_num
  print(iter_num - FLAGS.seq_len)
  series_2d = np.zeros((bs,iter_num), dtype = np.uint8).astype('int')

  f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'rb') for i in range(bs)]
  bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
  dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

  prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
  cumul = np.zeros(FLAGS.vocab_size+1, dtype = np.uint64)
  cumul[1:] = np.cumsum(prob*10000000 + 1)

  # Decode first K symbols in each stream with uniform probabilities
  for i in range(bs):
    for j in range(min(FLAGS.seq_len, iter_num)):
      series_2d[i,j] = dec[i].read(cumul, FLAGS.vocab_size)
  
  cumul_batch = np.zeros((bs, FLAGS.vocab_size+1), dtype = np.uint64)

  os.environ['_VISIBLE_DEVICES'] = FLAGS.gpu_id
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  model = BigramLanguageModel()

  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
  training_start = time.time()
  i = 0
  for train_index in range(iter_num-FLAGS.seq_len):
    if i==1:
      break
    model.train()
    train_batch = torch.LongTensor(series_2d[:, train_index:train_index + FLAGS.seq_len])
    logits,loss = model.forward(train_batch)
    print(logits.shape)
    prob = logits[:, -1, :]
    prob = F.softmax(prob, dim=1).detach().cpu().numpy()
    
    cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)

    # Decode with Arithmetic Encoder
    for i in range(bs):
      series_2d[i,train_index+FLAGS.seq_len] = dec[i].read(cumul_batch[i,:], FLAGS.vocab_size)
    
    logits = logits.transpose(1, 2)
    label = torch.from_numpy(series_2d[:, train_index+1:train_index+FLAGS.seq_len+1])
    train_loss = torch.nn.functional.cross_entropy(logits[:, :, -1], label[:, -1], reduction='mean')
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    if train_index % FLAGS.print_step == 0:
      print(train_index, ":", train_loss.item()/np.log(2))
    i+=1
  
    
  out = open('tttdecompressed_out', 'w')
  for i in range(len(series_2d)):
    out.write(utils.decode_tokens(series_2d[i]))
  
  
  for i in range(bs):
    bitin[i].close()
    f[i].close()

  if last:
    series = np.zeros(last, dtype = np.uint8).astype('int')
    f = open(temp_dir+"/"+compressed_file+'.last','rb')
    bitin = arithmeticcoding_fast.BitInputStream(f)
    dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
    prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)

    for j in range(last):
      series[j] = dec.read(cumul, FLAGS.vocab_size)
  
    print("Last decode part don't need inference.")
    out.write(utils.decode_tokens(series))
    print(utils.decode_tokens(series))
    bitin.close()
    f.close()
    return
 

def encode(temp_dir, compressed_file, FLAGS, series, train_data, test_data, iter_num):
  
  bs = FLAGS.batch_size

    
  f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'wb') for i in range(bs)]
  bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
  enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]
  
  prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
  cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
  cumul[1:] = np.cumsum(prob*10000000 + 1)
  ind = np.array(range(bs))*iter_num

  '''seq_len == block_size, like size of the sliding window'''
  for i in range(bs):
    for j in range(FLAGS.block_size):
      enc[i].write(cumul, series[ind[i]+j])
  
  cumul_batch = np.zeros((bs, FLAGS.vocab_size+1), dtype = np.uint64)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = BigramLanguageModel(FLAGS.n_heads,FLAGS.n_layers,0,FLAGS.vocab_size,FLAGS.hidden_dim,32, device)

  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
  for train_index in range(iter_num):
    model.train()
    id = random.randint(0,len(train_data)-bs-1)
    X = torch.stack(train_data[id:id+bs])
    Y = torch.stack(test_data[id:id+bs])
    y = Y[:,-1]
    logits, loss = model(X, Y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    prob = logits[:, -1, :]
    prob = F.softmax(prob, dim=1).detach().cpu().numpy()
    cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
    
    for i in range(bs):
      enc[i].write(cumul_batch[i,:], y[i])
    
    ind += 1
    if train_index % FLAGS.print_step == 0:
      size = 0
      for cf in os.listdir(temp_dir):
        size += os.path.getsize(temp_dir+"/"+cf)
      print(train_index, ":", loss.item()/np.log(2), "size:", size/(1024*1024))
  
  for i in range(bs):
    enc[i].finish()
    bitout[i].close()
    f[i].close()

  if last_train_data is not None:
    print("last series")
    f = open(temp_dir+"/"+compressed_file+'.last','wb')
    bitout = arithmeticcoding_fast.BitOutputStream(f)
    enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
    prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)
  
    for j in range(len(last_train_data)):
      enc.write(cumul, last_train_data[j])
    print("Last encode part don't need inference.")
  
    enc.finish()
    bitout.close()
    f.close()
  
  return
    
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
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len

def main(_):

  os.environ['_VISIBLE_DEVICES'] = FLAGS.gpu_id
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)
  temp_dir = "{}_{}_{}_{}_bs{}_{}_seq{}_temp".format(FLAGS.prefix, FLAGS.vocab_dim, FLAGS.hidden_dim, FLAGS.ffn_dim, FLAGS.batch_size, FLAGS.n_layers, FLAGS.seq_len)
  compressed_file = temp_dir.replace("_temp", ".compressed")
  if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)
  
  load_mnist()
  block_size = FLAGS.block_size
 
  
  tokenizer = BasicTokenizer()
  train_data = load_data(tokenizer)
  data = np.concatenate(train_data,axis = 0).tolist()
  tokenizer.train(data, 256 + 30,verbose =True)
  FLAGS.vocab_size = 286
  for i in range(len(train_data)):
    train_data[i] = tokenizer.encode(train_data[i])
  data = np.concatenate(train_data,axis=0).tolist()
  train,test = split(train_data,0.8)
  train_input,train_target = generate_dataset(train,32,3)


  vocab_size = len(tokenizer.vocab)
  total_length = len(train_data)

  encode(temp_dir, compressed_file, FLAGS, data, train_input, train_target,3000)
  
  #Combined compressed results
  f = open(compressed_file+'.combined','wb')
  for i in range(FLAGS.batch_size):
    f_in = open(temp_dir+'/'+compressed_file+'.'+str(i),'rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  
  if total_length % FLAGS.batch_size != 0:
    f_in = open(temp_dir+'/'+compressed_file+'.last','rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  f.close()
  
  total = 0
  for ff in os.listdir(temp_dir):
    total += os.path.getsize(temp_dir+'/'+ff)
  
  print(total/(1024*1024))
  
  #Remove temp file
  shutil.rmtree(temp_dir)
  
  #Decode
  os.mkdir(temp_dir)
  
  #Split compressed file
  
  f = open(compressed_file+'.combined','rb')
  len_series = len(series) 
  for i in range(FLAGS.batch_size):
    f_out = open(temp_dir+'/'+compressed_file+'.'+str(i),'wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
  
  f_out = open(temp_dir+'/'+compressed_file+'.last','wb')
  byte_str_len = var_int_decode(f)
  byte_str = f.read(byte_str_len)
  f_out.write(byte_str)
  f_out.close()
  f.close()

  len_series = len(series)
  if (len_series-FLAGS.seq_len) % FLAGS.batch_size == 0:
    decode(temp_dir, compressed_file, FLAGS, len_series, 0)
  else:
    last_length = (len_series - FLAGS.seq_len) % FLAGS.batch_size + FLAGS.seq_len
    decode(temp_dir, compressed_file, FLAGS, len_series, last_length)
  

if __name__ == '__main__':
  app.run(main)
