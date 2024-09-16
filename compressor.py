
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
from dataloader.mnist import load_mnist, load_data, split, generate_dataset
import Arithmeticcoding.arithmeticcoding_fast as arithmeticcoding_fast
from models import BigramLanguageModel,lstm
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["_DEVICE_ORDER"]="PCI_BUS_ID"
torch.set_printoptions(profile="full") 
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('batch_size', 1, 'Batch size for training.')
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
flags.DEFINE_integer('random_seed', 0, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('print_step', 100, 'Interval to print metrics.')
# Dataset parameters
flags.DEFINE_integer('seq_len', 16, 'Maximum sequence length (L).')
flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size of data.')
flags.DEFINE_string('input_dir', 'aaa', 'input data dir')
flags.DEFINE_string('prefix', 'text8', 'output dir')


def decode(temp_dir, compressed_file, FLAGS,model, len_datas):
  print("decode")
  bs = FLAGS.batch_size
  block_size = FLAGS.block_size
  f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'rb') for i in range(len(len_datas))]
  bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(len(len_datas))]
  dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(len(len_datas))]
  
  prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
  cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
  cumul[1:] = np.cumsum(prob*10000000 + 1)
  '''seq_len == block_size, like size of the sliding window'''
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
  # Decode first K symbols in each stream with uniform probabilities
  out = open('tttdecompressed_out', 'w')
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  for i in range(len(len_datas)):
    prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)
    series_1d = np.zeros((len_datas[i]), dtype = np.uint8).astype('int')
    
    for j in range(block_size):
      series_1d[j] = dec[i].read(cumul, FLAGS.vocab_size)
    for j in range(0,len_datas[i]-block_size):
      cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
      model.train()
      train_batch = series_1d[j:j+block_size]
      train_batch = torch.LongTensor(train_batch).reshape(len(train_batch),1)
      
      logits= model.forward(train_batch)
      prob = logits[-1,:, :]
      prob = F.softmax(prob).detach().cpu().numpy()  
      cumul[1:] = np.cumsum(prob*10000000 + 1, axis=0)
    # Decode with Arithmetic Encoder
      series_1d[j+block_size] = dec[i].read(cumul[:], FLAGS.vocab_size)
      # label = torch.from_numpy(series_1d[j+1:j+FLAGS.seq_len+1])
      # logits = logits.transpose(1, 2)
      # train_loss = torch.nn.functional.cross_entropy(logits[:, :, -1],label, reduction='mean')
      # train_loss.backward()
      # optimizer.step()
      # optimizer.zero_grad(set_to_none=True)
    
    out.write(utils.decode_tokens(series_1d))
    print(series_1d)
  
  
  for i in range(len(len_datas)):
    bitin[i].close()
    f[i].close()

  
 

def encode(temp_dir, compressed_file, FLAGS, train_data,model,batch_iter):
  print(f"there are {len(train_data)} data and each data is trained {batch_iter} times")
  print("encode")
  def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))
  
  train_batch = []
  for i in range(len(train_data)):
    train_batch.append(strided_app(train_data[i], FLAGS.block_size+1, 1))
  bs = FLAGS.batch_size
  block_size = FLAGS.block_size

  f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'wb') for i in range(len(train_data))]
  bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(len(train_data))]
  enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(len(train_data))]
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
  
  
  
  '''seq_len == block_size, like size of the sliding window'''

  idx = 0
  batch_index = 0
  
  for i in range(len(train_data)):
      # batch_index = np.random.randint(0,len(series),bs)
      cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
      prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
      cumul[1:] = np.cumsum(prob*10000000 + 1)
      for j in range(block_size):
          enc[i].write(cumul,train_data[i][j])
      for train_index in range(batch_iter):

        batch = torch.tensor(train_batch[i])
        cumul_batch  = np.zeros((len(batch), FLAGS.vocab_size+1), dtype = np.uint64)
        model.train()
        train_loss, logits = model.full_loss(batch, with_grad=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        y = batch[:,-1]
        prob = logits[:, :,-1]
        prob = F.softmax(prob,dim=1).detach().cpu().numpy()
        
        
        
        
        idx += 1
        if (train_index) % FLAGS.print_step == 0:
          size = 0
          for cf in os.listdir(temp_dir):
            size += os.path.getsize(temp_dir+"/"+cf)
          print(f"{i}th data with {train_index} time", ":", train_loss.item()/np.log(2), "size:", size/(1024*1024))
      cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)

      for j in y:
        enc[i].write(cumul_batch[i,:], j)

  for i in range(len(train_data)):
    enc[i].finish()
    bitout[i].close()
    f[i].close()
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
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)
  temp_dir = "{}_{}_{}_{}_bs{}_{}_seq{}_temp".format(FLAGS.prefix, FLAGS.vocab_dim, FLAGS.hidden_dim, FLAGS.ffn_dim, FLAGS.batch_size, FLAGS.n_layers, FLAGS.seq_len)
  compressed_file = temp_dir.replace("_temp", ".compressed")
  if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)
  load_mnist()

  

  tokenizer = BasicTokenizer()
  data = load_data(tokenizer)
  data_for_tokenzier = np.concatenate(data,axis = 0).tolist()
  tokenizer.train(data_for_tokenzier, 256 + 3,verbose =True)
  FLAGS.vocab_size = 259
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = BigramLanguageModel(FLAGS.n_heads,FLAGS.n_layers,0,FLAGS.vocab_size,FLAGS.hidden_dim,FLAGS.block_size, device)
  # model = lstm(FLAGS.vocab_size,FLAGS.hidden_dim,FLAGS.ffn_dim,FLAGS.n_layers,FLAGS.block_size)
  len_datas=[]

  for i in range(1):
    data[i] = np.array(tokenizer.encode(data[i]))
    len_datas.append(len(data[i]))

  train,test = split(data,0.01)
  train = train[:1]
  len_datas = len_datas[:1]

  encode(temp_dir, compressed_file, FLAGS, train,model,600)
  #Combined compressed results
  f = open(compressed_file+'.combined','wb')
  for i in range(len(train)):
    f_in = open(temp_dir+'/'+compressed_file+'.'+str(i),'rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  model.eval()
  tmp = torch.tensor([137 , 80 , 78 , 71  ,13 , 10 , 26 , 10 ,257,  13 , 73,  72  ,68 , 82 ,257 , 28]).view(16,1)
  print(model(tmp).argmax())
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
  for i in range(len(train)):
    f_out = open(temp_dir+'/'+compressed_file+'.'+str(i),'wb')
    byte_str_len = var_int_decode(f) 
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()

  decode(temp_dir, compressed_file, FLAGS, model,len_datas)
  

if __name__ == '__main__':
  app.run(main)
