import os
import struct
import shutil
from data_loader.data_utils import removeNestings
import numpy as np
import torch
import torch.nn.functional as F
import Arithmeticcoding.arithmeticcoding_fast as arithmeticcoding_fast
from data_loader.data_utils import transform_tokens,bitstring_to_bytes
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
torch.set_printoptions(profile="full") 
torch.manual_seed(0)

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))
class CompressionTRACE:
  def __init__(self):
    pass
  def decode(self,FLAGS, len_series, last_length,model, temp_file, decompress_path, file_name, decompress_extensions, method):

    bs = FLAGS.batch_size
    
    iter_num = (len_series - FLAGS.seq_len) // FLAGS.batch_size
    ind = np.array(range(bs))*iter_num
    # print(iter_num - FLAGS.seq_len)
    series_2d = np.zeros((bs,iter_num), dtype = np.uint8).astype('int')

    f = [open(temp_file+'.'+str(i),'rb') for i in range(bs)]
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

    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
    


    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
    for train_index in range(iter_num-FLAGS.seq_len):
      model.train()
      train_batch = torch.LongTensor(series_2d[:, train_index:train_index + FLAGS.seq_len])

      logits = model.forward(train_batch)
      prob = logits[:, -1, :]
      prob = F.softmax(prob, dim=1).detach().cpu().numpy()
      cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)

      for i in range(bs):
        series_2d[i,train_index+FLAGS.seq_len] = dec[i].read(cumul_batch[i,:], FLAGS.vocab_size)
      logits = logits.transpose(1, 2)
      label = torch.from_numpy(series_2d[:, train_index+1:train_index+FLAGS.seq_len+1])
      train_loss = torch.nn.functional.cross_entropy(logits[:, :, -1], label[:, -1], reduction='mean')
      train_loss.backward()
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      

      
    series1d = []
    for i in range(len(series_2d)):
      series1d.extend(series_2d[i])

    
    for i in range(bs):
      bitin[i].close()
      f[i].close()
    series = np.zeros(last_length+1, dtype = np.uint8).astype('int')
    # last = len_series%FLAGS.batch_size

    if last_length and len_series%FLAGS.batch_size!=0:
      f = open(temp_file+'.last','rb')
      bitin = arithmeticcoding_fast.BitInputStream(f)
      dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
      prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
      cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
      cumul[1:] = np.cumsum(prob*10000000 + 1)
      for j in range(last_length+1):
        
        series[j] = dec.read(cumul, FLAGS.vocab_size)

      bitin.close()
      f.close()
    series1d.extend(series)
    series1d=transform_tokens(series1d)
    
    series1d=bitstring_to_bytes(''.join([str(x) for x in series1d]))
    method(series1d,decompress_path+file_name+decompress_extensions)
  

  def encode(self,tmp_path,tmp_filename,FLAGS, series,train_data,last_train_data,model):
    bs = FLAGS.batch_size
    f = [open(tmp_path+tmp_filename+'.'+str(i),'wb') for i in range(bs)]
    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]
    prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)

    iter_num = len(train_data) // FLAGS.batch_size


    ind = np.array(range(bs))*iter_num
    
    iter_num -= FLAGS.seq_len
    for i in range(bs):
      for j in range(FLAGS.seq_len):
        enc[i].write(cumul, series[ind[i]+j])
    cumul_batch = np.zeros((bs, FLAGS.vocab_size+1), dtype = np.uint64)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
    for train_index in range(iter_num):
      model.train()
      train_batch = train_data[ind, :]

      y = train_batch[:, -1]
      train_batch = torch.from_numpy(train_batch).long()
      train_loss, logits = model.full_loss(train_batch, with_grad=True)
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      
      logits = logits.transpose(1, 2)
      prob = logits[:, -1, :]
      prob = F.softmax(prob, dim=1).detach().cpu().numpy()

      cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
      
      for i in range(bs):
        enc[i].write(cumul_batch[i,:], y[i])

      
      ind += 1
      if train_index % FLAGS.print_iter == 0:
        size = 0
        for cf in os.listdir(tmp_path):
          size += os.path.getsize(tmp_path+cf)
        # print(train_index, ":", train_loss.item()/np.log(2), "size:", size/(1024*1024))
    
    for i in range(bs):
      enc[i].finish()
      bitout[i].close()
      f[i].close()
    if last_train_data is not None:
      # print("last series")
      f = open(tmp_path+tmp_filename+'.last','wb')
      bitout = arithmeticcoding_fast.BitOutputStream(f)
      enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
      prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
      cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
      cumul[1:] = np.cumsum(prob*10000000 + 1)
      
      for j in range(len(last_train_data)):
        enc.write(cumul, last_train_data[j])
      # print("Last encode part don't need inference.")
    
      enc.finish()
      bitout.close()
      f.close()
    
    
    return
  def train(self,FLAGS,datas,model):
    pass
  def compress_single(self,data,FLAGS,file_path,model):
    tmp_filename='hello'+file_path[-1]
    tmp_path='tmp/'
    if not os.path.exists(tmp_path):
      os.mkdir(tmp_path)
    total_length = len(data)
    train_data = strided_app(data, FLAGS.seq_len+1, 1)
    
    if total_length % FLAGS.batch_size == 0:
      self.encode(tmp_path,tmp_filename, FLAGS, data, train_data, None,model)
    else:
      l = total_length // FLAGS.batch_size * FLAGS.batch_size
      self.encode(tmp_path,tmp_filename,FLAGS, data[:l+FLAGS.seq_len], train_data[:l], data[l-FLAGS.seq_len:],model)
    
    
    #Combined compressed results

    f = open(file_path+'.combined','wb')
    for i in range(FLAGS.batch_size):
    
      f_in = open(tmp_path+tmp_filename+'.'+str(i),'rb')
      byte_str = f_in.read()
      byte_str_len = len(byte_str)
      var_int_encode(byte_str_len, f)
      f.write(byte_str)
      f_in.close()
    
    if total_length % FLAGS.batch_size != 0:
      f_in = open(tmp_path+tmp_filename+'.last','rb')
      byte_str = f_in.read()
      byte_str_len = len(byte_str)
      var_int_encode(byte_str_len, f)
      f.write(byte_str)
      f_in.close()
    f.close()
    
    total = 0
    for ff in os.listdir(tmp_path):
      if ff=='.DS_Store':
        continue
      total += os.path.getsize(tmp_path+ff)
    # print(total/1024/1024)
    shutil.rmtree(tmp_path)
    return total/(1024*1024)
    
    #Remove temp file
  def compress(self,data,FLAGS, compress_path,file_name,model):
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)
    total=0
    if not os.path.exists(compress_path):
        os.mkdir(compress_path)
    for i in range(len(data)):
      total+=self.compress_single(data[i],FLAGS, compress_path+file_name+str(i),model)
    return total
  
  def decompress_single(self,FLAGS, len_series,input_file,decompress_path,file_name,decompress_extensions,method,model):
    # The model needs to be reinitialize 
    temp_dir='tmp/'
    if not os.path.exists(temp_dir):
      os.mkdir(temp_dir)
    f = open(input_file,'rb')
    for i in range(FLAGS.batch_size):
      f_out = open(temp_dir+file_name+'.'+str(i),'wb')
      byte_str_len = var_int_decode(f)
      byte_str = f.read(byte_str_len)
      f_out.write(byte_str)
      f_out.close()
    
    if len_series%FLAGS.batch_size!=0:
      f_out = open(temp_dir+file_name+'.last','wb')
      byte_str_len = var_int_decode(f)
      byte_str = f.read(byte_str_len)
      f_out.write(byte_str)
      f_out.close()
      f.close()
    if (len_series-FLAGS.seq_len) % FLAGS.batch_size == 0:
      self.decode(FLAGS, len_series,0,model,temp_dir+file_name,decompress_path,file_name,decompress_extensions,method)
    else:
      last_length = (len_series - FLAGS.seq_len) % FLAGS.batch_size + FLAGS.seq_len
      self.decode(FLAGS, len_series,last_length,model,temp_dir+file_name,decompress_path,file_name,decompress_extensions,method)
    shutil.rmtree(temp_dir)
  def decompress(self,FLAGS, compress_path,decompress_path,file_name,decompress_extensions,method,len_datas,model):
    c=0
    if not os.path.exists(decompress_path):
        os.mkdir(decompress_path)
    for i in reversed(os.listdir(compress_path)):
      if i=='.DS_Store':
        continue
      base_name,_=os.path.splitext(i)
      self.decompress_single(FLAGS, len_datas[c],compress_path+i,decompress_path,base_name,decompress_extensions,method,model)
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
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S

    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

  
