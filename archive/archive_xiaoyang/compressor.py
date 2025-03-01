import os

from absl import app
from absl import flags
import shutil
import torch
import numpy as np
from data_loader import load
from data_loader.data_utils import split, load_data
from methods import bpe
from methods.FastTransformer import CompressionTRACE
from methods.huffman import CompressionTraditional
from methods.bpe import CompressionBPE
from models.GPT_1 import BigramLanguageModel
from models.slimperformer import SLiMPerformer
import gzip
torch.manual_seed(0)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
os.environ["_DEVICE_ORDER"]="PCI_BUS_ID"
torch.set_printoptions(profile="full") 
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('batch_size', 5, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 6e-4, 'Adam Optimizer learning rate.')
flags.DEFINE_integer('hidden_dim', 64, 'Feature dimension.')
flags.DEFINE_integer('vocab_dim', 64, 'Feature dimension.')
flags.DEFINE_integer('n_layers', 4, 'Number of Attention layers.')
flags.DEFINE_integer('ffn_dim', 64, 'MLP dimension in model.')
flags.DEFINE_integer('random_seed', 0, 'Random seed for both Numpy and Torch.')
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
flags.DEFINE_float('test_split', 0.8, 'splite train and test dataset')
flags.DEFINE_integer('print_iter', 20, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('batch_iter', 10, 'Number of time to run a batch training.')
# Dataset parameters
flags.DEFINE_integer('seq_len', 16, 'Starter sequence to encode sequence')
flags.DEFINE_integer('vocab_size', 267, 'Vocabulary size of data.')
flags.DEFINE_string('input_dir', 'aaa', 'input data dir')
flags.DEFINE_string('prefix', 'text8', 'output dir')

def method(bytes_array,file_name):
    with open(file_name, "wb") as file:
        file.write(bytes_array)
    file.close()

def main(_):
  data_path='dataset/mnist/0/'
  compress_path='compressed_file/'
  decompress_path='decompressed_file/'
  compress_extentions='.bin'
  decompress_extentions='.png'
  file_name='hello'

#traditional compressor: data_path, compress_path, file_name,compress_extension
#traditional decompressor: compress_path, decompress_path, file_name,decompress_extension

#AI compressor: data,compress_path,file_name
#AI decompressor: compress_path,decompress_path,file_name,comrpess_extensions, decompress_method
  dataset_path=load("mnist")
  dataset_path=dataset_path+"0"
  datas,org_size,len_datas = load_data(dataset_path, sample_num=20)

  # a = CompressionTRACE()
  # model = SLiMPerformer(FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,FLAGS.n_layers, FLAGS.ffn_dim,FLAGS.n_heads, FLAGS.feature_type, FLAGS.compute_type)
  # a.train(FLAGS,datas,model)
  # com_size=a.compress(datas,FLAGS,compress_path,file_name, model)
  # model = SLiMPerformer(FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,FLAGS.n_layers, FLAGS.ffn_dim,FLAGS.n_heads, FLAGS.feature_type, FLAGS.compute_type)
  # a.decompress(FLAGS, compress_path,decompress_path,file_name,decompress_extentions,method, len_datas, model)

  # b=CompressionBPE()

  # b.train(FLAGS,datas,BigramLanguageModel)
  # encode_size=b.compress(FLAGS,compress_path,datas)
  # b.decompress(FLAGS,compress_path,decompress_path,method,decompress_extentions)
  # print(org_size/encode_size)
  # model = lstm(FLAGS.vocab_size,FLAGS.hidden_dim,FLAGS.ffn_dim,FLAGS.n_layers,FLAGS.block_size)

  c=CompressionTraditional(gzip.compress,gzip.decompress)
  enc_size=c.compress(data_path,'hello',compress_path,'.bin',num_files=50)
  c.decompress(compress_path,'hello',decompress_path,method,'.png',num_files=50)
  print(org_size/enc_size)

  '''
  data: a 2D array data[n]: the nth data. In this case, the nth tokenized image
  len_datas: size of each tokenzied img data
  '''
  # os.mkdir(temp_dir)

if __name__ == '__main__':
  app.run(main)
