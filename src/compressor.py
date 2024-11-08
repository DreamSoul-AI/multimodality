import argparse
import datetime
import os
import shutil
import struct
import time
import numpy as np
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import load_mnist, load_data, split, generate_dataset, removeNestings, transform_tokens
from metric import make_logger
from model import make_model, make_optimizer, make_scheduler
from module import check, resume, to_device, process_control, arithmeticcoding_fast
import bpe.utils as utils
from bpe import BasicTokenizer
from torch.utils.data import DataLoader
from thop import profile

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)
torch.set_printoptions(profile="full")  # display the full tensor list


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        run_experiment()
    return


def run_experiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint')
    cfg['best_path'] = os.path.join(cfg['tag_path'], 'best')
    cfg['logger_path'] = os.path.join('output', 'logger', 'train', 'runs', cfg['tag'])
    load_mnist()
    tokenizer = BasicTokenizer()
    data, org_size = load_data(tokenizer, "data/mnist/0", 20)
    data_for_tokenizer = np.concatenate(data, axis=0).tolist()
    tokenizer.train(data_for_tokenizer, cfg['compressor']['vocab_size'], verbose=True)
    temp_dir = "{}_bs{}_{}_temp".format(cfg['model']['gpt1']['n_embd'],
                                        cfg['batch_size'], cfg['model']['gpt1']['n_layer'])
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    compressed_file = temp_dir.replace("_temp", ".compressed")

    '''
    data: a 2D array data[n]: the nth data. In this case, the nth tokenized image
    len_datas: size of each tokenized img data
    '''
    for i in range(len(data)):
        data[i] = np.array(tokenizer.encode(data[i]))

    datas, len_datas, max_len = padding(data, cfg['compressor']['vocab_size'])
    train_data, train_data_lens, test_data, test_data_lens = split(datas, len_datas, 0.8)

    model = make_model(cfg['model'])
    # print(model)
    result = resume(cfg['checkpoint_path'], resume_mode=cfg['resume_mode'])
    if result is None:
        cfg['step'] = 0
        model = model.to(cfg['device'])
        optimizer = make_optimizer(model.parameters(), cfg[cfg['tag']]['optimizer'])
        scheduler = make_scheduler(optimizer, cfg[cfg['tag']]['optimizer'])
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'])
    else:
        cfg['step'] = result['cfg']['step']
        model = model.to(cfg['device'])
        optimizer = make_optimizer(model.parameters(), cfg[cfg['tag']]['optimizer'])
        scheduler = make_scheduler(optimizer, cfg[cfg['tag']]['optimizer'])
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'])
        model.load_state_dict(result['model'])
        optimizer.load_state_dict(result['optimizer'])
        scheduler.load_state_dict(result['scheduler'])
        logger.load_state_dict(result['logger'])
        logger.reset()
    dataloader = DataLoader(train_data, batch_size=cfg[cfg['tag']]['optimizer']['batch_size']['train'], shuffle=True)
    train(dataloader, optimizer, model, cfg['compressor']['starter_seqlen'], cfg['model']['gpt1']['vocab_size'])

    # Encode
    file_num, enc_size = encode(test_data,
                                temp_dir,
                                compressed_file,
                                model,
                                cfg['compressor']['starter_seqlen'],
                                cfg['compressor']['vocab_size'],
                                cfg[cfg['tag']]['optimizer']['batch_size']['train'])
    f = open(compressed_file + '.combined', 'wb')  # Combined compressed results
    for i in range(file_num):
        f_in = open(temp_dir + '/' + compressed_file + '.' + str(i), 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
    shutil.rmtree(temp_dir)  # Remove temp file

    # Decode
    os.mkdir(temp_dir)
    # Split the compressed file
    f = open(compressed_file + '.combined', 'rb')
    for i in range(file_num):
        f_out = open(temp_dir + '/' + compressed_file + '.' + str(i), 'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
    # decoded_data = decode(temp_dir, compressed_file, model, cfg['starter_seqlen'], cfg['model']['gpt1'][
    # 'vocab_size'], test_data_lens, tokenizer)
    print(enc_size)
    print(org_size)
    print(org_size / enc_size)

    # while cfg['step'] < cfg['num_steps']:
    #     train(data_iterator, model, optimizer, scheduler, logger)
    #     test(data_loader['test'], model, logger)
    #     result = {'cfg': cfg, 'model': model.state_dict(),
    #               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
    #               'logger': logger.state_dict()}
    #     check(result, cfg['checkpoint_path'])
    #     if logger.compare('test'):
    #         shutil.copytree(cfg['checkpoint_path'], cfg['best_path'], dirs_exist_ok=True)
    #     logger.reset()
    return


def train(dataloader, optimizer, model, starter_seqlen, vocab_size):
    idx = 0
    batch_index = 0
    for epoch in range(1):
        for input_data in dataloader:
            cumul_batch = np.zeros((len(input_data), vocab_size + 2), dtype=np.uint64)
            model.train(True)
            input_data = to_device(input_data, cfg['device'])
            # print("input_data: ", input_data, "\n", "input_data shape: ", input_data.shape)
            output = model(input_data, starter_seqlen)
            loss = 1 / cfg['step_period'] * output['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            idx += 1
        print(f"{epoch}th epoch", ":", loss.item())


def encode(data, temp_dir, compressed_file, model, starter_seqlen, vocab_size, batch_size):
    size = 0
    idx = 0
    f = [open(temp_dir + "/" + compressed_file + '.' + str(i), 'wb') for i in range(len(data))]
    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(len(data))]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(len(data))]

    cumul = np.zeros((len(data), vocab_size + 2), dtype=np.uint64)
    print('cumul.shape: ', cumul.shape)
    prob = np.ones(vocab_size + 1) / vocab_size
    cumul[:, 1:] = np.cumsum(prob * 10000000 + 1)
    for j in range(len(data)):
        for k in range(starter_seqlen):
            enc[j].write(cumul[j], data[j, k])
    cumul_batch = np.zeros((len(data), len(data[0]) - starter_seqlen, vocab_size + 2), dtype=np.uint64)
    print('cumul_batch.shape: ', cumul_batch.shape)
    output = model(torch.tensor(data), starter_seqlen)
    logits = output['pred']
    # logits = model.forward(torch.tensor(data[:, :-1]))
    y = data[:, starter_seqlen:]

    prob = logits[:, starter_seqlen - 1:, :]
    prob = F.softmax(prob, dim=-1).detach().cpu().numpy()
    print('prob.shape: ', prob.shape)

    cumul_batch[:, :, 1:] = np.cumsum(prob * 10000000 + 1, axis=2)
    for j in range(len(y)):
        for k in range(len(y[j])):
            if y[j, k] == vocab_size:
                break
            enc[j].write(cumul_batch[j, k, :], y[j, k])

    for j in range(len(data)):
        enc[j].finish()
        bitout[j].close()
        f[j].close()
        size += os.path.getsize(temp_dir + "/" + compressed_file + '.' + str(j + idx * batch_size))
    return len(data), size


def decode(temp_dir, compressed_file, model, starter_seqlen, vocab_size, len_datas, tokenizer):
    '''seq_len == block_size, like size of the sliding window'''
    model.eval()
    decoded_data = []
    idx = 0
    for i in range(len(len_datas)):
        f = open(temp_dir + "/" + compressed_file + '.' + str(idx), 'rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size + 1) / vocab_size
        cumul = np.zeros(vocab_size + 2, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)
        series_1d = np.zeros((len_datas[i]), dtype=np.uint8).astype('int')

        for j in range(starter_seqlen):
            series_1d[j] = dec.read(cumul, vocab_size + 1)
        for j in range(0, len_datas[i] - starter_seqlen):
            cumul = np.zeros(vocab_size + 2, dtype=np.uint64)
            train_batch = series_1d[:j + starter_seqlen]
            train_batch = torch.LongTensor(train_batch).reshape(1, len(train_batch))
            logits = model.forward(train_batch)
            prob = logits[:, -1, :]

            prob = F.softmax(prob, dim=-1).detach().cpu().numpy()

            cumul[1:] = np.cumsum(prob * 10000000 + 1, axis=1)

            series_1d[j + starter_seqlen] = dec.read(cumul[:], vocab_size + 1)

        bitin.close()
        f.close()
        idx += 1
        tokenized = tokenizer.decode(series_1d)
        bits_array = transform_tokens(removeNestings(tokenized))
        out_bytes = np.packbits(bits_array)
        out_bytes.tofile(f'hello_level{i}.png')


def var_int_encode(byte_str_len, f):
    while True:
        this_byte = byte_str_len & 127
        byte_str_len >>= 7
        if byte_str_len == 0:
            f.write(struct.pack('B', this_byte))
            break
        f.write(struct.pack('B', this_byte | 128))
        byte_str_len -= 1


def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        a = f.read(1)
        if a == b'':
            break
        this_byte = struct.unpack('B', a)[0]
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


def padding(datas, vocab_size):
    len_datas = []
    max_len = 0
    for i in datas:
        len_datas.append(len(i))
        max_len = max(max_len, len_datas[-1])

    for i in range(len(datas)):
        datas[i] = np.pad(datas[i], (0, max_len - len(datas[i])), mode="constant", constant_values=vocab_size)

    return np.stack(datas, axis=0), len_datas, max_len


if __name__ == '__main__':
    main()
