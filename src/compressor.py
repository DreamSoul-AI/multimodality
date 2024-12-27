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
from model import make_model, make_optimizer, make_scheduler
from module import check, resume, to_device, process_control, arithmeticcoding_fast, decode_token, decode_tokens
from bpe import BasicTokenizer

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
    time_start = time.time()
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    temp_dir = "{}_{}_{}_bs{}_{}_seq{}_temp".format(cfg['model']['trace']['vocab_dim'],
                                                    cfg['model']['trace']['hidden_dim'],
                                                    cfg['model']['trace']['ffn_dim'],
                                                    cfg['batch_size'],
                                                    cfg['model']['trace']['n_layers'],
                                                    cfg['compressor']['seq_len'])
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    compressed_file = temp_dir.replace("_temp", ".compressed")
    print("compressed_file will be {}".format(compressed_file))

    def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

    ori_seq_len = cfg['compressor']['seq_len']
    scaled_seq_len = ori_seq_len * (cfg['model']['trace']['hidden_dim'] // cfg['model']['trace']['vocab_dim'])
    print("seq_len change from {} to {} due to vocab_dim = {} and hidden_dim = {}.".format(ori_seq_len,scaled_seq_len,cfg['model']['trace']['vocab_dim'], cfg['model']['trace']['hidden_dim']))


    input_dir = 'input/dickens_test'
    prefix = cfg['data_name']
    with open(input_dir, 'rb') as fp:  # , encoding='latin-1') as fp:
        series = np.frombuffer(fp.read(), dtype=np.uint8)
    print("len(series): ", len(series))
    print("series.shape: ", series.shape)
    print("type(series): ", type(series))
    print("series[0]: ", series[0])
    print("series: ", series)
    print("series.max(): ", series.max())
    tokenizer = BasicTokenizer()
    os.makedirs("models", exist_ok=True)
    prefix = os.path.join("models", prefix)

    # train BPE
    # time_train_start = time.time()
    # tokenizer.train(series, cfg['model']['trace']['vocab_size'], 2, resume=True, verbose=False)
    # time_train_finish = time.time()
    # print(f"Training BPE took {time_train_finish - time_train_start:.2f} seconds")
    # tokenizer.save(prefix)

    # encode with BPE
    # time_encode_start = time.time()
    # tokenizer.load(prefix)  # load trained model
    # series = tokenizer.encode(series)
    # series = np.asarray(series)
    # print("len(series) after BPE: ", len(series))
    # time_encode_finish = time.time()
    # print(f"Encoding with BPE took {time_encode_finish - time_encode_start:.2f} seconds")

    # arithmeic coding
    train_data = strided_app(series, scaled_seq_len + 1, 1)
    total_length = len(train_data)
    print("\nEncoding: ")
    if total_length % cfg['batch_size'] == 0:
        encode(temp_dir, compressed_file, cfg['model'], cfg[cfg['tag']]['optimizer'], cfg['batch_size'], cfg['model']['trace']['vocab_size'], scaled_seq_len, cfg['print_step'], series, train_data, None)
    else:
        l = total_length // cfg['batch_size'] * cfg['batch_size']
        encode(temp_dir, compressed_file, cfg['model'], cfg[cfg['tag']]['optimizer'], cfg['batch_size'], cfg['model']['trace']['vocab_size'], scaled_seq_len, cfg['print_step'], series[:l + scaled_seq_len], train_data[:l], series[l:])

    # Combined compressed results
    f = open(compressed_file + '.combined', 'wb')
    for i in range(cfg['batch_size']):
        f_in = open(temp_dir + '/' + compressed_file + '.' + str(i), 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()

    if total_length % cfg['batch_size'] != 0:
        f_in = open(temp_dir + '/' + compressed_file + '.last', 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
    f.close()

    time_end = time.time()
    print(f"Compression took {time_end - time_start:.2f} seconds")

    total = 0
    for ff in os.listdir(temp_dir):
        total += os.path.getsize(temp_dir + '/' + ff)

    print("total: ", total)
    # print(total/(1024*1024))

    # Remove temp file
    shutil.rmtree(temp_dir)

    # Decode
    os.mkdir(temp_dir)

    # Split compressed file

    f = open(compressed_file + '.combined', 'rb')
    len_series = len(series)
    for i in range(cfg['batch_size']):
        f_out = open(temp_dir + '/' + compressed_file + '.' + str(i), 'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()

    f_out = open(temp_dir + '/' + compressed_file + '.last', 'wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    len_series = len(series)
    print("\nDecoding: ")
    if (len_series - scaled_seq_len) % cfg['batch_size'] == 0:
        decode(temp_dir, compressed_file, cfg['model'], cfg[cfg['tag']]['optimizer'], cfg['batch_size'], cfg['model']['trace']['vocab_size'], scaled_seq_len, cfg['print_step'], len_series, 0, cfg['seed'], bpe_ckpt=prefix)
    else:
        last_length = (len_series - scaled_seq_len) % cfg['batch_size'] + scaled_seq_len
        decode(temp_dir, compressed_file, cfg['model'], cfg[cfg['tag']]['optimizer'], cfg['batch_size'], cfg['model']['trace']['vocab_size'], scaled_seq_len, cfg['print_step'], len_series, last_length, cfg['seed'], bpe_ckpt=prefix)

    return


def encode(temp_dir, compressed_file, model, optimizer, batch_size, vocab_size, seq_len, print_step, series, train_data, last_train_data):
    f = [open(temp_dir + "/" + compressed_file + '.' + str(i), 'wb') for i in range(batch_size)]
    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(batch_size)]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(batch_size)]

    prob = np.ones(vocab_size) / vocab_size
    cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob * 10000000 + 1)

    iter_num = len(train_data) // batch_size
    print("len(train_data): ", len(train_data))
    ind = np.array(range(batch_size)) * iter_num
    iter_num -= seq_len

    for i in range(batch_size):
        for j in range(seq_len):
            enc[i].write(cumul, series[ind[i] + j])

    cumul_batch = np.zeros((batch_size, vocab_size + 1), dtype=np.uint64)

    model = make_model(model)
    model = model.to(cfg['device'])
    optimizer = make_optimizer(model.parameters(), optimizer)
    print("iter_num: ", iter_num)
    for train_index in range(iter_num):
        model.train()
        train_batch = train_data[ind, :]
        y = train_batch[:, -1]
        train_batch = torch.from_numpy(train_batch).cuda().long()

        output = model(train_batch)
        logits = output['pred']
        output['loss'].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()
        # print("Shape of prob:", prob.shape)

        cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

        for i in range(batch_size):
            enc[i].write(cumul_batch[i, :], y[i])

        ind += 1
        if train_index % print_step == 0:
            size = 0
            for cf in os.listdir(temp_dir):
                size += os.path.getsize(temp_dir + "/" + cf)
            print(train_index, ": ", output['loss'].item() / np.log(2), "size: ", size)
            # print(train_index, ":", train_loss.item()/np.log(2), "size:", size/(1024*1024))

    for i in range(batch_size):
        enc[i].finish()
        bitout[i].close()
        f[i].close()

    if last_train_data is not None:
        print("last series")
        f = open(temp_dir + "/" + compressed_file + '.last', 'wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(len(last_train_data)):
            enc.write(cumul, last_train_data[j])
        print("Last encode part don't need inference.")

        enc.finish()
        bitout.close()
        f.close()

    return


def decode(temp_dir, compressed_file, model, optimizer, batch_size, vocab_size, seq_len, print_step, len_series, last, seed, bpe_ckpt):
    bs = batch_size

    iter_num = (len_series - seq_len) // batch_size

    ind = np.array(range(bs)) * iter_num
    print(iter_num - seq_len)
    series_2d = np.zeros((bs, iter_num), dtype=np.uint8).astype('int')

    f = [open(temp_dir + "/" + compressed_file + '.' + str(i), 'rb') for i in range(bs)]
    bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
    dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

    prob = np.ones(vocab_size) / vocab_size
    cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob * 10000000 + 1)

    # Decode first K symbols in each stream with uniform probabilities
    for i in range(bs):
        for j in range(min(seq_len, iter_num)):
            series_2d[i, j] = dec[i].read(cumul, vocab_size)

    cumul_batch = np.zeros((bs, vocab_size + 1), dtype=np.uint64)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = make_model(model)
    model = model.to(cfg['device'])
    optimizer = make_optimizer(model.parameters(), optimizer)

    training_start = time.time()
    for train_index in range(iter_num - seq_len):
        model.train()
        train_batch = torch.LongTensor(series_2d[:, train_index:train_index + seq_len]).cuda()
        label = torch.from_numpy(series_2d[:, train_index + 1:train_index + seq_len + 1]).long().cuda()
        extended_tensor = torch.cat((train_batch, label[:, -1].unsqueeze(1)), dim=1)
        output = model(extended_tensor)
        logits = output['pred']

        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()

        cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

        # Decode with Arithmetic Encoder
        for i in range(bs):
            series_2d[i, train_index + seq_len] = dec[i].read(cumul_batch[i, :], vocab_size)

        output['loss'].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if train_index % print_step == 0:
            print(train_index, ":", output['loss'].item() / np.log(2))

    # out = open('decompressed', 'w', encoding='utf-8')
    out = open('decompressed', 'w')
    for i in range(len(series_2d)):
        # decode with BPE
        # series_list = series_2d[i].tolist()
        # tokenizer = BasicTokenizer()
        # tokenizer.load(bpe_ckpt)  # load trained model
        # bpe_decoded_list = tokenizer.decode(series_list)
        # array = np.asarray(bpe_decoded_list)
        # out.write(decode_tokens(array))
        out.write(decode_tokens(series_2d[i]))

    for i in range(bs):
        bitin[i].close()
        f[i].close()

    if last:
        series = np.zeros(last, dtype=np.uint8).astype('int')
        f = open(temp_dir + "/" + compressed_file + '.last', 'rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(last):
            series[j] = dec.read(cumul, vocab_size)
        # print("Last series: \n", series)
        # print("Last decode part don't need inference.")

        # decode with BPE
        # series_list = series.tolist()
        # tokenizer = BasicTokenizer()
        # tokenizer.load(bpe_ckpt)  # load trained model
        # bpe_decoded_list = tokenizer.decode(series_list)
        # series = np.asarray(bpe_decoded_list)
        # print("Last bpe decoded series: \n", series)

        out.write(decode_tokens(series))
        # print(utils.decode_tokens(series))
        bitin.close()
        f.close()
        return


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


if __name__ == '__main__':
    main()
