import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_compression_dataset
from metric import make_logger
from model import make_model
from module import var_int_decode, resume, to_device, process_control, BitInputStream, ArithmeticDecoder
import numpy as np
import shutil
import json

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint')
    cfg['result_path'] = os.path.join('output', 'result', cfg['tag'])
    output_path = os.path.join(cfg['result_path'], 'output')
    temp_dir = os.path.join(cfg['result_path'], 'temp')
    if os.path.exists(temp_dir):
        os.system("rm -r {}".format(temp_dir))
    temp_file_prefix = temp_dir + '/compressed'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    f = open(output_path + '.params', 'r')
    params = json.loads(f.read())
    f.close()

    num_chunks = params['num_chunks']
    seq_len = params['seq_len']
    len_series = params['len_series']
    id2char_dict = params['id2char_dict']
    vocab_size = 256

    # Break into multiple streams
    f = open(output_path + '.combined', 'rb')
    for i in range(num_chunks):
        f_out = open(temp_file_prefix + '.' + str(i), 'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
    f_out = open(temp_file_prefix + '.last', 'wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    series = np.zeros(len_series, dtype=np.uint8)
    l = int(len(series) / num_chunks) * num_chunks

    model = make_model(cfg['model'])
    result = resume(cfg['checkpoint_path'])
    if result is None:
        raise ValueError('No valid model, please train model first')
    cfg['step'] = result['cfg']['step']
    model = model.to(cfg['device'])
    model.load_state_dict(result['model'])

    series[:l] = decompress(model, l, num_chunks, vocab_size, seq_len, temp_file_prefix)
    if l < len_series - seq_len:
        series[l:] = decompress(model, len_series - l, 1, vocab_size, seq_len, temp_file_prefix, final_step=True)
    else:
        f = open(temp_file_prefix + '.last', 'rb')
        bitin = BitInputStream(f)
        dec = ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size) / vocab_size

        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)
        for j in range(l, len_series):
            series[j] = dec.read(cumul, vocab_size)

        bitin.close()
        f.close()

    # np.save(FLAGS.output, series)
    f = open(output_path + '.decompressed', 'wb')
    f.write(bytearray([id2char_dict[str(s)] for s in series]))
    f.close()
    shutil.rmtree(temp_dir)


def decompress(model, len_series, num_chunks, vocab_size, seq_len, temp_file_prefix, final_step=False):
    if not final_step:
        num_iters = len_series // num_chunks
        series_2d = np.zeros((num_chunks, num_iters), dtype=np.uint8).astype('int')
        ind = np.array(range(num_chunks)) * num_iters

        f = [open(temp_file_prefix + '.' + str(i), 'rb') for i in range(num_chunks)]
        bitin = [BitInputStream(f[i]) for i in range(num_chunks)]
        dec = [ArithmeticDecoder(32, bitin[i]) for i in range(num_chunks)]

        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        # Decode first K symbols in each stream with uniform probabilities
        for i in range(num_chunks):
            for j in range(min(seq_len, num_iters)):
                series_2d[i, j] = dec[i].read(cumul, vocab_size)

        cumul = np.zeros((num_chunks, vocab_size + 1), dtype=np.uint64)

        for j in (range(num_iters - seq_len)):
            # Create Batch
            train_data = torch.from_numpy(series_2d[:, j:j + seq_len])
            train_target = torch.from_numpy(series_2d[:, j + seq_len]).type(torch.LongTensor)
            input = {'data': train_data, 'target': train_target}
            input = to_device(input, cfg['device'])
            with torch.no_grad():
                model.train(False)
                output = model(**input)
                prob = torch.exp(output['pred']).detach().cpu().numpy()
            cumul[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

            # Decode with Arithmetic Encoder
            for i in range(num_chunks):
                series_2d[i, j + seq_len] = dec[i].read(cumul[i, :], vocab_size)

            if (j + 1) % 100 == 0:
                print("Step {}/{} ".format(j + 1, num_iters - seq_len), flush=True)

        # close files
        for i in range(num_chunks):
            bitin[i].close()
            f[i].close()

        return series_2d.reshape(-1)

    else:
        series = np.zeros(len_series, dtype=np.uint8).astype('int')
        f = open(temp_file_prefix + '.last', 'rb')
        bitin = BitInputStream(f)
        dec = ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(min(seq_len, len_series)):
            series[j] = dec.read(cumul, vocab_size)
        for i in range(len_series - seq_len):
            train_data = torch.from_numpy(series[i:i + seq_len].reshape(1, -1))
            train_target = torch.from_numpy(series[i]).type(torch.LongTensor)
            input = {'data': train_data, 'target': train_target}
            input = to_device(input, cfg['device'])
            with torch.no_grad():
                model.train(False)
                output = model(**input)
                prob = torch.exp(output['pred']).detach().cpu().numpy()
            cumul[1:] = np.cumsum(prob * 10000000 + 1)
            series[i + seq_len] = dec.read(cumul, vocab_size)
        bitin.close()
        f.close()
        return series


if __name__ == "__main__":
    main()
