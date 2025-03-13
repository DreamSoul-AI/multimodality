import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_compression_dataset
from metric import make_logger
from model import make_model
from module import var_int_encode, resume, to_device, process_control, BitOutputStream, ArithmeticEncoder
import numpy as np
import shutil

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

    model = make_model(cfg['model'])
    result = resume(cfg['checkpoint_path'])
    if result is None:
        raise ValueError('No valid model, please train model first')
    cfg['step'] = result['cfg']['step']
    model = model.to(cfg['device'])
    model.load_state_dict(result['model'])

    dataset = make_compression_dataset(cfg['data_name'], output_path, cfg['num_chunks'], cfg['seq_len'])
    series = dataset['series']
    train_data = dataset['train_data']
    train_target = dataset['train_target']
    length = dataset['length']
    truncating_len = dataset['truncating_len']

    compress(model, train_data, train_target, cfg['num_chunks'], cfg['vocab_size'], cfg['seq_len'], temp_file_prefix)
    if truncating_len < length - cfg['seq_len']:
        compress(model, train_data[truncating_len:], train_target[truncating_len:], 1, cfg['vocab_size'],
                 cfg['seq_len'], temp_file_prefix, final_step=True)
    else:
        f = open(temp_file_prefix + '.last', 'wb')
        bitout = BitOutputStream(f)
        enc = ArithmeticEncoder(32, bitout)
        prob = np.ones(cfg['vocab_size']) / cfg['vocab_size']

        cumul = np.zeros(cfg['vocab_size'] + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)
        for j in range(truncating_len, length):
            enc.write(cumul, series[j])
        enc.finish()
        bitout.close()
        f.close()

    print("Done")

    # combine files into one file
    f = open(output_path + '.combined', 'wb')
    for i in range(cfg['num_chunks']):
        f_in = open(temp_file_prefix + '.' + str(i), 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
    f_in = open(temp_file_prefix + '.last', 'rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
    f.close()
    shutil.rmtree(temp_dir)

    return


def compress(model, X, Y, num_chunks, vocab_size, seq_len, temp_file_prefix, final_step=False):
    if not final_step:
        num_iters = (len(X) + seq_len) // num_chunks
        index = np.array(range(num_chunks)) * num_iters

        f = [open(temp_file_prefix + '.' + str(i), 'wb') for i in range(num_chunks)]
        bitout = [BitOutputStream(f[i]) for i in range(num_chunks)]
        enc = [ArithmeticEncoder(32, bitout[i]) for i in range(num_chunks)]

        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        # Encode first K symbols in each stream with uniform probabilities
        for i in range(num_chunks):
            for j in range(min(seq_len, num_iters)):
                enc[i].write(cumul, X[index[i], j])

        cumul = np.zeros((num_chunks, vocab_size + 1), dtype=np.uint64)

        for j in (range(num_iters - seq_len)):
            train_data = torch.from_numpy(X[index, :])
            train_target = torch.from_numpy(Y[index]).type(torch.LongTensor)
            input = {'data': train_data, 'target': train_target}
            input = to_device(input, cfg['device'])
            with torch.no_grad():
                model.train(False)
                output = model(**input)
                prob = torch.exp(output['pred']).detach().cpu().numpy()
            cumul[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

            # Encode with Arithmetic Encoder
            for i in range(num_chunks):
                enc[i].write(cumul[i, :], Y[index[i]])
            index = index + 1

            if (j + 1) % 100 == 0:
                print("Step {}/{} ".format(j + 1, num_iters - seq_len), flush=True)

        # close files
        for i in range(num_chunks):
            enc[i].finish()
            bitout[i].close()
            f[i].close()

    else:
        f = open(temp_file_prefix + '.last', 'wb')
        bitout = BitOutputStream(f)
        enc = ArithmeticEncoder(32, bitout)
        prob = np.ones(vocab_size) / vocab_size
        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(seq_len):
            enc.write(cumul, X[0, j])
        for i in (range(len(X))):
            train_data = torch.from_numpy(X[i:i + 1, :])
            train_target = torch.from_numpy(Y[i:i + 1]).type(torch.LongTensor)
            input = {'data': train_data, 'target': train_target}
            input = to_device(input, cfg['device'])
            with torch.no_grad():
                model.train(False)
                output = model(**input)
                prob = torch.exp(output['pred']).detach().cpu().numpy()
            cumul[1:] = np.cumsum(prob * 10000000 + 1)
            enc.write(cumul, Y[i])
        enc.finish()
        bitout.close()
        f.close()

    return


if __name__ == "__main__":
    main()
