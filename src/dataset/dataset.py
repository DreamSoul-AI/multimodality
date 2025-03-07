import dataset
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from module import apply_recursively
from config import cfg
import json
import glob


def make_dataset(data_name, batch_size=2048, seq_len=64, transform=True, process=False, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)

    if data_name in ['DICKENS']:
        dataset_['train'] = eval('dataset.{}(root=root, batch_size=batch_size, seq_len=seq_len, split="train", '
                                 'process=process)'.format(data_name))
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def make_compression_dataset(data_name, output_path, num_chunks, seq_len, verbose=True):
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    dict_folder = os.path.join(root, 'dict')

    npy_files = [f for f in os.listdir(dict_folder) if f.endswith('.npy')]
    series = np.load(os.path.join(dict_folder, npy_files[0]))
    series = series.reshape(-1)
    series = series.copy()

    params_file_name = next(f for f in os.listdir(dict_folder) if f.startswith('params_'))
    params_file_path = os.path.join(dict_folder, params_file_name)
    with open(params_file_path, 'r') as f:
        params = json.load(f)
    params['len_series'] = len(series)
    params['num_chunks'] = num_chunks
    params['seq_len'] = seq_len
    with open(output_path+'.params','w') as f:
        json.dump(params, f, indent=4)

    reshaped_series = strided_app(series, seq_len + 1, 1)

    train_data = reshaped_series[:, :-1]
    train_target = reshaped_series[:, -1]
    truncating_len = int(len(reshaped_series) / num_chunks) * num_chunks

    dataset_ = {'series': series, 'train_data': train_data, 'train_target': train_target, 'length': params['len_series'], 'truncating_len': truncating_len}

    if verbose:
        print('data ready')
    return dataset_


def input_collate(input):
    def add_(input_, key=None):
        split_names = key.split('.')
        current = batch
        for split_name in split_names[:-1]:
            if split_name not in current:
                current[split_name] = {}
            current = current[split_name]
        if split_names[-1] not in current:
            current[split_names[-1]] = input_.unsqueeze(0)
        else:
            current[split_names[-1]] = torch.cat([current[split_names[-1]], input_.unsqueeze(0)], dim=0)
        return

    batch = {}
    apply_condition = lambda x: isinstance(x, torch.Tensor)
    identity_condition = lambda x: isinstance(x, (str, type(None)))
    for i in range(len(input)):
        input_i = input[i]
        # print("i: {}, input_i: {}".format(i, input_i))
        apply_recursively(add_, input_i, apply_condition=apply_condition, identity_condition=identity_condition)
    return batch


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, batch_size, num_steps=None, step=0, step_period=1, pin_memory=True,
                     num_workers=0, collate_mode='dict', seed=0, shuffle=True):
    data_loader = {}
    for k in dataset:
        if k == 'train' and num_steps is not None:
            num_samples = batch_size[k] * (num_steps - step) * step_period
            if num_samples > 0:
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = torch.utils.data.RandomSampler(dataset[k], replacement=False, num_samples=num_samples,
                                                         generator=generator)
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], sampler=sampler,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
        else:
            if k == 'train':
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=shuffle,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
            else:
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=False,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
    return data_loader


def process_dataset(dataset):
    processed_dataset = dataset
    cfg['num_samples'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    cfg['model']['data_size'] = dataset['train'].data_size
    if 'num_epochs' in cfg:
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs']
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size']))
        cfg[cfg['tag']]['optimizer']['num_steps'] = cfg['num_steps']
    return processed_dataset


def strided_app(a, L, S):  # Window len = L, Stride len = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)
