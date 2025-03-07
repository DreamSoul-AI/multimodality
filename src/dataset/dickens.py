import codecs
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load


class DICKENS(Dataset):
    data_name = 'DICKENS'

    def __init__(self, root, batch_size, seq_len, split, process=False, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.batch_size = batch_size
        self.seq_len = seq_len
        if not check_exists(self.processed_folder) or process:
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, self.split), mode='torch')
        # print(f"Sample IDs: {self.id[:5]}")
        # print(f"Sample Data: {self.data[:5]}")
        # print(f"Sample Targets: {self.target[:5]}")
        self.other = {}
        self.data_size = load(os.path.join(self.processed_folder, 'meta'), mode='torch')

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        id, data, target = torch.tensor(self.id[index]), torch.tensor(self.data[index]), torch.tensor(
            self.target[index]).type(torch.LongTensor)
        input = {'id': id, 'data': data, 'target': target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def dict_folder(self):
        return os.path.join(self.root, 'dict')

    def process(self):
        if not check_exists(self.dict_folder):
            print("dict_folder does not exist")
        train_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train'), mode='torch')
        save(meta, os.path.join(self.processed_folder, 'meta'), mode='torch')
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        npy_files = [f for f in os.listdir(self.dict_folder) if f.endswith('.npy')]
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in '{self.dict_folder}'.")
        elif len(npy_files) > 1:
            raise ValueError(
                f"More than one .npy file found in '{self.dict_folder}'. Please specify the file or ensure only one "
                f"file exists.")
        series = np.load(os.path.join(self.dict_folder, npy_files[0]))
        series = series.reshape(-1)
        series = series.copy()
        reshaped_series = strided_app(series, self.seq_len + 1, 1)

        '''Training truncation'''
        truncating_len = int(len(reshaped_series) / self.batch_size) * self.batch_size
        truncated_reshaped_series = reshaped_series[:truncating_len]
        reshaped_series = truncated_reshaped_series

        train_data = reshaped_series[:, :-1]
        train_target = reshaped_series[:, -1]
        train_id = np.arange(len(train_data)).astype(np.int64)

        data_size = (1, self.seq_len)  # occupancy purpose
        target_size = 256  # occupancy purpose

        # print("train_data: {}, train_target: {}, data_size: {}".format(train_data.shape, train_target.shape,
        # data_size))

        return (train_id, train_data, train_target), (data_size, target_size)


def strided_app(a, L, S):  # Window len = L, Stride len = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)
