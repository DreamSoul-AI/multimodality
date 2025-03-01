import os
import torch
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset, Compose
from module import save, makedir_exist_ok, process_control
import json
import numpy as np


if __name__ == "__main__":
    data_names = ['DICKENS']
    with torch.no_grad():
        for data_name in data_names:
            directory = f'data/{data_name}/raw/'
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            filename = files[0] if files else None

            output_file = os.path.splitext(filename)[0]

            input_file = os.path.join(directory, filename)
            with open(input_file, 'rb') as fp:
                data = fp.read()

            print("Seq Length {}".format(len(data)))
            vals = list(set(data))
            vals.sort()
            # print(vals)

            char2id_dict = {c: i for (i, c) in enumerate(vals)}
            id2char_dict = {i: c for (i, c) in enumerate(vals)}

            # print(char2id_dict)
            # print(id2char_dict)

            out = [char2id_dict[c] for c in data]
            integer_encoded = np.array(out)

            dict_path = os.path.join('output', 'dict', data_name)
            makedir_exist_ok(dict_path)
            save(integer_encoded, os.path.join(dict_path, '{}'.format(filename)), 'np')

            param_file = os.path.join(dict_path, "params_" + os.path.splitext(filename)[0])
            params = {'char2id_dict': char2id_dict, 'id2char_dict': id2char_dict}
            with open(param_file, 'w') as f:
                json.dump(params, f, indent=4)
