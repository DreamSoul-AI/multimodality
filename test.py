path = "/Users/alexanderlee/Documents/编程/AI/项目/multimodality/dataset/0/0.png"

import base64
import numpy as np
from dataloader import transform_bits, transform_tokens

f = open(path, mode="rb")
f=transform_bits(f)
f = np.array(f)
f = transform_tokens(f)
out_bytes = np.packbits(f)
out_bytes.tofile('hello_level.png')
