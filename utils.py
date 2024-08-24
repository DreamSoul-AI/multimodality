import torch
import os

from collections.abc import Iterable
from collections import Counter



def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))
 



