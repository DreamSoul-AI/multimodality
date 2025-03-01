from .mnist import *
from .data_utils import *
def load(dataset):
    if dataset=="mnist":
        return load_mnist()