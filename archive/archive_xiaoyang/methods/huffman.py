
import os

from data_loader.data_utils import transform_tokens
import zlib
import numpy as np
import bz2
import gzip
"""
author: Bhrigu Srivastava
website: https:bhrigu.me
"""
def method(bytes_array,file_name):
    with open(file_name, "wb") as file:
        file.write(bytes_array)
    file.close()

class CompressionTraditional:
    def __init__(self,compress,decompress):
        self.compress_method=compress
        self.decompress_method=decompress

    def compress(self,input_path,file_name,target_path,compress_extension,num_files=None):
        c=0
        total=0
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        if num_files is None:
            num_files=len(os.listdir(input_path))
        for i in os.listdir(input_path):
            tmp = open(input_path+i,mode="rb")
            tmp = tmp.read()
            tmp=self.compress_method(tmp)
            with open(target_path+file_name+str(c)+compress_extension, 'wb') as file:
                file.write(tmp)
                file.close()
            total+=os.path.getsize(target_path+file_name+str(c)+compress_extension)
            if c==num_files:
                break
            c+=1
        return total/1024/1024
    def decompress(self,input_path,file_name,target_path,bytes_to_file,decompress_extension,num_files=None):
        c=0
        total=0
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        if num_files is None:
            num_files=len(os.listdir(input_path))
        for i in os.listdir(input_path):
            if i=='.DS_Store':
                continue
            with open(input_path+i, 'rb') as file:
                tmp=file.read()
                tmp=self.decompress_method(tmp)
                bytes_to_file(tmp,target_path+file_name+str(c)+decompress_extension)
                file.close()
            total+=os.path.getsize(target_path+file_name+str(c)+decompress_extension)
            if c==num_files:
                break
            c=c+1
        return total








