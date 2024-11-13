import struct

f_out = open('text8_64_256_64_bs3_4_seq16.compressed_3.combined','rb')
this_byte = struct.unpack('B', f_out.read(1))[0]