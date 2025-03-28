import os
import subprocess
import requests
import zipfile


file_path = r"data/DICKENS/raw/dickens"
output_dir = "output/result/nncp/"
nncp_dir = "nncp-2024-06-05-win64/"
nncp_url = "https://bellard.org/nncp/nncp-2024-06-05-win64.zip"
nncp_zip_path = os.path.join(nncp_dir, "nncp-2024-06-05-win64.zip")

if not os.path.exists(nncp_dir):
    os.makedirs(nncp_dir, exist_ok=True)

    print("Downloading NNCP zip...")
    response = requests.get(nncp_url)
    with open(nncp_zip_path, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

    print("Extracting NNCP zip...")
    with zipfile.ZipFile(nncp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(nncp_dir)
    print("Extraction complete.")
else:
    print(f"NNCP directory '{nncp_dir}' already exists. Skipping download and extraction.")

os.makedirs(output_dir, exist_ok=True)
if os.path.isfile(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base}.nncp")
    subprocess.run([os.path.join(nncp_dir, "nncp.exe"), "--cuda", "c", file_path, output_path])
    print(f"NNCP process completed for {file_path}")

'''
NNCP version 2024-06-05, Copyright (c) 2018-2021 Fabrice Bellard
Lossless data compression with Neural Networks
usage: nncp [options] cmd args...

Commands:
c infile outfile         compress 'infile' to 'outfile'
d infile outfile         decompres 'infile' to 'outfile'
pc infile outfile        preprocessor-only encoding
pd infile outfile        preprocessor-only decoding

General options:
-h --help               show the help
-d dir                  set the debug directory
-q                      enable quiet mode
-T arg                  number of CPU threads
-p --profile arg        set the encoding profile: default, enwik8, enwik9, lstm, lstm_fast.
--max_size N            truncate the input to N symbols
--plot arg              set the plot filename
--load_coefs arg        load the model coefficients from file
--dump_interval arg     dump interval of statistics
--cuda                  enable CUDA support
--bf16 [0|1]            enable bf16 processing
--encode_only           faster encode only mode (output cannot be decompressed)
--batch_size arg        batch size
--seed arg              random number seed
--block_len arg         set the encoding block length
--train_len arg         training segment length
--lr lr0[,step0,lr1]... learning rate
--retrain_period arg    retrain period in symbols, 0 to disable retrain
--retrain_len arg       retrain length
--retrain_lr arg        retrain learning rate
--n_symb arg            vocabulary size (2 to 65535)
--preprocess n_words,min_word_freq      enable text preprocessing
--dict filename         set the dictionary filename (pc, pd, and c commands)

Options for the trf model:
--n_layer arg           number of layers
--d_model arg           model dimension
--n_head arg            number of attention heads
--d_key arg             set the attention key dimension
--d_value arg           set the attention value dimension
--mem_len arg           recurrent memory length
--d_pos arg             number of relative positions
--d_inner arg           dimension of the feed forward layer
--query_bias arg        add a query bias
--rot_pos arg           rotary position embedding
--init_range arg        initial range
--tied_embed arg        use tied embedding
--use_bias arg          use bias in matmul
--use_w_r arg           use relative pos dot products
--tied_w_r arg          use tied relative pos encodings
--tied_b_r arg          use tied relative pos bias
--ln_flags arg          layer normalisation flags
--gradient_clip arg     per parameter gradient clip value
--attn_len arg          per layer attention length
--embed_mult arg        embedding multiplier
--retrain_dropout arg   retrain dropout
--retrain_dropout_att arg       retrain dropout for the attention
--ff_act arg            feed forward activation: 0=RELU, 1=GELU, 2=GEGLU
--sparse_grad arg       use sparse gradient update

Options for the lstm model:
--n_layer arg           number of layers
--hidden_size n         number of LSTM hidden states
--cell [lstm|lstmc|lstmt]       LSTM cell variant
--full_connect [0|1]    fully connect all the layers
--n_embed_out n         number of layers in output embedding
--layer_norm [0|1]      enable layer normalization
--adam_beta1 arg        ADAM beta1 parameter
--adam_beta2 arg        ADAM beta2 parameter
--adam_eps arg          ADAM epsilon parameter
--sparse_grad arg       use sparse gradient update
'''