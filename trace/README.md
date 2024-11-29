## Results 
Employing BPE halves both of the length of series and inference time. However, the encoding time of BPE seems too long.

| Method | Dataset            | Series Length | Inference Time | BPE training time |
| --- |--------------------| --- | --- |-------------------|
|TRACE without BPE | dickens (9,954 KB) | 10,192,446 | 835.47 seconds | NA                |
|TRACE with BPE | dickens            |  10,192,446 -> 4,696,974 | 1526.15 seconds (including 1133.62 seconds BPE encoding time) | 1117.67 seconds   |

## Dataset
https://github.com/mynotwo/A-Fast-Transformer-based-General-Purpose-LosslessCompressor/blob/main/dickens

## Key Parameter Setting
'vocab_size' = 512

## Quick Start
```bash
python compressor.py --input_dir ./dickensbpe --batch_size 512 --gpu_id 0 --prefix dickensbpe --hidden_dim 256 --ffn_dim 4096 --seq_len 8 --learning_rate 1e-3 --vocab_dim 64
```
