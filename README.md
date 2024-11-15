## Resutls
Aligned the experiment in TRACE, to ensure fairness,pre-training
stage of Dzip is removed. Compression Ratio on 20 images of MNIST with Dzip-combined is: 0.9871713540565307.
Additionally, Compression Ratio on hmm40.txt with Dzip-combined is about 16.
## Instruction
```bash
cd dzip
```
Then, put your input data into `files_to_be_compressed/`
```bash
# Compress the data
bash compress.sh
# Calculate the compression ratio
python cal_ratio.py
```
