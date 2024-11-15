## Results
### Dzip 
- Setup: Aligned with the experiment setup in TRACE, pre-training
stage of Dzip is removed to ensure fairness. Results here are reported on MacOS.
- Dataset: 20 images of MNIST 
- Compression Ratio: 0.9871713540565307 
- Additionally, Compression Ratio on hmm40.txt with Dzip-combined is about 16.
### NNCP
- Setup: Results here are reported on Windows.
- Dataset: 20 images of MNIST 
- Compression Ratio: 0.8169170428476282

## Instruction
### Dzip
```bash
cd dzip
```
- Put your input data into `files_to_be_compressed/`.
```bash
# Compress the data
bash compress.sh
# Calculate the compression ratio
python cal_ratio.py
```
### NNCP

```bash
cd nncp
```
- Modify {{input_dir}} in `compress.py`.
```bash
# Compress the data
python compress.py
# Calculate the compression ratio
python cal_ratio.py
```