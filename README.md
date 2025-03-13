# Multimodality

## Instructions

- Use `image_to_binary.py` or your own scripts to convert your dataset into a single binary file.
- Create directory `data/YOURDATA/raw` manually, and put your binary file here.
- Modify variable `data_names = ['YOURDATA']` in `make_dataset.py`, and run `make_dataset.py`. It will generate a `.npy` file to increase the robustness of the processing of the input and a `params_yourdata` file for human to read.
- Copy and paste `dataset/dickens.py` and change names related to 'dickens' to 'yourdata'. （This module obviously needs to be optimized in the future）
- Hyperparameters can be found in `config.yml` and `process_control()` in `module/hyper.py`
- Train the probability estimation model 
   ```ruby
   python train_model.py
   ```
- Compress
   ```ruby
   python compress.py
   ```
- Decompress
   ```ruby
   python decompress.py
   ```

## [Update] - 2025-03-07

### Update
- Refactor compressing and decompressing.

### Todo
- Add more models and datasets.

## [Update] - 2025-03-05

### Update
- Refactor in a new way from scratch, referring to RPipe and Dzip.
- Refactor sequence model training.

### Todo
- Refactor compressing and decompressing.


## [Update] - 2025-02-21

### Fixed
- Solve label missing error while decoding which caused decoding errors.

### Note
- TRACE can't decode binary file of images.
- TRACE can't process t10k-images-idx3-ubyte like data.

### Todo
- Integrate Dzip.
 