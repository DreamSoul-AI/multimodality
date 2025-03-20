# Multimodality

## Instructions

- Use `image_binary_converter.py` or your own scripts to convert your dataset into a single binary file.
- Create directory `data/YOURDATA/raw` manually, and put your binary file here.
- Modify variable `data_names = ['YOURDATA']` in `make_dataset.py`, and run `make_dataset.py`. It will generate a `.npy` file to increase the robustness of the processing of the input and a `params_yourdata` file for human to read.
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
